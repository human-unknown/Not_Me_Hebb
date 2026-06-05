"""
audio_io.py — 真实音频输入/输出 [v5.3]

提供真实音频输入支持，替换 v5.2 语义代理模式。
支持音频文件加载 (WAV/MP3/FLAC) 和麦克风实时录制，
计算 mel 频谱以匹配耳蜗核的 32 通道 tonotopic 编码。

功能:
  - AudioInput.from_file(): 加载音频文件 → mel 频谱
  - AudioInput.from_mic(): 麦克风录制 → mel 频谱
  - compute_mel_spectrogram(): 底层 mel 频谱计算
  - 立体声支持: 独立左/右耳频谱 (双耳定位)

在 NotMe 中的应用:
  - 替代 CochlearNucleus.semantic_to_pseudo_spectrum() (语义代理)
  - 为 auditory_hierarchy.process() 提供真实的 spectrum 参数
  - 双耳立体声 → SOC ITD/ILD 双耳定位获得真实空间线索
"""

import numpy as np
from typing import Optional, Tuple


# ============================================================
# Mel 频谱计算 (匹配 CochlearNucleus 32 通道格式)
# ============================================================

def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    """Hz → mel scale."""
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    """mel scale → Hz."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(n_fft: int, sample_rate: float,
                    n_mels: int = 32,
                    f_min: float = 80.0,
                    f_max: float = 8000.0) -> np.ndarray:
    """生成 mel 滤波器组矩阵 (n_freq_bins, n_mels).

    与 CochlearNucleus._mel_filterbank() 使用相同的频率范围
    和通道数，确保频谱格式完全兼容。

    Args:
        n_fft: FFT 点数
        sample_rate: 采样率 (Hz)
        n_mels: mel 通道数 (默认 32)
        f_min: 最低频率 (Hz)
        f_max: 最高频率 (Hz)

    Returns:
        filters: (n_freq_bins, n_mels) 滤波器矩阵
    """
    n_freq_bins = n_fft // 2 + 1

    # mel 空间等间距中心频率
    mel_min = _hz_to_mel(np.array([f_min]))[0]
    mel_max = _hz_to_mel(np.array([f_max]))[0]
    mel_centers = np.linspace(mel_min, mel_max, n_mels)
    hz_centers = _mel_to_hz(mel_centers)

    # 线性频率 bin 中心
    freq_bins = np.linspace(0, sample_rate / 2, n_freq_bins)

    filters = np.zeros((n_freq_bins, n_mels), dtype=np.float32)

    for i in range(n_mels):
        lo = hz_centers[i - 1] if i > 0 else hz_centers[0] / 2
        hi = hz_centers[i + 1] if i < n_mels - 1 else hz_centers[-1] * 2
        center = hz_centers[i]

        for j, f in enumerate(freq_bins):
            if f <= lo or f >= hi:
                continue
            if f <= center:
                filters[j, i] = (f - lo) / (center - lo + 1e-8)
            else:
                filters[j, i] = (hi - f) / (hi - center + 1e-8)

    # 归一化每个滤波器
    for i in range(n_mels):
        n = np.sum(filters[:, i])
        if n > 1e-8:
            filters[:, i] /= n

    return filters


def compute_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: float = 16000.0,
    n_mels: int = 32,
    n_fft: int = 512,
    hop_length: int = 128,
    f_min: float = 80.0,
    f_max: float = 8000.0,
) -> np.ndarray:
    """计算 mel 频谱图 — 与 CochlearNucleus 32 通道 tonotopic 格式兼容.

    Args:
        audio: 音频波形 (n_samples,) 或 (n_samples, n_channels)
        sample_rate: 采样率 (Hz)
        n_mels: mel 通道数 (默认 32, 匹配 CN.N_FREQ_CHANNELS)
        n_fft: FFT 点数
        hop_length: 帧移 (samples)
        f_min: 最低频率 (Hz)
        f_max: 最高频率 (Hz)

    Returns:
        mel_spec: (n_frames, n_mels) mel 频谱图
    """
    audio = np.asarray(audio, dtype=np.float32)

    # 单声道: (n_samples,) → (n_samples, 1)
    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)

    n_samples, n_channels = audio.shape

    # 预计算 mel 滤波器组
    mel_fb = _mel_filterbank(n_fft, sample_rate, n_mels, f_min, f_max)

    # 汉明窗
    window = np.hamming(n_fft).astype(np.float32)

    # 帧数
    n_frames = max(1, (n_samples - n_fft) // hop_length + 1)

    # 为每帧计算 mel 频谱 (对所有声道取平均)
    mel_spec = np.zeros((n_frames, n_mels), dtype=np.float32)

    for frame_idx in range(n_frames):
        start = frame_idx * hop_length
        end = start + n_fft

        if end > n_samples:
            # 最后不完整帧: 零填充
            frame = np.zeros(n_fft, dtype=np.float32)
            frame[:n_samples - start] = audio[start:n_samples, :].mean(axis=1)
        else:
            frame = audio[start:end, :].mean(axis=1)

        # 加窗 + FFT
        frame_windowed = frame * window
        mag_spec = np.abs(np.fft.rfft(frame_windowed, n=n_fft))

        # mel 滤波
        mel_spec[frame_idx] = mag_spec @ mel_fb

    # dB 转换 (log-mel, 匹配听觉感知的对数特性)
    # 加小常数防止 log(0)
    mel_spec = np.log1p(mel_spec * 10.0) / np.log(11.0)

    return mel_spec.astype(np.float32)


def _compute_average_spectrum(
    mel_spec: np.ndarray,
    method: str = 'energy',
) -> np.ndarray:
    """从 mel 频谱图计算代表性频谱向量 (用于单帧处理).

    Args:
        mel_spec: (n_frames, n_mels) mel 频谱图
        method: 'energy' (能量均值) | 'max' (各通道最大值) | 'mean' (简单平均)

    Returns:
        spectrum: (n_mels,) 代表性频谱
    """
    if mel_spec.shape[0] <= 1:
        return mel_spec[0].copy() if mel_spec.shape[0] == 1 else np.zeros(
            mel_spec.shape[1], dtype=np.float32)

    if method == 'energy':
        # 能量加权平均: 高能量帧贡献更大
        frame_energy = np.sum(mel_spec, axis=1) + 1e-8
        weights = frame_energy / np.sum(frame_energy)
        spectrum = np.sum(mel_spec * weights[:, np.newaxis], axis=0)
    elif method == 'max':
        spectrum = np.max(mel_spec, axis=0)
    else:  # 'mean'
        spectrum = np.mean(mel_spec, axis=0)

    return spectrum.astype(np.float32)


# ============================================================
# AudioInput — 真实音频输入接口
# ============================================================

class AudioInput:
    """真实音频输入 — 加载/录制音频并转换为耳蜗核兼容的频谱格式.

    用法:
      # 从文件加载
      audio_data = AudioInput.from_file('sound.wav')
      # audio_data['spectrum']       → (32,) mono mel 频谱
      # audio_data['left_spectrum']  → (32,) 左耳 (立体声时)
      # audio_data['right_spectrum'] → (32,) 右耳 (立体声时)

      # 从麦克风录制
      audio_data = AudioInput.from_mic(duration_sec=3.0)
    """

    @staticmethod
    def from_file(
        path: str,
        sample_rate: float = 16000.0,
        n_mels: int = 32,
        mono_mix: str = 'average',
    ) -> dict:
        """从音频文件加载并计算 mel 频谱.

        支持 WAV, FLAC, OGG 等 soundfile 可读格式。
        MP3 需要 soundfile >= 0.12 + libsndfile 支持，
        或可先转换为 WAV。

        Args:
            path: 音频文件路径
            sample_rate: 目标采样率 (Hz, 默认 16000 语音优化)
            n_mels: mel 通道数 (默认 32, 匹配 CochlearNucleus)
            mono_mix: 立体声→单声道混合方式
                      'average' = 左右平均
                      'left'    = 仅左声道
                      'right'   = 仅右声道

        Returns:
            dict with:
              'spectrum': mono mel 频谱 (n_mels,)
              'left_spectrum': 左耳频谱 (n_mels,) — 立体声时
              'right_spectrum': 右耳频谱 (n_mels,) — 立体声时
              'is_stereo': 是否立体声
              'duration': 音频时长 (秒)
              'sample_rate': 实际采样率
              'n_frames': mel 频谱帧数
              'waveform': 原始波形 (n_samples, n_channels)
              'rms_energy': 均方根能量
              'is_speech': 粗略语音检测 (基于频谱能量分布)
        """
        import soundfile as sf

        # 读取音频
        waveform, file_sr = sf.read(path, dtype='float32')
        waveform = np.asarray(waveform, dtype=np.float32)

        # 确保至少是 2D
        if waveform.ndim == 1:
            waveform = waveform.reshape(-1, 1)

        n_samples, n_channels = waveform.shape
        duration = n_samples / file_sr

        # 重采样 (如需要)
        if abs(file_sr - sample_rate) > 100:
            waveform = _resample(waveform, file_sr, sample_rate)
            n_samples = waveform.shape[0]
            duration = n_samples / sample_rate
        else:
            sample_rate = file_sr

        is_stereo = (n_channels >= 2)

        # 计算 mel 频谱图
        if is_stereo:
            left_wave = waveform[:, 0]
            right_wave = waveform[:, 1]

            left_mel = compute_mel_spectrogram(
                left_wave, sample_rate, n_mels=n_mels)
            right_mel = compute_mel_spectrogram(
                right_wave, sample_rate, n_mels=n_mels)

            # 双耳频谱: 能量加权平均
            left_spectrum = _compute_average_spectrum(left_mel)
            right_spectrum = _compute_average_spectrum(right_mel)

            # 混合单声道
            if mono_mix == 'left':
                mono_mel = left_mel
                spectrum = left_spectrum.copy()
            elif mono_mix == 'right':
                mono_mel = right_mel
                spectrum = right_spectrum.copy()
            else:
                mono_mel = (left_mel + right_mel) / 2.0
                spectrum = (left_spectrum + right_spectrum) / 2.0

        else:
            mono_mel = compute_mel_spectrogram(
                waveform[:, 0], sample_rate, n_mels=n_mels)
            spectrum = _compute_average_spectrum(mono_mel)
            left_spectrum = None
            right_spectrum = None

        # RMS 能量
        rms_energy = float(np.sqrt(np.mean(waveform ** 2)))

        # 粗略语音检测: 语音能量集中在 300-3400 Hz
        # mel 通道 4-28 大致覆盖此范围
        if mono_mel.shape[0] > 0:
            speech_band = mono_mel[:, 4:28].mean()
            total_band = mono_mel.mean()
            speech_ratio = float(speech_band / (total_band + 1e-8))
            is_speech = speech_ratio > 1.1
        else:
            is_speech = False

        result = {
            'spectrum': spectrum.astype(np.float32),
            'is_stereo': is_stereo,
            'duration': duration,
            'sample_rate': int(sample_rate),
            'n_frames': mono_mel.shape[0],
            'waveform': waveform,
            'rms_energy': rms_energy,
            'is_speech': is_speech,
        }

        if is_stereo:
            result['left_spectrum'] = left_spectrum.astype(np.float32)
            result['right_spectrum'] = right_spectrum.astype(np.float32)

        return result

    @staticmethod
    def from_mic(
        duration_sec: float = 3.0,
        sample_rate: float = 16000.0,
        n_mels: int = 32,
        device: Optional[int] = None,
    ) -> dict:
        """从麦克风录制音频并计算 mel 频谱.

        Args:
            duration_sec: 录制时长 (秒)
            sample_rate: 采样率 (Hz)
            n_mels: mel 通道数
            device: 输入设备 ID (None=默认设备)

        Returns:
            dict: 与 from_file() 相同的格式
        """
        import sounddevice as sd

        n_samples = int(duration_sec * sample_rate)

        # 录制
        recording = sd.rec(
            n_samples,
            samplerate=int(sample_rate),
            channels=1,  # 单声道 (大多数麦克风)
            dtype='float32',
            device=device,
        )
        sd.wait()

        recording = np.asarray(recording, dtype=np.float32)
        if recording.ndim == 1:
            recording = recording.reshape(-1, 1)

        n_samples_actual = recording.shape[0]
        duration_actual = n_samples_actual / sample_rate

        # 计算 mel 频谱
        mono_mel = compute_mel_spectrogram(
            recording[:, 0], sample_rate, n_mels=n_mels)
        spectrum = _compute_average_spectrum(mono_mel)

        rms_energy = float(np.sqrt(np.mean(recording ** 2)))

        return {
            'spectrum': spectrum.astype(np.float32),
            'is_stereo': False,
            'duration': duration_actual,
            'sample_rate': int(sample_rate),
            'n_frames': mono_mel.shape[0],
            'waveform': recording,
            'rms_energy': rms_energy,
            'is_speech': False,  # 麦克风输入不预判
        }

    @staticmethod
    def from_waveform(
        waveform: np.ndarray,
        sample_rate: float = 16000.0,
        n_mels: int = 32,
    ) -> dict:
        """从内存波形计算 mel 频谱 (用于程序化输入).

        Args:
            waveform: (n_samples,) 或 (n_samples, n_channels) 音频波形
            sample_rate: 采样率 (Hz)
            n_mels: mel 通道数

        Returns:
            dict: 与 from_file() 相同的格式
        """
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim == 1:
            waveform = waveform.reshape(-1, 1)

        n_samples, n_channels = waveform.shape
        duration = n_samples / sample_rate
        is_stereo = (n_channels >= 2)

        if is_stereo:
            left_mel = compute_mel_spectrogram(
                waveform[:, 0], sample_rate, n_mels=n_mels)
            right_mel = compute_mel_spectrogram(
                waveform[:, 1], sample_rate, n_mels=n_mels)
            left_spectrum = _compute_average_spectrum(left_mel)
            right_spectrum = _compute_average_spectrum(right_mel)
            spectrum = (left_spectrum + right_spectrum) / 2.0
        else:
            mono_mel = compute_mel_spectrogram(
                waveform[:, 0], sample_rate, n_mels=n_mels)
            spectrum = _compute_average_spectrum(mono_mel)
            left_spectrum = None
            right_spectrum = None

        rms_energy = float(np.sqrt(np.mean(waveform ** 2)))

        return {
            'spectrum': spectrum.astype(np.float32),
            'is_stereo': is_stereo,
            'duration': duration,
            'sample_rate': int(sample_rate),
            'n_frames': (mono_mel if not is_stereo else left_mel).shape[0],
            'waveform': waveform,
            'rms_energy': rms_energy,
            'is_speech': False,
            'left_spectrum': left_spectrum.astype(np.float32) if is_stereo and left_spectrum is not None else None,
            'right_spectrum': right_spectrum.astype(np.float32) if is_stereo and right_spectrum is not None else None,
        }

    @staticmethod
    def spectrum_to_features(spectrum: np.ndarray) -> dict:
        """从 mel 频谱提取人类可读的声学特征描述.

        用于对话显示: 让用户看到 Agent "听到" 了什么。

        Args:
            spectrum: (32,) mel 频谱

        Returns:
            dict with:
              'loudness': 整体响度 [0, 1]
              'pitch_class': 音高类别 ('low'/'mid'/'high')
              'brightness': 频谱明亮度 [0, 1]
              'spectral_centroid': 频谱质心 (mel 通道索引)
              'has_low_freq': 是否有显著低频能量
              'has_high_freq': 是否有显著高频能量
              'description': 自然语言描述
        """
        spec = np.asarray(spectrum, dtype=np.float32).ravel()

        # 响度
        loudness = float(np.clip(np.mean(spec) * 3.0, 0.0, 1.0))

        # 频谱质心 (mel 通道索引)
        channels = np.arange(len(spec), dtype=np.float32)
        total_energy = np.sum(spec) + 1e-8
        spectral_centroid = float(np.sum(channels * spec) / total_energy)

        # 音高类别
        if spectral_centroid < 10:
            pitch_class = 'low'
        elif spectral_centroid < 22:
            pitch_class = 'mid'
        else:
            pitch_class = 'high'

        # 明亮度: 高频能量占比
        high_bands = spec[20:] if len(spec) > 20 else spec
        low_bands = spec[:12] if len(spec) > 12 else spec
        brightness = float(np.mean(high_bands) / (np.mean(low_bands) + 1e-8))
        brightness_norm = float(np.clip(brightness / 3.0, 0.0, 1.0))

        # 频段检测
        has_low_freq = float(np.mean(spec[:8])) > 0.15
        has_high_freq = float(np.mean(spec[20:])) > 0.1

        # 自然语言描述
        loud_desc = '大声' if loudness > 0.6 else ('轻声' if loudness < 0.2 else '')
        pitch_desc = {'low': '低沉', 'mid': '中音', 'high': '尖锐'}[pitch_class]
        bright_desc = '明亮' if brightness_norm > 0.6 else ('沉闷' if brightness_norm < 0.2 else '')

        parts = [p for p in [loud_desc, pitch_desc, bright_desc] if p]
        description = '，'.join(parts) if parts else '安静'

        return {
            'loudness': loudness,
            'pitch_class': pitch_class,
            'brightness': brightness_norm,
            'spectral_centroid': spectral_centroid,
            'has_low_freq': has_low_freq,
            'has_high_freq': has_high_freq,
            'description': description,
        }


# ============================================================
# 重采样工具
# ============================================================

def _resample(
    waveform: np.ndarray,
    orig_sr: float,
    target_sr: float,
) -> np.ndarray:
    """简单的线性插值重采样 (不依赖 scipy).

    Args:
        waveform: (n_samples, n_channels)
        orig_sr: 原始采样率
        target_sr: 目标采样率

    Returns:
        resampled: (new_n_samples, n_channels)
    """
    if abs(orig_sr - target_sr) < 1:
        return waveform.copy()

    ratio = target_sr / orig_sr
    n_samples, n_channels = waveform.shape
    new_n = int(n_samples * ratio)

    resampled = np.zeros((new_n, n_channels), dtype=np.float32)

    for ch in range(n_channels):
        old_x = np.arange(n_samples, dtype=np.float64)
        new_x = np.arange(new_n, dtype=np.float64) / ratio
        resampled[:, ch] = np.interp(new_x, old_x, waveform[:, ch].astype(np.float64))

    return resampled.astype(np.float32)
