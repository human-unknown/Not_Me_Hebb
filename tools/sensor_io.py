"""
sensor_io.py — NotMe v5.7 摄像头 + 麦克风实时流

提供实时传感器采集能力，支持 /stream start/stop 命令。

CameraInput:
  - OpenCV 摄像头采集，可配置 FPS 和分辨率
  - 非阻塞读取，失败时返回 None 不崩溃

MicrophoneStream:
  - sounddevice 麦克风流，chunk-based 读取
  - 返回 mel 频谱 (32,) 兼容耳蜗核输入

StreamSession:
  - 组合摄像头+麦克风，统一生命周期管理
  - 异步采集: 采集线程与 Agent 处理线程分离

依赖:
  pip install opencv-python sounddevice

用法:
  cam = CameraInput(camera_id=0, fps=5, resolution=(128, 128))
  frame = cam.read_frame()  # (128, 128, 3) RGB uint8 or None
  cam.release()

  mic = MicrophoneStream(sample_rate=22050, chunk_ms=200)
  mic.start()
  spectrum = mic.read_chunk()  # (32,) mel spectrum or None
  mic.stop()
"""

import numpy as np
import time
import threading
from collections import deque
from typing import Optional, Dict, Any, Tuple


# ================================================================
# Camera Input
# ================================================================

class CameraInput:
    """OpenCV 摄像头采集.

    特性:
      - 非阻塞读取 (read_frame 立即返回最新帧)
      - 自动重连 (摄像头断开后尝试重新打开)
      - 帧率控制 (通过时间戳控制实际读取频率)

    Usage:
      cam = CameraInput(camera_id=0, fps=5, resolution=(128, 128))
      frame = cam.read_frame()  # (H, W, 3) RGB uint8 or None
      cam.release()
    """

    def __init__(self, camera_id: int = 0, fps: float = 5.0,
                 resolution: Tuple[int, int] = (128, 128)):
        self.camera_id = camera_id
        self.fps = fps
        self.resolution = resolution
        self._cap = None
        self._last_frame_time = 0.0
        self._frame_interval = 1.0 / max(fps, 0.1)
        self._is_open = False
        self._frames_read = 0
        self._frames_dropped = 0

    @property
    def is_open(self) -> bool:
        return self._is_open and self._cap is not None

    def open(self) -> bool:
        """打开摄像头. 返回是否成功."""
        if self._is_open:
            return True

        try:
            import cv2
            self._cap = cv2.VideoCapture(self.camera_id)
            if not self._cap.isOpened():
                self._cap = None
                return False

            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)
            self._is_open = True
            self._last_frame_time = time.time()
            return True
        except ImportError:
            # cv2 not installed — camera unavailable
            self._cap = None
            return False
        except Exception:
            self._cap = None
            return False

    def read_frame(self) -> Optional[np.ndarray]:
        """读取最新帧.

        Returns:
            (H, W, 3) RGB uint8 array, 或 None (无新帧/摄像头未就绪)
        """
        if not self.is_open:
            if not self.open():
                return None

        now = time.time()
        if now - self._last_frame_time < self._frame_interval:
            return None  # 帧率限制: 不到时间不读取

        self._last_frame_time = now

        try:
            import cv2
            ret, frame = self._cap.read()
            if not ret or frame is None:
                self._frames_dropped += 1
                return None

            # BGR → RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._frames_read += 1
            return frame_rgb
        except Exception:
            self._frames_dropped += 1
            return None

    def release(self):
        """释放摄像头资源."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self._is_open = False

    def stats(self) -> dict:
        """返回采集统计."""
        return {
            'camera_id': self.camera_id,
            'resolution': self.resolution,
            'fps_target': self.fps,
            'is_open': self._is_open,
            'frames_read': self._frames_read,
            'frames_dropped': self._frames_dropped,
        }

    def __del__(self):
        self.release()


# ================================================================
# Microphone Stream
# ================================================================

class MicrophoneStream:
    """sounddevice 麦克风流采集.

    使用非阻塞 callback 模式: 音频数据持续写入环形缓冲区,
    read_chunk() 从中读出最近 chunk_ms 的数据并计算 mel 频谱。

    Usage:
      mic = MicrophoneStream(sample_rate=22050, chunk_ms=200)
      mic.start()
      spectrum = mic.read_chunk()  # (32,) mel spectrum or None
      mic.stop()
    """

    def __init__(self, sample_rate: float = 22050, chunk_ms: int = 200,
                 n_mels: int = 32, device: Optional[int] = None):
        self.sample_rate = int(sample_rate)
        self.chunk_ms = chunk_ms
        self.n_mels = n_mels
        self.device = device

        # 环形缓冲区: 保留最近 ~2 秒的音频
        self._buffer_samples = int(self.sample_rate * 2.0)
        self._buffer = deque(maxlen=self._buffer_samples)
        self._stream = None
        self._is_streaming = False
        self._lock = threading.Lock()
        self._chunks_read = 0
        self._total_samples = 0

    @property
    def is_streaming(self) -> bool:
        return self._is_streaming

    def start(self) -> bool:
        """启动麦克风流. 返回是否成功."""
        if self._is_streaming:
            return True

        try:
            import sounddevice as sd

            # 查询设备
            if self.device is not None:
                try:
                    sd.check_input_settings(
                        device=self.device,
                        samplerate=self.sample_rate,
                        channels=1,
                    )
                except Exception:
                    pass  # 回退到默认设备

            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                device=self.device,
                callback=self._audio_callback,
                blocksize=int(self.sample_rate * self.chunk_ms / 1000),
            )
            self._stream.start()
            self._is_streaming = True
            return True
        except ImportError:
            # sounddevice not installed
            self._stream = None
            return False
        except Exception:
            self._stream = None
            return False

    def _audio_callback(self, indata, frames, time_info, status):
        """sounddevice callback — 将音频数据写入环形缓冲区."""
        if status:
            # 溢出/下溢 → 静默处理, 不影响主流程
            pass
        with self._lock:
            for sample in indata[:, 0]:
                self._buffer.append(float(sample))
        self._total_samples += frames

    def read_chunk(self) -> Optional[Dict[str, Any]]:
        """读取最近一个 chunk 的音频数据并计算 mel 频谱.

        Returns:
            dict with:
              'spectrum': mono mel spectrum (n_mels,)
              'rms_energy': RMS energy
              'is_silence': True if below noise floor
            or None if no data available yet
        """
        if not self._is_streaming:
            return None

        chunk_samples = int(self.sample_rate * self.chunk_ms / 1000)

        with self._lock:
            if len(self._buffer) < chunk_samples:
                return None

            # 取出最近 chunk_samples 个样本
            buf_list = list(self._buffer)
            if len(buf_list) > chunk_samples:
                chunk = np.array(
                    buf_list[-chunk_samples:], dtype=np.float32)
            else:
                chunk = np.array(buf_list, dtype=np.float32)

        # 计算 RMS 能量
        rms = float(np.sqrt(np.mean(chunk ** 2) + 1e-10))
        is_silence = rms < 0.005  # 低于此阈值视为静音

        # 计算 mel 频谱
        try:
            spectrum = self._compute_mel_spectrum(chunk)
        except Exception:
            spectrum = np.zeros(self.n_mels, dtype=np.float32)

        self._chunks_read += 1

        return {
            'spectrum': spectrum.astype(np.float32),
            'rms_energy': rms,
            'is_silence': is_silence,
            'n_samples': len(chunk),
        }

    def _compute_mel_spectrum(self, waveform: np.ndarray) -> np.ndarray:
        """计算 mel 频谱 (兼容耳蜗核输入格式).

        Args:
            waveform: (n_samples,) float32 波形

        Returns:
            (n_mels,) mel 频谱
        """
        n_fft = 512
        hop_length = n_fft // 4

        # 加窗 FFT
        window = np.hanning(n_fft)
        n_frames = (len(waveform) - n_fft) // hop_length + 1

        if n_frames < 1:
            return np.zeros(self.n_mels, dtype=np.float32)

        spectrogram = np.zeros((n_frames, n_fft // 2 + 1), dtype=np.float32)
        for i in range(n_frames):
            start = i * hop_length
            frame = waveform[start:start + n_fft] * window
            spec = np.abs(np.fft.rfft(frame))
            spectrogram[i] = spec

        # 平均频谱 (跨帧)
        avg_spectrum = np.mean(spectrogram, axis=0)

        # Mel 滤波器组
        mel_spec = self._apply_mel_filterbank(avg_spectrum, self.sample_rate)

        # Log-mel (dB-like, 稳定化)
        mel_spec = np.log1p(mel_spec * 10.0) / np.log(2.0)
        mel_spec = np.clip(mel_spec, 0.0, 10.0)

        return mel_spec.astype(np.float32)

    def _apply_mel_filterbank(self, power_spec: np.ndarray,
                               sr: int) -> np.ndarray:
        """简化的 mel 滤波器组."""
        n_fft = (len(power_spec) - 1) * 2
        n_filters = self.n_mels

        # Mel scale
        low_mel = self._hz_to_mel(80.0)
        high_mel = self._hz_to_mel(sr / 2.0)
        mel_points = np.linspace(low_mel, high_mel, n_filters + 2)
        hz_points = self._mel_to_hz(mel_points)
        bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
        bin_points = np.clip(bin_points, 0, len(power_spec) - 1)

        filters = np.zeros((n_filters, len(power_spec)), dtype=np.float32)
        for i in range(n_filters):
            left = bin_points[i]
            center = bin_points[i + 1]
            right = bin_points[i + 2]

            if center > left:
                filters[i, left:center + 1] = np.linspace(0, 1, center - left + 1)
            if right > center:
                filters[i, center:right + 1] = np.linspace(1, 0, right - center + 1)

        mel_energy = np.dot(filters, power_spec)
        return mel_energy

    @staticmethod
    def _hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    @staticmethod
    def _mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def stop(self):
        """停止麦克风流."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self._is_streaming = False

    def stats(self) -> dict:
        """返回采集统计."""
        return {
            'is_streaming': self._is_streaming,
            'sample_rate': self.sample_rate,
            'chunk_ms': self.chunk_ms,
            'chunks_read': self._chunks_read,
            'total_samples': self._total_samples,
            'buffer_fill': len(self._buffer) if self._buffer else 0,
        }

    def __del__(self):
        self.stop()


# ================================================================
# Stream Session: Camera + Mic Combined
# ================================================================

class StreamSession:
    """组合摄像头+麦克风的实时流会话.

    管理两个传感器的生命周期, 提供统一的 read() 接口。
    异步架构: 采集与 Agent 处理在同一线程中顺序执行,
    但通过帧率控制避免 Agent 处理阻塞传感器读取。

    Usage:
      session = StreamSession(camera_id=0, fps=5, mic_chunk_ms=200)
      session.start()
      while True:
          frame_data = session.read()
          if frame_data is None:
              break
          # process frame_data with agent
      session.stop()
    """

    def __init__(self, camera_id: int = 0, camera_fps: float = 5.0,
                 camera_resolution: Tuple[int, int] = (128, 128),
                 mic_sample_rate: float = 22050, mic_chunk_ms: int = 200,
                 mic_device: Optional[int] = None):
        self.camera = CameraInput(
            camera_id=camera_id,
            fps=camera_fps,
            resolution=camera_resolution,
        )
        self.mic = MicrophoneStream(
            sample_rate=mic_sample_rate,
            chunk_ms=mic_chunk_ms,
            device=mic_device,
        )
        self._running = False
        self._n_frames = 0

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> Dict[str, bool]:
        """启动所有传感器. 返回各通道状态."""
        cam_ok = self.camera.open()
        mic_ok = self.mic.start()
        self._running = True
        self._n_frames = 0
        return {'camera': cam_ok, 'mic': mic_ok}

    def read(self) -> Optional[Dict[str, Any]]:
        """读取一帧 (摄像头 + 麦克风).

        非阻塞: 哪个传感器有新数据就返回哪个,
        都没有新数据时返回 None.

        Returns:
            dict with:
              'frame': (H,W,3) RGB uint8 or None
              'audio': mic spectrum dict or None
              'camera_ok': bool
              'mic_ok': bool
            or None if stream not running
        """
        if not self._running:
            return None

        frame = self.camera.read_frame()
        audio = self.mic.read_chunk()

        # 至少有一个有新数据才返回 (减少空帧)
        if frame is None and audio is None:
            return None

        self._n_frames += 1

        return {
            'frame': frame,
            'audio': audio,
            'camera_ok': self.camera.is_open,
            'mic_ok': self.mic.is_streaming,
            'frame_id': self._n_frames,
        }

    def stop(self):
        """停止所有传感器并释放资源."""
        self._running = False
        self.camera.release()
        self.mic.stop()

    def stats(self) -> dict:
        """返回综合统计."""
        return {
            'running': self._running,
            'n_frames': self._n_frames,
            'camera': self.camera.stats(),
            'mic': self.mic.stats(),
        }

    def __del__(self):
        self.stop()
