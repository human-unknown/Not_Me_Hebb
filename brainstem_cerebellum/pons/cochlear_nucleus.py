"""
cochlear_nucleus.py — 耳蜗核 (Cochlear Nucleus) [v5.2]

对应脑区: 耳蜗核 (延髓上部, 脑桥-延髓交界)
所属层级: 脑干 → 脑桥 (Level 3)
脑区标签: AVCN · PVCN · DCN

功能职责:
  - 听觉系统CNS第一突触中继站 — 听神经→耳蜗核→上位核团
  - 频率分解 — 将输入频谱映射到tonotopic通道
  - 时间编码 — 相位锁定保留 (AVCN) + 起始检测 (PVCN)
  - 频谱分析 — 频谱缺口检测 (DCN, HRTF线索)
  - 响度编码 — 对数压缩 + 纤维募集

三大亚区:
  - AVCN (前腹侧): 大球形细胞 — 精确相位锁定, primary-like反应
  - PVCN (后腹侧): 章鱼细胞 — 起始检测, 亚毫秒时间精度
  - DCN (背侧): 频谱缺口/峰值选择性 — 垂直定位HRTF线索

知觉规律:
  1. 音调拓扑组织 (Tonotopy) — 频率→空间映射
  2. 部位说 (Place Code) — 不同频率→不同通道位置
  3. 频率说/齐射 (Volley Code) — 低频相位锁定
  4. 响度编码 — 速率编码 + 纤维募集 + 对数压缩
  5. 感觉适应 — 持续恒定声音响应衰减
  6. 声反射 — 高强度→增益衰减 (保护机制)

在 NotMe 中的应用:
  - 接收频谱输入 (mel-spectrum 或语义合成的伪频谱)
  - 输出 tonotopic map → SuperiorOlive / LateralLemniscus
  - 为全听觉通路提供频率-空间组织基础
"""

import numpy as np
from typing import Optional, Tuple

# ============================================================
# 常量
# ============================================================

N_FREQ_CHANNELS = 32          # 频率通道数 (mel-spaced)
N_DCN_CHANNELS = 16           # DCN 频谱缺口检测输出维度
LOW_FREQ_CUTOFF = 1500.0      # 相位锁定截止频率 (Hz)
MID_FREQ_CUTOFF = 4000.0      # 时间编码过渡区上界 (Hz)
ACOUSTIC_REFLEX_THRESHOLD = 0.7  # 声反射触发阈值 (归一化声强)
ACOUSTIC_REFLEX_ATTEN = 0.4      # 声反射衰减因子

# mel-scale 中心频率 (32通道, 覆盖 ~80Hz ~ 8000Hz 语音相关范围)
_MEL_CENTERS = np.logspace(np.log10(80), np.log10(8000), N_FREQ_CHANNELS)


def _mel_filterbank(n_input: int, sample_rate: float = 16000.0) -> np.ndarray:
    """生成 mel 滤波器组矩阵 (n_input, N_FREQ_CHANNELS)."""
    mel_min = 2595.0 * np.log10(1 + 80.0 / 700.0)
    mel_max = 2595.0 * np.log10(1 + 8000.0 / 700.0)
    mel_centers = np.linspace(mel_min, mel_max, N_FREQ_CHANNELS)
    hz_centers = 700.0 * (10.0 ** (mel_centers / 2595.0) - 1.0)

    freq_bins = np.linspace(0, sample_rate / 2, n_input)
    filters = np.zeros((n_input, N_FREQ_CHANNELS), dtype=np.float32)

    for i in range(N_FREQ_CHANNELS):
        lo = hz_centers[i - 1] if i > 0 else hz_centers[0] / 2
        hi = hz_centers[i + 1] if i < N_FREQ_CHANNELS - 1 else hz_centers[-1] * 2
        center = hz_centers[i]

        for j, f in enumerate(freq_bins):
            if f <= lo or f >= hi:
                continue
            if f <= center:
                filters[j, i] = (f - lo) / (center - lo + 1e-8)
            else:
                filters[j, i] = (hi - f) / (hi - center + 1e-8)

    # 归一化每个滤波器
    for i in range(N_FREQ_CHANNELS):
        n = np.sum(filters[:, i])
        if n > 1e-8:
            filters[:, i] /= n

    return filters


class CochlearNucleus:
    """耳蜗核 — 听觉系统CNS第一中继站.

    将频谱输入分解为tonotopic通道，保留时间编码，
    检测频谱缺口(HRTF线索)，编码响度。

    用法:
      cn = CochlearNucleus()
      output = cn.process(spectrum)
      # output['tonotopic']  → (32,) 频率通道激活
      # output['temporal']   → (32,) 相位锁定模式
      # output['onset']      → (32,) 起始检测
      # output['dcn']        → (16,) 频谱缺口特征
    """

    def __init__(self, n_freq: int = N_FREQ_CHANNELS,
                 n_dcn: int = N_DCN_CHANNELS):
        self.n_freq = n_freq
        self.n_dcn = n_dcn

        # Mel 滤波器组 (懒加载, 在首次 process 时初始化)
        self._mel_filters: Optional[np.ndarray] = None
        self._n_input: int = 0

        # 适应状态: 每个频率通道的适应水平 [0, 1]
        self.adaptation_state: np.ndarray = np.zeros(n_freq, dtype=np.float32)
        self.adaptation_tau: float = 0.95  # 适应时间常数 (越大适应越慢)

        # 声反射状态: 整体增益衰减 [0, 1]
        self.acoustic_reflex_state: float = 0.0
        self.reflex_attack: float = 0.3    # 攻击速率
        self.reflex_release: float = 0.02  # 释放速率

        # 前一帧频谱 (用于起始检测和预测)
        self._prev_spectrum: Optional[np.ndarray] = None

        # 预测状态 (用于计算预测误差)
        self._prediction: Optional[np.ndarray] = None

    # ================================================================
    # 核心处理
    # ================================================================

    def process(self, spectrum: np.ndarray,
                sample_rate: float = 16000.0,
                use_mel: bool = True) -> dict:
        """处理一帧频谱输入.

        Args:
            spectrum: 频谱向量 (n_freq_bins,) 或已mel化的 (N_FREQ_CHANNELS,)
            sample_rate: 采样率 (仅用于mel滤波器组初始化)
            use_mel: 是否使用mel滤波器组 (输入为线性频谱时设为True)

        Returns:
            dict with:
              'tonotopic': 频率通道激活 (N_FREQ_CHANNELS,) — AVCN输出
              'temporal':  相位锁定模式 (N_FREQ_CHANNELS,) — 低频保留
              'onset':     起始检测 (N_FREQ_CHANNELS,) — PVCN输出
              'dcn':       频谱缺口特征 (N_DCN_CHANNELS,) — DCN输出
              'loudness':  总体响度 (scalar)
              'adapted':   适应后的tonotopic (供后续核团使用)
        """
        spectrum = np.asarray(spectrum, dtype=np.float32).ravel()

        # ---- Mel 滤波 ----
        if use_mel and len(spectrum) != self.n_freq:
            spectrum = self._apply_mel_filter(spectrum, sample_rate)

        # 确保长度匹配
        if len(spectrum) != self.n_freq:
            if len(spectrum) > self.n_freq:
                spectrum = spectrum[:self.n_freq]
            else:
                padded = np.zeros(self.n_freq, dtype=np.float32)
                padded[:len(spectrum)] = spectrum
                spectrum = padded

        # ---- 声反射: 高强度声音→衰减 ----
        total_intensity = float(np.mean(np.abs(spectrum)))
        if total_intensity > ACOUSTIC_REFLEX_THRESHOLD:
            self.acoustic_reflex_state = min(1.0,
                self.acoustic_reflex_state + self.reflex_attack)
        else:
            self.acoustic_reflex_state = max(0.0,
                self.acoustic_reflex_state - self.reflex_release)

        reflex_gain = 1.0 - self.acoustic_reflex_state * ACOUSTIC_REFLEX_ATTEN
        spectrum = spectrum * reflex_gain

        # ---- 感觉适应: 恒定声音响应衰减 ----
        # 适应状态向当前输入靠拢，差异越大→适应越少
        self.adaptation_state = (
            self.adaptation_tau * self.adaptation_state +
            (1.0 - self.adaptation_tau) * spectrum
        )
        # 适应后的输出: 原始输入减去适应状态 (恒定部分被抑制)
        adapted = np.maximum(0.0, spectrum - self.adaptation_state * 0.7)

        # ---- AVCN: 频率通道激活 (tonotopic map) ----
        # 大球形细胞: primary-like 反应, 保留频率调谐
        # 响度编码: 对数压缩 (Weber-Fechner律)
        tonotopic = self._loudness_encode(adapted)

        # ---- AVCN: 相位锁定 (temporal code) ----
        # 低频通道 (<1.5kHz) 保留精确的相位锁定模式
        # 高频通道 (>4kHz) 丧失锁相能力
        temporal = self._compute_phase_locking(adapted)

        # ---- PVCN: 起始检测 (onset response) ----
        # 章鱼细胞: 对声音起始产生精确的单个spike
        onset = self._detect_onset(spectrum)

        # ---- DCN: 频谱缺口检测 (HRTF线索) ----
        dcn_output = self._detect_spectral_notches(adapted)

        # ---- 预测更新 (用于预测编码) ----
        if self._prev_spectrum is not None:
            # 简单预测: 前一帧 + 趋势
            trend = adapted - self._prev_spectrum if self._prev_spectrum is not None else 0
            self._prediction = adapted + 0.3 * (adapted - self._prev_spectrum)
        else:
            self._prediction = adapted.copy()
        self._prev_spectrum = adapted.copy()

        # ---- 总响度 ----
        loudness = float(np.log1p(total_intensity * 10.0) / np.log(11.0))

        return {
            'tonotopic': tonotopic.astype(np.float32),
            'temporal': temporal.astype(np.float32),
            'onset': onset.astype(np.float32),
            'dcn': dcn_output.astype(np.float32),
            'loudness': loudness,
            'adapted': adapted.astype(np.float32),
        }

    # ================================================================
    # 预测编码接口
    # ================================================================

    def get_prediction(self) -> np.ndarray:
        """返回当前预测 (供上位核团反馈使用)."""
        if self._prediction is None:
            return np.zeros(self.n_freq, dtype=np.float32)
        return self._prediction.copy()

    def receive_feedback(self, prediction_error: np.ndarray,
                         lr: float = 0.1):
        """接收上位核团的预测误差反馈, 更新内部预测.

        Args:
            prediction_error: 上位核团计算的预测误差 (N_FREQ_CHANNELS,)
            lr: 学习率
        """
        if self._prediction is not None:
            pe = np.asarray(prediction_error, dtype=np.float32).ravel()
            if len(pe) == self.n_freq:
                self._prediction += lr * pe

    def compute_prediction_error(self) -> dict:
        """计算当前输入的预测误差 (用于汇入F_accuracy)."""
        if self._prev_spectrum is None or self._prediction is None:
            return {'tonotopic': np.zeros(self.n_freq, dtype=np.float32),
                    'dcn': np.zeros(self.n_dcn, dtype=np.float32)}

        pe_tonotopic = self._prev_spectrum - self._prediction
        # DCN的PE是其频谱缺口表征的变化
        current_dcn = self._detect_spectral_notches(self._prev_spectrum)
        if hasattr(self, '_prev_dcn'):
            pe_dcn = current_dcn - self._prev_dcn
        else:
            pe_dcn = np.zeros(self.n_dcn, dtype=np.float32)
        self._prev_dcn = current_dcn

        return {
            'tonotopic': pe_tonotopic.astype(np.float32),
            'dcn': pe_dcn.astype(np.float32),
        }

    # ================================================================
    # 内部方法
    # ================================================================

    def _apply_mel_filter(self, spectrum: np.ndarray,
                          sample_rate: float) -> np.ndarray:
        """线性频谱 → mel滤波器组 → mel频谱."""
        n_input = len(spectrum)
        if self._mel_filters is None or self._n_input != n_input:
            self._mel_filters = _mel_filterbank(n_input, sample_rate)
            self._n_input = n_input
        return (spectrum @ self._mel_filters).astype(np.float32)

    def _loudness_encode(self, spectrum: np.ndarray) -> np.ndarray:
        """响度编码: 对数压缩 + 纤维募集模拟.

        Weber-Fechner律: 感知响度 ∝ log(强度).
        不同阈值纤维的募集: 低阈值纤维在低强度饱和,
        高阈值纤维在高强度才激活 → 扩展动态范围.
        """
        # 对数压缩 (Weber-Fechner)
        compressed = np.log1p(spectrum * 10.0) / np.log(11.0)

        # 模拟纤维募集: 低强度重点在低通道, 高强度激活全通道
        total = float(np.mean(compressed))
        # 低阈值纤维 (high-spontaneous-rate): 对弱信号敏感
        low_thresh = np.tanh(spectrum * 3.0)
        # 高阈值纤维 (low-spontaneous-rate): 对强信号响应
        high_thresh = np.tanh(np.maximum(0.0, spectrum - 0.3) * 5.0)

        # 合并: 低阈值在低强度主导, 高阈值扩展动态范围
        encoded = 0.6 * low_thresh + 0.4 * high_thresh
        return encoded.astype(np.float32)

    def _compute_phase_locking(self, spectrum: np.ndarray) -> np.ndarray:
        """模拟相位锁定: 低频通道保留时间编码.

        频率说/齐射原理 (Volley Principle):
        - <1.5kHz: 精确相位锁定 → 时间编码主导
        - 1.5-4kHz: 过渡区 → 部分锁相
        - >4kHz: 锁相丧失 → 部位编码主导
        """
        temporal = np.zeros(self.n_freq, dtype=np.float32)

        for i in range(self.n_freq):
            freq = _MEL_CENTERS[i]
            if freq < LOW_FREQ_CUTOFF:
                # 精确相位锁定: 输出 ≈ 输入 (保留波形时间结构)
                lock_strength = 1.0
            elif freq < MID_FREQ_CUTOFF:
                # 过渡区: 锁相精度逐渐下降
                lock_strength = 1.0 - (freq - LOW_FREQ_CUTOFF) / (
                    MID_FREQ_CUTOFF - LOW_FREQ_CUTOFF)
            else:
                # 高频: 锁相丧失, 仅保留包络
                lock_strength = 0.0

            temporal[i] = spectrum[i] * lock_strength

        return temporal

    def _detect_onset(self, spectrum: np.ndarray) -> np.ndarray:
        """PVCN 章鱼细胞: 起始检测.

        对声音起始 (onset) 产生精确反应。
        计算当前帧与前一帧的差异，正差异→onset信号。
        """
        if self._prev_spectrum is None:
            onset = np.maximum(0.0, spectrum)
        else:
            # 正的时间差分 → 起始
            diff = spectrum - self._prev_spectrum
            onset = np.maximum(0.0, diff)

        # 锐化: 只保留显著起始 (章鱼细胞的高阈值特性)
        threshold = np.mean(onset) + 0.5 * np.std(onset)
        onset = np.where(onset > threshold, onset, 0.0)

        # 归一化
        max_val = np.max(onset)
        if max_val > 1e-8:
            onset = onset / max_val

        return onset.astype(np.float32)

    def _detect_spectral_notches(self, spectrum: np.ndarray) -> np.ndarray:
        """DCN: 频谱缺口/峰值检测 — HRTF垂直定位线索.

        DCN神经元对频谱结构(缺口/峰值)有选择性,
        这是单耳垂直定位(仰角)和前/后判断的基础。
        耳廓不对称形状在频谱中产生特征性缺口,
        DCN检测这些缺口模式。
        """
        s = spectrum.astype(np.float32)

        # 计算频谱的二阶差分 → 检测缺口和峰值
        if len(s) >= 3:
            # 一阶差分
            d1 = np.diff(s, n=1, prepend=s[0])
            # 二阶差分
            d2 = np.diff(d1, n=1, prepend=d1[0])

            # 缺口 = 负二阶差分 (局部凹陷)
            notches = np.maximum(0.0, -d2)
            # 峰值 = 正二阶差分 (局部突起)
            peaks = np.maximum(0.0, d2)
        else:
            notches = np.zeros_like(s)
            peaks = np.zeros_like(s)

        # DCN输出: 缺口模式 + 峰值模式
        # 池化到 n_dcn 维度
        dcn_features = np.zeros(self.n_dcn, dtype=np.float32)

        # 低频段缺口 (对应大耳廓特征)
        if len(s) >= self.n_dcn:
            bin_size = len(s) // self.n_dcn
            for i in range(self.n_dcn):
                start = i * bin_size
                end = min(start + bin_size, len(s))
                dcn_features[i] = float(
                    0.6 * np.mean(notches[start:end]) +
                    0.4 * np.mean(peaks[start:end])
                )
        else:
            dcn_features[:len(s)] = 0.6 * notches + 0.4 * peaks

        # 归一化
        max_val = np.max(dcn_features)
        if max_val > 1e-8:
            dcn_features = dcn_features / max_val

        return dcn_features.astype(np.float32)

    # ================================================================
    # 语义代理模式: 从语义向量合成伪频谱
    # ================================================================

    @staticmethod
    def semantic_to_pseudo_spectrum(semantic_vec: np.ndarray) -> np.ndarray:
        """从语义向量合成伪频谱 (无真实音频时的代理输入).

        利用语义向量的统计结构生成模拟的频谱激活模式。
        这不是真实的声学特征, 而是让听觉通路在对话模式下
        有可处理的数据流。

        Args:
            semantic_vec: 语义编码向量 (64,)

        Returns:
            pseudo_spectrum: 模拟频谱 (N_FREQ_CHANNELS,)
        """
        sem = np.asarray(semantic_vec, dtype=np.float32).ravel()
        n_sem = len(sem)

        # 将语义向量映射到频谱空间
        # 使用固定的随机投影矩阵 (种子固定保证确定性)
        rng = np.random.RandomState(42)
        proj = rng.randn(N_FREQ_CHANNELS, min(n_sem, 64)).astype(np.float32)

        pseudo = proj @ sem[:proj.shape[1]]
        pseudo = np.tanh(pseudo)  # 限制在 [-1, 1]
        pseudo = np.maximum(0.0, pseudo)  # ReLU → 非负频谱

        return pseudo.astype(np.float32)
