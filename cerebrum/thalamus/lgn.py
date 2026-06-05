"""
lgn.py — 外侧膝状体核 (Lateral Geniculate Nucleus) [v5.0]

对应脑区: 丘脑 LGN (背侧部)
所属层级: 大脑 → 丘脑 → LGN

6层结构 (v5.0):
  第 1 层 (大细胞, 对侧眼): M 通路
  第 2 层 (大细胞, 同侧眼): M 通路
  第 3 层 (小细胞, 同侧眼): P 通路
  第 4 层 (小细胞, 对侧眼): P 通路
  第 5 层 (小细胞, 同侧眼): P 通路
  第 6 层 (小细胞, 对侧眼): P 通路
  层间区 (粒状细胞):        K 通路

门控机制:
  1. V1 第 6 层反馈 → 调制各层增益
  2. 脑干状态 → tonic (清醒/中继) vs burst (睡眠/衰减) 模式
  3. TRN 侧抑制 → 层间抑制增强信噪比

参考:
  - 中枢神经系统视觉通路.md §2 (外侧膝状体核)
  - Sherman & Guillery (2002). The role of the thalamus.
"""

import numpy as np
from typing import Optional


class LGN:
    """外侧膝状体核 — 视觉中继站与主动门控 (v5.0).

    输入: 视网膜 M/P/K 信号 (每种为原始特征向量)
    输出: 门控后的 M/P/K 信号

    关键特性:
      - 80-90% 突触输入来自非视网膜来源 (V1反馈、脑干、TRN)
      - 不是被动中继, 而是主动的、状态依赖的信息过滤器
    """

    def __init__(self, M_dim: int = 1024, P_dim: int = 1024, K_dim: int = 1024,
                 trn_strength: float = 0.05):
        self.M_dim = M_dim
        self.P_dim = P_dim
        self.K_dim = K_dim

        # ---- 6 层增益 (初始 = 1.0, 全通透) ----
        # M 通路: 层 1 (对侧), 层 2 (同侧)
        self.M_gain = np.ones(2, dtype=np.float32)   # [layer1, layer2]
        # P 通路: 层 3 (同侧), 层 4 (对侧), 层 5 (同侧), 层 6 (对侧)
        self.P_gain = np.ones(4, dtype=np.float32)   # [layer3, layer4, layer5, layer6]
        # K 通路: 层间区 (单通道)
        self.K_gain: float = 1.0

        # ---- TRN 侧抑制强度 ----
        self.trn_strength = trn_strength

        # ---- 脑干状态 ----
        self.brainstem_arousal: float = 0.5  # [0, 1], 0=睡眠, 1=高警觉
        self.burst_threshold: float = 0.3    # 低于此值 → burst 模式

        # ---- V1 反馈缓存 ----
        self._v1_feedback: Optional[np.ndarray] = None

    def relay(self, M_signal: np.ndarray, P_signal: np.ndarray,
              K_signal: np.ndarray,
              brainstem_arousal: float = 0.5,
              v1_feedback: Optional[np.ndarray] = None) -> dict:
        """中继视网膜信号到 V1, 应用门控调制.

        Args:
            M_signal: M 通路信号 (M_dim,)
            P_signal: P 通路信号 (P_dim,)
            K_signal: K 通路信号 (K_dim,)
            brainstem_arousal: 脑干唤醒度 [0, 1]
            v1_feedback: V1 第6层反馈信号 (用于增益调制, 可选)

        Returns:
            dict with keys 'M', 'P', 'K' — 门控后信号
        """
        self.brainstem_arousal = brainstem_arousal
        self._v1_feedback = v1_feedback

        # ---- 1. V1 反馈 → 调制各层增益 ----
        if v1_feedback is not None:
            fb_len = len(v1_feedback)
            # V1 反馈分量为 M/P/K 增益调整信号
            # 反馈前2维→M增益, 中4维→P增益, 后1维→K增益
            if fb_len >= 7:
                self.M_gain = np.clip(1.0 + np.tanh(v1_feedback[:2]), 0.1, 3.0)
                self.P_gain = np.clip(1.0 + np.tanh(v1_feedback[2:6]), 0.1, 3.0)
                self.K_gain = float(np.clip(1.0 + np.tanh(v1_feedback[6]), 0.1, 3.0))
            elif fb_len >= 2:
                self.M_gain = np.clip(1.0 + np.tanh(v1_feedback[:2]), 0.1, 3.0)

        # ---- 2. 脑干状态 → tonic/burst 模式 ----
        if brainstem_arousal >= self.burst_threshold:
            # Tonic mode: 线性中继 (清醒状态)
            M_out = M_signal * float(np.mean(self.M_gain))
            P_out = P_signal * float(np.mean(self.P_gain))
            K_out = K_signal * self.K_gain
        else:
            # Burst mode: 阈值衰减 — 弱信号被抑制 (睡眠/低唤醒)
            M_thresh = M_signal * (np.abs(M_signal) > 0.1).astype(np.float32)
            P_thresh = P_signal * (np.abs(P_signal) > 0.1).astype(np.float32)
            K_thresh = K_signal * (np.abs(K_signal) > 0.1).astype(np.float32)
            attenuation = brainstem_arousal / self.burst_threshold
            M_out = M_thresh * float(np.mean(self.M_gain)) * attenuation
            P_out = P_thresh * float(np.mean(self.P_gain)) * attenuation
            K_out = K_thresh * self.K_gain * attenuation

        # ---- 3. TRN 侧抑制 (跨层竞争, 增强信噪比) ----
        M_out = self._apply_trn(M_out)
        P_out = self._apply_trn(P_out)
        K_out = self._apply_trn_single(K_out)

        return {'M': M_out.astype(np.float32),
                'P': P_out.astype(np.float32),
                'K': K_out.astype(np.float32)}

    def _apply_trn(self, signal: np.ndarray) -> np.ndarray:
        """TRN 侧抑制: 维度内竞争的软版本.

        强分量压制弱分量 → 增强信噪比.
        """
        if self.trn_strength <= 0:
            return signal
        abs_s = np.abs(signal)
        mean_abs = float(np.mean(abs_s)) + 1e-8
        suppression = 1.0 - self.trn_strength * (1.0 - abs_s / (mean_abs * 2.0))
        suppression = np.clip(suppression, 0.5, 1.0)
        return signal * suppression

    def _apply_trn_single(self, signal: np.ndarray) -> np.ndarray:
        """K 通道侧抑制 (标量增益, 无同层竞争)."""
        return signal

    def get_state(self) -> dict:
        """返回 LGN 当前状态 (诊断用)."""
        return {
            'M_gain_mean': float(np.mean(self.M_gain)),
            'P_gain_mean': float(np.mean(self.P_gain)),
            'K_gain': float(self.K_gain),
            'brainstem_arousal': float(self.brainstem_arousal),
            'mode': 'tonic' if self.brainstem_arousal >= self.burst_threshold else 'burst',
        }
