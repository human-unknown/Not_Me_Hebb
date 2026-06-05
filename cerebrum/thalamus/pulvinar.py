"""
pulvinar.py — 丘脑枕 (Pulvinar) [v5.0]

对应脑区: 丘脑枕核
所属层级: 大脑 → 丘脑 → Pulvinar

v5.0 职责:
  - SC → 皮层快速中继 (第二条视觉通路)
  - 低空间频率快速通路
  - 空间显著性 → 关联皮层广播

第二条视觉通路: 视网膜 → SC → Pulvinar → 纹外皮层
  处理快速空间定位和眼动协调, 与 LGN→V1 的意识辨别通路互补.

参考:
  - 中枢神经系统视觉通路.md §1.3 (视束靶区)
  - Crick (1984). Function of the thalamic reticular complex.
"""

import numpy as np


class Pulvinar:
    """丘脑枕 — 第二条视觉通路中继 (v5.0)."""

    def __init__(self, sc_dim: int = 16, output_dim: int = 12):
        self.sc_dim = sc_dim
        self.output_dim = output_dim

        # 低空间频率 + SC 信号 → 皮层广播
        self._W_lowpass = np.random.randn(output_dim,
                                            sc_dim + 8).astype(np.float32) * 0.01

    def relay(self, sc_output: dict,
              low_sf_signal: np.ndarray = None) -> np.ndarray:
        """SC 显著性 + 低空间频率信号 → 皮层广播.

        Args:
            sc_output: SC.feedforward() 输出
            low_sf_signal: 低空间频率视觉信号 (来自快速通路)

        Returns:
            Pulvinar 输出向量 (output_dim,)
        """
        saliency = sc_output.get('saliency_map',
                    np.zeros(self.sc_dim, dtype=np.float32))
        attention_shift = sc_output.get('attention_shift',
                           np.zeros(self.sc_dim, dtype=np.float32))

        # 合并 SC 信号
        sc_combined = np.concatenate([
            self._pad_or_trunc(saliency, self.sc_dim),
            self._pad_or_trunc(attention_shift, 8),
        ]).astype(np.float32)

        if low_sf_signal is not None:
            min_len = min(8, len(low_sf_signal))
            sc_combined[:min_len] += low_sf_signal[:min_len] * 0.5

        output = np.tanh(self._W_lowpass @ sc_combined)
        return output.astype(np.float32)

    def _pad_or_trunc(self, vec: np.ndarray, target_len: int) -> np.ndarray:
        if len(vec) >= target_len:
            return vec[:target_len]
        out = np.zeros(target_len, dtype=np.float32)
        out[:len(vec)] = vec
        return out
