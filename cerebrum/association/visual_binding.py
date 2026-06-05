"""
visual_binding.py — 跨通道视觉绑定 (Visual Binding) [v5.0]

对应功能: FPN 探照灯驱动的 M/P/K 跨通道特征绑定
所属层级: 大脑 → 联合皮层

机制:
  1. FPN 关注的空间位置 → 该位置 M/P/K 增益同时提升
  2. 绑定 = 同位置 × 同时间的 M/P/K 特征被联合增强
  3. 与 Treisman 特征整合理论的注意力绑定假说一致

参考:
  - Treisman & Gelade (1980). A feature-integration theory of attention.
  - 中枢神经系统视觉通路.md §6.2 (绑定问题)
"""

import numpy as np
from typing import Optional


class VisualBinding:
    """FPN 驱动的跨通道特征绑定 (v5.0)."""

    def __init__(self, n_spatial_positions: int = 16, binding_dim: int = 8):
        self.n_positions = n_spatial_positions
        self.binding_dim = binding_dim

        # ---- 空间注意力权重 ----
        self.spatial_attention = np.ones(n_spatial_positions, dtype=np.float32) / n_spatial_positions

        # ---- 通道间关联强度 (M↔P↔K) ----
        self.binding_strength = np.zeros(binding_dim, dtype=np.float32)

    def bind(self, fpn_spatial_focus: np.ndarray,
             channel_outputs: dict) -> np.ndarray:
        """FPN 空间注意力 → 跨通道绑定信号.

        Args:
            fpn_spatial_focus: FPN 的空间注意力权重 (n_positions,)
            channel_outputs: {'M': ..., 'P': ..., 'K': ...}

        Returns:
            binding 向量 (binding_dim,)
        """
        focus_sum = float(np.sum(fpn_spatial_focus)) + 1e-8
        self.spatial_attention = fpn_spatial_focus / focus_sum

        # 对各通道特征计算注意力加权
        M_attn = self._apply_spatial_attention(channel_outputs.get('M'))
        P_attn = self._apply_spatial_attention(channel_outputs.get('P'))
        K_attn = self._apply_spatial_attention(channel_outputs.get('K'))

        # 绑定强度 = 同空间位置 × 同时间的跨通道特征联合激活
        cross_MP = 0.0
        cross_PK = 0.0

        if M_attn is not None and P_attn is not None:
            m_seg = M_attn[:self.binding_dim]
            p_seg = P_attn[:self.binding_dim]
            denom = np.linalg.norm(m_seg) * np.linalg.norm(p_seg) + 1e-8
            cross_MP = float(np.dot(m_seg, p_seg) / denom)

        if P_attn is not None and K_attn is not None:
            p_seg2 = P_attn[:self.binding_dim]
            k_seg = K_attn[:self.binding_dim]
            denom = np.linalg.norm(p_seg2) * np.linalg.norm(k_seg) + 1e-8
            cross_PK = float(np.dot(p_seg2, k_seg) / denom)

        # 绑定向量 = 空间注意力 + 跨通道协调
        spatial_summary = self.spatial_attention[:self.binding_dim]
        cross_summary = np.array([
            cross_MP, cross_PK,
            cross_MP * cross_PK,
            float(np.mean(self.spatial_attention)),
            0.0, 0.0, 0.0, 0.0
        ], dtype=np.float32)

        self.binding_strength = np.tanh(spatial_summary + cross_summary)
        return self.binding_strength

    def _apply_spatial_attention(self, channel_output) -> Optional[np.ndarray]:
        """对通道输出应用空间注意力加权."""
        if channel_output is None:
            return None
        ch_len = len(channel_output)
        n = min(self.n_positions, ch_len)
        attn = np.zeros(ch_len, dtype=np.float32)
        for i in range(n):
            attn[i::n] = channel_output[i::n] * self.spatial_attention[i]
        return attn + channel_output * 0.5  # 50% 原信号 + 50% 注意力调制
