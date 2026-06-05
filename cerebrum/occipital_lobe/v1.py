"""
v1.py — 初级视皮层 V1 (Primary Visual Cortex / Striate Cortex) [v5.0]

对应脑区: BA17 (纹状皮层, V1)
所属层级: 大脑 → 枕叶 → V1

v5.0 层状结构:
  4Cα:    M 通路入口 → 运动/方向滤波 → 4B 层
  4Cβ:    P 通路入口 → 空间频率/朝向滤波 → 2/3 斑点间区
  2/3 斑块 (blobs):    K 通路入口 → 颜色拮抗
  2/3 斑点间区 (interblobs): P 通路的朝向选择性
  4B:     M 通路出口 → MT + V2 粗条纹
  5:      输出到上丘 (眼球运动)
  6:      反馈到 LGN (皮层-膝状体门控)

双向连接:
  前馈: LGN → 4Cα/4Cβ/2/3斑块 → 输出到 V2/MT
  反馈: V2 条纹预测 → 调制 4C 层增益, V1→LGN 反馈

参考:
  - 中枢神经系统视觉通路.md §3-4
  - Hubel & Wiesel (1962). Receptive fields, binocular interaction.
"""

import numpy as np
from typing import Optional
from cerebrum.occipital_lobe.visual_pathway import GaborFilterBank


class V1:
    """BA17 纹状皮层 — v5.0 层状模块.

    输入: LGN 门控后的 M/P/K 信号 (或直接从图像编码)
    输出: 按通路 × 层组织:
      - M_V1: 4B 层方向选择信号 (→ MT + V2 粗条纹)
      - P_V1: 2/3 斑点间区朝向信号 (→ V2 苍白条纹)
      - K_V1: 2/3 斑块颜色信号 (→ V2 细条纹)
      - SC:    第 5 层上丘输出
      - LGN_fb: 第 6 层 LGN 反馈信号
    """

    def __init__(self, image_size: int = 64, grid_size: int = 4):
        self.image_size = image_size
        self.grid_size = grid_size
        self._gabor = GaborFilterBank(image_size=image_size, grid_size=grid_size)

        # ---- V2 反馈预测缓存 (自上而下) ----
        self._v2_feedback_M: Optional[np.ndarray] = None   # 来自 V2 粗条纹
        self._v2_feedback_P: Optional[np.ndarray] = None   # 来自 V2 苍白条纹
        self._v2_feedback_K: Optional[np.ndarray] = None   # 来自 V2 细条纹

        # ---- 预测误差缓存 ----
        self.PE_M: Optional[np.ndarray] = None
        self.PE_P: Optional[np.ndarray] = None
        self.PE_K: Optional[np.ndarray] = None

        # ---- FPN 增益 (按通道) ----
        self.gain_M: float = 1.0
        self.gain_P: float = 1.0
        self.gain_K: float = 1.0

    # ================================================================
    # 前馈 (自下而上)
    # ================================================================

    def feedforward(self, lgn_output: dict, image: Optional[np.ndarray] = None,
                    learn: bool = False) -> dict:
        """LGN 门控信号 → V1 层状编码 (或直接从图像编码).

        Args:
            lgn_output: LGN.relay() 的输出 {'M', 'P', 'K'}
            image: 原始图像 (可选, 用于直接从图像编码, 绕过 LGN)
            learn: 是否更新 Hebb 增益

        Returns:
            dict with keys:
              'M_V1': V1 4B 层 M 通路输出 (→ MT)
              'P_V1': V1 2/3 斑点间区 P 通路输出 (→ V2 苍白条纹)
              'K_V1': V1 2/3 斑块 K 通路输出 (→ V2 细条纹)
              'SC':   V1 第 5 层上丘输出
              'LGN_fb': V1 第 6 层 LGN 反馈信号
        """
        # 如果提供了原始图像, 直接用 Gabor 编码
        if image is not None:
            M_raw = self._gabor.encode_M(image, learn=learn)
            P_raw = self._gabor.encode_P(image, learn=learn)
            K_raw = self._gabor.encode_K(image, learn=learn)
        else:
            M_raw = lgn_output.get('M', np.zeros(1024, dtype=np.float32))
            P_raw = lgn_output.get('P', np.zeros(1024, dtype=np.float32))
            K_raw = lgn_output.get('K', np.zeros(1024, dtype=np.float32))

        # 4Cα → 4B: M 通路方向编码 (低空间频率, 快速)
        M_pooled = self._pool_spatial(M_raw, 0.5)
        # 4Cβ → 2/3 斑点间区: P 通路朝向编码 (高空间频率)
        P_pooled = self._pool_spatial(P_raw, 1.0)
        # 2/3 斑块: K 通路颜色编码
        K_pooled = self._pool_spatial(K_raw, 0.75)

        # FPN 增益调制
        M_pooled = M_pooled * self.gain_M
        P_pooled = P_pooled * self.gain_P
        K_pooled = K_pooled * self.gain_K

        # 第 5 层: 上丘输出 (显著性引导的空间定向)
        sc_out = self._layer5_output(M_pooled, P_pooled)

        # 第 6 层: LGN 反馈信号
        lgn_fb = self._compute_lgn_feedback(M_pooled, P_pooled, K_pooled)

        return {
            'M_V1': M_pooled.astype(np.float32),
            'P_V1': P_pooled.astype(np.float32),
            'K_V1': K_pooled.astype(np.float32),
            'SC': sc_out.astype(np.float32),
            'LGN_fb': lgn_fb.astype(np.float32),
        }

    def _pool_spatial(self, raw: np.ndarray, detail_factor: float) -> np.ndarray:
        """空间池化, detail_factor 控制保留度 (1.0=全细节, 0.0=粗化).

        M 通路 (0.5): 相邻 kernel 合并 → 运动/方向检测优化
        P 通路 (1.0): 保持细节 → 精细形状分析
        K 通路 (0.75): 中等 → 颜色恒常
        """
        n_cells = self.grid_size * self.grid_size
        total = len(raw)
        n_kernels = total // (n_cells * 2)
        if n_kernels <= 1:
            return raw

        reshaped = raw[:n_kernels * n_cells * 2].reshape(n_kernels, n_cells, 2)

        if detail_factor < 1.0:
            n_coarse = max(2, int(n_kernels * detail_factor))
            step = max(1, n_kernels // n_coarse)
            pooled = np.zeros((n_coarse, n_cells, 2), dtype=np.float32)
            for i in range(n_coarse):
                start = i * step
                end = min(start + step, n_kernels)
                if end > start:
                    pooled[i] = reshaped[start:end].mean(axis=0)
            return pooled.reshape(-1)
        return raw

    def _layer5_output(self, M_pooled: np.ndarray,
                       P_pooled: np.ndarray) -> np.ndarray:
        """V1 第 5 层 → 上丘: 空间显著性信号."""
        n_cells = self.grid_size * self.grid_size
        M_resp = np.abs(M_pooled).reshape(-1, n_cells, 2).mean(axis=(0, 2))
        P_resp = np.abs(P_pooled).reshape(-1, n_cells, 2).mean(axis=(0, 2))

        M_resp = self._pad_or_trunc(M_resp, n_cells)
        P_resp = self._pad_or_trunc(P_resp, n_cells)

        sc = (M_resp + P_resp) / 2.0
        sc = sc / (np.linalg.norm(sc) + 1e-8)
        return sc.astype(np.float32)

    def _compute_lgn_feedback(self, M, P, K) -> np.ndarray:
        """V1 第 6 层 → LGN 反馈: 7维增益信号 [M0,M1, P0,P1,P2,P3, K0]."""
        M_act = float(np.mean(np.abs(M))) if len(M) > 0 else 0.0
        P_act = float(np.mean(np.abs(P))) if len(P) > 0 else 0.0
        K_act = float(np.mean(np.abs(K))) if len(K) > 0 else 0.0

        m_fb = np.clip(M_act * 2.0, -1.0, 1.0)
        p_fb = np.clip(P_act * 2.0, -1.0, 1.0)
        k_fb = np.clip(K_act * 2.0, -1.0, 1.0)

        return np.array([m_fb, m_fb, p_fb, p_fb, p_fb, p_fb, k_fb],
                        dtype=np.float32)

    # ================================================================
    # 反馈 (自上而下): 接收 V2 预测, 计算 PE
    # ================================================================

    def receive_feedback(self, v2_feedback: dict):
        """接收 V2 自上而下的预测信号.

        Args:
            v2_feedback: {'M': prediction_M, 'P': prediction_P, 'K': prediction_K}
        """
        self._v2_feedback_M = v2_feedback.get('M')
        self._v2_feedback_P = v2_feedback.get('P')
        self._v2_feedback_K = v2_feedback.get('K')

    def compute_prediction_error(self, current_output: dict) -> dict:
        """计算 V1 各通道的预测误差.

        PE = |feedforward - feedback_prediction|

        Returns:
            dict with 'M', 'P', 'K' prediction errors
        """
        M_current = current_output.get('M_V1')
        P_current = current_output.get('P_V1')
        K_current = current_output.get('K_V1')

        # M 通道 PE
        if self._v2_feedback_M is not None and M_current is not None:
            fb_len = min(len(self._v2_feedback_M), len(M_current))
            self.PE_M = np.abs(M_current[:fb_len] - self._v2_feedback_M[:fb_len])
        else:
            self.PE_M = np.zeros(1 if M_current is None else min(1, len(M_current)),
                                 dtype=np.float32)

        # P 通道 PE
        if self._v2_feedback_P is not None and P_current is not None:
            fb_len = min(len(self._v2_feedback_P), len(P_current))
            self.PE_P = np.abs(P_current[:fb_len] - self._v2_feedback_P[:fb_len])
        else:
            self.PE_P = np.zeros(1 if P_current is None else min(1, len(P_current)),
                                 dtype=np.float32)

        # K 通道 PE
        if self._v2_feedback_K is not None and K_current is not None:
            fb_len = min(len(self._v2_feedback_K), len(K_current))
            self.PE_K = np.abs(K_current[:fb_len] - self._v2_feedback_K[:fb_len])
        else:
            self.PE_K = np.zeros(1 if K_current is None else min(1, len(K_current)),
                                 dtype=np.float32)

        return {'M': self.PE_M, 'P': self.PE_P, 'K': self.PE_K}

    # ================================================================
    # FPN 增益调制
    # ================================================================

    def set_gain(self, gain_M: float = 1.0, gain_P: float = 1.0,
                 gain_K: float = 1.0):
        """FPN 探照灯调制各通道增益."""
        self.gain_M = gain_M
        self.gain_P = gain_P
        self.gain_K = gain_K

    # ================================================================
    # Hebb 可塑性 + 诊断 (委托给 GaborFilterBank)
    # ================================================================

    def get_gain_profile(self) -> dict:
        return self._gabor.get_gain_profile()

    def reset_gains(self):
        self._gabor.reset_gains()

    @property
    def n_filters(self) -> int:
        return self._gabor.n_filters

    # ================================================================
    # Utility
    # ================================================================

    def _pad_or_trunc(self, vec: np.ndarray, target_len: int) -> np.ndarray:
        if len(vec) >= target_len:
            return vec[:target_len]
        out = np.zeros(target_len, dtype=np.float32)
        out[:len(vec)] = vec
        return out
