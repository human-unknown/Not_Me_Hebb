"""
v2.py — 次级视皮层 V2 (Secondary Visual Cortex) [v5.0]

对应脑区: BA18 (旁纹状皮层, V2)
所属层级: 大脑 → 枕叶 → V2

v5.0 三类条纹:
  粗条纹 (Thick, M→):  方向池化 + 运动对比度 → 向 MT 前馈
  苍白条纹 (Pale, P→):  空间频率交互 + 共线促进 → 向 V4 前馈
  细条纹 (Thin, K→):   颜色恒常性初步 → 向 V4 前馈

条纹间横向连接 (关键创新):
  粗条纹 ↔ 苍白条纹: 运动-形状绑定 (共同命运律基础)
  细条纹 ↔ 苍白条纹: 颜色-形状绑定

双向连接:
  前馈: V1 各层 → V2 对应条纹 → MT / V4
  反馈: MT → V2 粗条纹, V4 → V2 苍白+细条纹 → V1

参考:
  - 中枢神经系统视觉通路.md §3.4 (V2 条纹组织)
  - Livingstone & Hubel (1988). Segregation of form, color, movement, and depth.
"""

import numpy as np
from typing import Optional


class V2:
    """BA18 旁纹状皮层 — v5.0 三类条纹模块."""

    def __init__(self, M_dim: int = 512, P_dim: int = 1024, K_dim: int = 768,
                 grid_size: int = 4):
        self.M_dim = M_dim
        self.P_dim = P_dim
        self.K_dim = K_dim
        self.grid_size = grid_size

        # ---- 粗条纹 (Thick, M): 方向池化 → 运动对比 ----
        self.thick_dim = min(128, max(32, M_dim // 4))
        self._W_M_thick = np.random.randn(self.thick_dim,
                                           min(512, M_dim)).astype(np.float32) * 0.01

        # ---- 苍白条纹 (Pale, P): 空间频率交互 ----
        self.pale_dim = min(128, max(64, P_dim // 8))
        self._W_P_pale = np.random.randn(self.pale_dim,
                                          min(1024, P_dim)).astype(np.float32) * 0.01

        # ---- 细条纹 (Thin, K): 颜色恒常 ----
        self.thin_dim = min(64, max(32, K_dim // 8))
        self._W_K_thin = np.random.randn(self.thin_dim,
                                          min(768, K_dim)).astype(np.float32) * 0.01

        # ---- 条纹间横向连接 ----
        # 粗→苍白: 运动信息传递给形状分析 (共同命运律的基础)
        self._W_thick_to_pale = np.random.randn(self.pale_dim,
                                                  self.thick_dim).astype(np.float32) * 0.005
        # 细→苍白: 颜色边界传递给形状分析
        self._W_thin_to_pale = np.random.randn(self.pale_dim,
                                                 self.thin_dim).astype(np.float32) * 0.005

        # ---- 反馈预测缓存 ----
        self._mt_feedback: Optional[np.ndarray] = None    # MT → 粗条纹
        self._v4_feedback_P: Optional[np.ndarray] = None  # V4 → 苍白条纹
        self._v4_feedback_K: Optional[np.ndarray] = None  # V4 → 细条纹

        # ---- 预测误差缓存 ----
        self.PE_thick: Optional[np.ndarray] = None
        self.PE_pale: Optional[np.ndarray] = None
        self.PE_thin: Optional[np.ndarray] = None

        # ---- Hebb 学习率 ----
        self.lr: float = 0.001

    # ================================================================
    # 前馈
    # ================================================================

    def feedforward(self, v1_output: dict) -> dict:
        """V1 输出 → V2 三类条纹编码.

        Args:
            v1_output: {'M_V1': ..., 'P_V1': ..., 'K_V1': ...}

        Returns:
            dict with keys 'thick', 'pale', 'thin'
        """
        M_v1 = v1_output.get('M_V1', np.zeros(self.M_dim, dtype=np.float32))
        P_v1 = v1_output.get('P_V1', np.zeros(self.P_dim, dtype=np.float32))
        K_v1 = v1_output.get('K_V1', np.zeros(self.K_dim, dtype=np.float32))

        # 粗条纹 (M → Thick): 方向池化
        M_v1_pad = self._pad_or_trunc(M_v1, self._W_M_thick.shape[1])
        thick_raw = np.tanh(self._W_M_thick @ M_v1_pad)

        # 细条纹 (K → Thin): 颜色编码
        K_v1_pad = self._pad_or_trunc(K_v1, self._W_K_thin.shape[1])
        thin_raw = np.tanh(self._W_K_thin @ K_v1_pad)

        # 苍白条纹 (P → Pale): 空间频率交互 + 横向调制
        P_v1_pad = self._pad_or_trunc(P_v1, self._W_P_pale.shape[1])
        pale_from_P = np.tanh(self._W_P_pale @ P_v1_pad)
        # 粗条纹→苍白条纹 横向调制 (运动→形状)
        pale_lateral_M = self._W_thick_to_pale @ thick_raw
        # 细条纹→苍白条纹 横向调制 (颜色→形状)
        pale_lateral_K = self._W_thin_to_pale @ thin_raw
        pale_raw = np.tanh(pale_from_P + 0.3 * pale_lateral_M + 0.2 * pale_lateral_K)

        return {
            'thick': thick_raw.astype(np.float32),
            'pale': pale_raw.astype(np.float32),
            'thin': thin_raw.astype(np.float32),
        }

    # ================================================================
    # 反馈 (自上而下 + 向下预测)
    # ================================================================

    def receive_feedback_from_MT(self, mt_prediction: np.ndarray):
        """MT → V2 粗条纹: 运动预期 (共同命运律)."""
        self._mt_feedback = mt_prediction

    def receive_feedback_from_V4(self, v4_prediction: dict):
        """V4 → V2 苍白+细条纹: 形状和颜色预期."""
        self._v4_feedback_P = v4_prediction.get('P')
        self._v4_feedback_K = v4_prediction.get('K')

    def predict_to_V1(self, current_output: dict) -> dict:
        """V2 → V1 自上而下预测 (条纹映射到 V1 对应层).

        Returns:
            dict with 'M', 'P', 'K' predictions for V1
        """
        thick = current_output['thick']
        pale = current_output['pale']
        thin = current_output['thin']

        pred_M = (self._W_M_thick.T @ thick)[:self.M_dim]
        pred_P = (self._W_P_pale.T @ pale)[:self.P_dim]
        pred_K = (self._W_K_thin.T @ thin)[:self.K_dim]

        return {'M': pred_M.astype(np.float32),
                'P': pred_P.astype(np.float32),
                'K': pred_K.astype(np.float32)}

    def compute_prediction_error(self, current_output: dict) -> dict:
        """计算 V2 预测误差."""
        thick = current_output['thick']
        pale = current_output['pale']
        thin = current_output['thin']

        if self._mt_feedback is not None:
            fb_len = min(len(self._mt_feedback), len(thick))
            self.PE_thick = np.abs(thick[:fb_len] - self._mt_feedback[:fb_len])
        else:
            self.PE_thick = np.zeros_like(thick)

        if self._v4_feedback_P is not None:
            fb_len = min(len(self._v4_feedback_P), len(pale))
            self.PE_pale = np.abs(pale[:fb_len] - self._v4_feedback_P[:fb_len])
        else:
            self.PE_pale = np.zeros_like(pale)

        if self._v4_feedback_K is not None:
            fb_len = min(len(self._v4_feedback_K), len(thin))
            self.PE_thin = np.abs(thin[:fb_len] - self._v4_feedback_K[:fb_len])
        else:
            self.PE_thin = np.zeros_like(thin)

        return {'thick': self.PE_thick, 'pale': self.PE_pale, 'thin': self.PE_thin}

    # ================================================================
    # 格式塔: 接近律 + 连续律 (V2 内部计算)
    # ================================================================

    def compute_proximity(self, pale_features: np.ndarray) -> np.ndarray:
        """接近律: 基于苍白条纹空间特征的临近度分组.

        Returns:
            (n_cells,) 空间分组标签
        """
        n_cells = self.grid_size * self.grid_size
        chunk = min(len(pale_features), n_cells * 8)
        if chunk < n_cells * 2:
            return np.zeros(n_cells, dtype=np.float32)
        reshaped = pale_features[:chunk].reshape(n_cells, -1)
        proximity = np.zeros(n_cells, dtype=np.float32)
        for i in range(n_cells):
            gy, gx = i // self.grid_size, i % self.grid_size
            neighbors = []
            if gy > 0: neighbors.append(i - self.grid_size)
            if gy < self.grid_size - 1: neighbors.append(i + self.grid_size)
            if gx > 0: neighbors.append(i - 1)
            if gx < self.grid_size - 1: neighbors.append(i + 1)
            if neighbors:
                sim = np.mean([np.dot(reshaped[i], reshaped[n]) /
                              (np.linalg.norm(reshaped[i]) * np.linalg.norm(reshaped[n]) + 1e-8)
                              for n in neighbors])
                proximity[i] = sim
        return np.tanh(proximity)

    def compute_continuity(self, pale_features: np.ndarray) -> np.ndarray:
        """连续律: 沿空间相邻细胞的朝向一致性.

        Returns:
            (n_cells,) 连续性得分
        """
        n_cells = self.grid_size * self.grid_size
        chunk = min(len(pale_features), n_cells * 4)
        if chunk < n_cells:
            return np.zeros(n_cells, dtype=np.float32)
        reshaped = pale_features[:chunk].reshape(n_cells, -1)
        continuity = np.zeros(n_cells, dtype=np.float32)
        for i in range(n_cells):
            gy, gx = i // self.grid_size, i % self.grid_size
            if gx < self.grid_size - 1:
                right = i + 1
                if right < n_cells:
                    sim = np.dot(reshaped[i], reshaped[right]) / \
                          (np.linalg.norm(reshaped[i]) * np.linalg.norm(reshaped[right]) + 1e-8)
                    continuity[i] += sim
            if gy < self.grid_size - 1:
                down = i + self.grid_size
                if down < n_cells:
                    sim = np.dot(reshaped[i], reshaped[down]) / \
                          (np.linalg.norm(reshaped[i]) * np.linalg.norm(reshaped[down]) + 1e-8)
                    continuity[i] += sim
        return np.tanh(continuity)

    # ================================================================
    # Utility
    # ================================================================

    def _pad_or_trunc(self, vec: np.ndarray, target_len: int) -> np.ndarray:
        if len(vec) >= target_len:
            return vec[:target_len]
        out = np.zeros(target_len, dtype=np.float32)
        out[:len(vec)] = vec
        return out
