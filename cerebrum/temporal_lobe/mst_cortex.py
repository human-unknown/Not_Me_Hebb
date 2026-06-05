"""
mst_cortex.py — 内上颞区 MST (Medial Superior Temporal Area) [v5.0]

对应脑区: MST (内上颞区)
所属层级: 大脑 → 颞叶 → MST

功能职责 (v5.0):
  - 光流模式检测 — 扩张/收缩/旋转/平移
  - 接受 MT 方向能量
  - 反馈到 MT (运动连贯性预期)
  - 自身运动感知

参考:
  - 中枢神经系统视觉通路.md §5.1 (背侧通路)
"""

import numpy as np
from typing import Optional


class MST:
    """MST — 光流模式检测 (v5.0).

    输入: MT 方向能量 (8 方向)
    输出: 4 种光流模式强度 (expansion, contraction, rotation, translation)
    """

    def __init__(self, n_directions: int = 8):
        self.n_directions = n_directions

        # ---- 光流模式: 8 方向 → 4 模式 (expansion/contraction/rotation/translation) ----
        self.n_patterns = 4
        self._W_flow = np.random.randn(self.n_patterns,
                                        n_directions).astype(np.float32) * 0.1

        # ---- 反馈缓存 ----
        self._ppc_feedback: Optional[np.ndarray] = None

        # ---- 预测误差 ----
        self.PE: Optional[np.ndarray] = None

    def feedforward(self, mt_output: dict) -> dict:
        """MT 方向能量 → 光流模式.

        Args:
            mt_output: MT.feedforward() 输出

        Returns:
            dict with 'flow_patterns' (4,), 'dominant_flow' (int)
        """
        direction_energy = mt_output['direction_energy']
        dir_e = self._pad_or_trunc(direction_energy, self.n_directions)

        # 光流模式: 方向能量 × 光流模板
        flow_patterns = np.tanh(self._W_flow @ dir_e)
        dominant = int(np.argmax(np.abs(flow_patterns)))

        return {
            'flow_patterns': flow_patterns.astype(np.float32),
            'dominant_flow': dominant,
        }

    def predict_to_MT(self, current_output: dict) -> np.ndarray:
        """MST → MT: 光流连贯性预期 — "这些方向变化是一致的"."""
        flow_patterns = current_output['flow_patterns']
        # 反投影: 光流模式 → 方向能量预期
        prediction = self._W_flow.T @ flow_patterns
        return prediction.astype(np.float32)

    def compute_prediction_error(self, current_output: dict,
                                  mt_output: dict) -> np.ndarray:
        """MST 预测误差 = |实际方向能量 - 反投影重建|."""
        flow_patterns = current_output['flow_patterns']
        direction_energy = mt_output['direction_energy']
        dir_e = self._pad_or_trunc(direction_energy, self.n_directions)

        # PE = 方向能量与光流模式反投影的不匹配度
        reconstructed = self._W_flow.T @ flow_patterns
        self.PE = np.abs(dir_e[:self.n_directions] - reconstructed[:self.n_directions])
        return self.PE

    def _pad_or_trunc(self, vec: np.ndarray, target_len: int) -> np.ndarray:
        if len(vec) >= target_len:
            return vec[:target_len]
        out = np.zeros(target_len, dtype=np.float32)
        out[:len(vec)] = vec
        return out
