"""
mt_cortex.py — 中颞区 MT/V5 (Middle Temporal Area) [v5.0]

对应脑区: MT (V5, 中颞区)
所属层级: 大脑 → 颞叶 → MT

功能职责 (v5.0):
  - 方向选择性柱状组织 — 8 方向通道
  - 运动能量计算 — 当前帧 vs 前一帧 (方向能量差异)
  - 接受 V1-4B (M 输出) + V2 粗条纹
  - 反馈到 V2 粗条纹 (共同命运律: "一起动的属于一组")
  - 前馈到 MST

参考:
  - 中枢神经系统视觉通路.md §5.1 (背侧通路)
  - Salzman et al. (1990). Cortical microstimulation and motion perception.
"""

import numpy as np
from typing import Optional


class MT:
    """MT/V5 — 方向选择性运动检测 (v5.0).

    输入: V1-4B M 通路输出 + V2 粗条纹输出
    输出: 8 方向运动能量图 + 运动对比度
    """

    def __init__(self, n_directions: int = 8):
        self.n_directions = n_directions

        # ---- 方向选择性投影 (V1-M + V2-thick → 方向能量) ----
        self.input_dim = 256  # 组合 V1-M(128) + V2-thick(128)
        self._W_direction = np.random.randn(n_directions,
                                              self.input_dim).astype(np.float32) * 0.01

        # ---- 时序状态 (运动检测需要帧间差异) ----
        self._prev_direction_energy: Optional[np.ndarray] = None

        # ---- MST 反馈 ----
        self._mst_feedback: Optional[np.ndarray] = None

        # ---- 预测误差 ----
        self.PE: Optional[np.ndarray] = None

    def feedforward(self, v1_M: np.ndarray, v2_thick: np.ndarray) -> dict:
        """V1-4B + V2 粗条纹 → 方向选择 + 运动能量.

        Args:
            v1_M: V1 4B 层输出 (M 通路)
            v2_thick: V2 粗条纹输出

        Returns:
            dict with 'direction_energy' (n_directions,),
                     'motion_contrast' (n_directions,),
                     'dir_encoded' (n_directions,)
        """
        # 合并 V1-M 和 V2-粗条纹
        combined = np.concatenate([
            self._pad_or_trunc(v1_M, 128),
            self._pad_or_trunc(v2_thick, 128),
        ]).astype(np.float32)

        # 方向选择性编码
        dir_encoded = np.tanh(self._W_direction @ combined[:self.input_dim])
        direction_energy = dir_encoded  # (n_directions,)

        # 运动对比度: 帧间方向能量差异
        if self._prev_direction_energy is not None:
            motion_contrast = np.abs(direction_energy - self._prev_direction_energy)
        else:
            motion_contrast = np.zeros(self.n_directions, dtype=np.float32)

        self._prev_direction_energy = direction_energy.copy()

        return {
            'direction_energy': direction_energy.astype(np.float32),
            'motion_contrast': motion_contrast.astype(np.float32),
            'dir_encoded': dir_encoded.astype(np.float32),
        }

    def predict_to_V2(self, current_output: dict) -> np.ndarray:
        """MT → V2 粗条纹: 运动预期.

        "这些空间位置的点具有同向运动 → 应归为一组"

        这是共同命运律的关键: MT 告诉 V2 哪些特征一起运动.
        """
        direction_energy = current_output['direction_energy']
        motion_contrast = current_output['motion_contrast']

        # 运动预期 = 方向能量 × (1 + 运动对比度增强)
        prediction = direction_energy * (1.0 + np.tanh(motion_contrast))
        # 返回与 V2 粗条纹维度匹配的预测
        pred_padded = np.zeros(128, dtype=np.float32)
        pred_padded[:self.n_directions] = prediction
        return pred_padded

    def receive_feedback_from_MST(self, mst_prediction: np.ndarray):
        """MST → MT: 光流连贯性预期."""
        self._mst_feedback = mst_prediction

    def compute_prediction_error(self, current_output: dict) -> np.ndarray:
        """MT 预测误差 = |实际方向能量 - MST 预期|."""
        direction_energy = current_output['direction_energy']
        if self._mst_feedback is not None:
            fb_len = min(len(self._mst_feedback), len(direction_energy))
            self.PE = np.abs(direction_energy[:fb_len] - self._mst_feedback[:fb_len])
        else:
            self.PE = np.zeros_like(direction_energy)
        return self.PE

    def _pad_or_trunc(self, vec: np.ndarray, target_len: int) -> np.ndarray:
        if len(vec) >= target_len:
            return vec[:target_len]
        out = np.zeros(target_len, dtype=np.float32)
        out[:len(vec)] = vec
        return out
