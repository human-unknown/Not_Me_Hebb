"""
superior_colliculus.py — 上丘 (Superior Colliculus) [v5.0]

对应脑区: 上丘 (视顶盖, Optic Tectum)
所属层级: 脑干 → 中脑 → 上丘

v5.0 职责:
  - 显著性图: 空间对比度 + 时间变化
  - 快速空间定向 (不经皮层的皮层下通路)
  - 输出到丘脑枕 (Pulvinar) → 第二条视觉通路

参考:
  - 中枢神经系统视觉通路.md §1.3 (视束靶区)
  - Sparks (1986). Translation of sensory signals into saccadic eye movements.
"""

import numpy as np
from typing import Optional


class SuperiorColliculus:
    """上丘 — 快速显著性检测 + 空间定向 (v5.0).

    浅层: 视觉输入, retinotopic map
    深层: 多感官整合 + 眼动控制
    """

    def __init__(self, spatial_cells: int = 16, sc_dim: int = 16):
        self.spatial_cells = spatial_cells
        self.sc_dim = sc_dim

        # ---- 显著性图 (空间) ----
        self.saliency_map = np.zeros(spatial_cells, dtype=np.float32)

        # ---- 时序缓存 (新颖性检测) ----
        self._prev_saliency: Optional[np.ndarray] = None

    def feedforward(self, v1_sc_output: np.ndarray,
                    retinal_M: Optional[np.ndarray] = None) -> dict:
        """V1 第5层 + 视网膜 M 通路 → 显著性图.

        Args:
            v1_sc_output: V1 第5层输出 (空间显著性)
            retinal_M: 视网膜 M 通路直达信号 (不经 LGN 的快速通路)

        Returns:
            dict with 'saliency_map', 'novelty', 'attention_shift'
        """
        # V1 第5层空间信号
        sc_sig = self._pad_or_trunc(v1_sc_output, self.spatial_cells)

        # 视网膜快速通路 (不经 LGN)
        if retinal_M is not None:
            chunk = self.spatial_cells * 8
            retinal_spatial = np.abs(retinal_M[:min(chunk, len(retinal_M))])
            if len(retinal_spatial) >= self.spatial_cells:
                retinal_spatial = retinal_spatial.reshape(self.spatial_cells, -1).mean(axis=1)
            else:
                retinal_spatial = self._pad_or_trunc(retinal_spatial, self.spatial_cells)
            retinal_spatial = retinal_spatial / (np.linalg.norm(retinal_spatial) + 1e-8)
            combined = 0.6 * sc_sig + 0.4 * retinal_spatial
        else:
            combined = sc_sig

        # 显著性图
        self.saliency_map = combined / (np.linalg.norm(combined) + 1e-8)

        # 新颖性: 显著性图的变化量
        if self._prev_saliency is not None:
            novelty = float(np.linalg.norm(self.saliency_map - self._prev_saliency))
        else:
            novelty = 0.0

        self._prev_saliency = self.saliency_map.copy()

        # 注意定向: 最大显著性的空间位置
        max_idx = int(np.argmax(self.saliency_map))
        attention_shift = np.zeros(self.spatial_cells, dtype=np.float32)
        attention_shift[max_idx] = 1.0

        return {
            'saliency_map': self.saliency_map.astype(np.float32),
            'novelty': np.float32(novelty),
            'attention_shift': attention_shift,
        }

    def _pad_or_trunc(self, vec: np.ndarray, target_len: int) -> np.ndarray:
        if len(vec) >= target_len:
            return vec[:target_len]
        out = np.zeros(target_len, dtype=np.float32)
        out[:len(vec)] = vec
        return out
