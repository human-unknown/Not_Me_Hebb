"""
visual_hierarchy.py — 视觉层级管线编排 (Visual Hierarchy Orchestrator) [v5.0]

组装完整视觉管线:
  视网膜 → LGN → V1 → V2 → MT/MST (背侧) + V4 → IT (腹侧)
  → SC → Pulvinar (第二条通路) → 反馈链 → PE → F_accuracy

提供单步 process() 接口供 agent.step() 调用.

v5.0 数据流:
  Phase 1: 前馈 (自下而上) — 图像 → 各脑区编码
  Phase 2: 反馈 (自上而下) — IT→V4→V2→V1, MST→MT→V2
  Phase 3: 预测误差 — 各层 PE 汇总
  Phase 4: FPN 绑定
  Phase 5: 构建感知向量 (D_V5=372)
"""

import numpy as np
from typing import Optional
from cerebrum.thalamus.lgn import LGN
from cerebrum.thalamus.pulvinar import Pulvinar
from cerebrum.occipital_lobe.v1 import V1
from cerebrum.occipital_lobe.v2 import V2
from cerebrum.occipital_lobe.v4 import V4
from cerebrum.temporal_lobe.it_cortex import ITCortex
from cerebrum.temporal_lobe.mt_cortex import MT
from cerebrum.temporal_lobe.mst_cortex import MST
from brainstem_cerebellum.midbrain.superior_colliculus import SuperiorColliculus
from cerebrum.association.visual_binding import VisualBinding
from cns.data_types import (
    D_V5, TEXT_V5_WIDTH,
    M_V1_WIDTH, M_V2_WIDTH, MT_WIDTH, MST_WIDTH,
    M_V1_START, M_V2_START, MT_START, MST_START,
    P_V1_WIDTH, P_V2_WIDTH, V4_SHAPE_WIDTH,
    P_V1_START, P_V2_START, V4_SHAPE_START,
    K_V1_WIDTH, K_V2_WIDTH, V4_COLOR_WIDTH,
    K_V1_START, K_V2_START, V4_COLOR_START,
    IT_WIDTH, IT_START,
    SC_WIDTH, SC_START,
    PULVINAR_WIDTH, PULVINAR_START,
    BINDING_WIDTH, BINDING_START,
)


def _place(arr, start, values):
    """Place values into array at start index."""
    end = min(start + len(values), len(arr))
    arr[start:end] = values[:end - start]


class VisualHierarchy:
    """v5.0 视觉层级管线 — 编排全流程.

    用法:
      vh = VisualHierarchy(image_size=64)
      result = vh.process(image, brainstem_arousal=0.8, fpn=fpn)
      # result['sensory']   → 感知向量 (D_V5=372,)
      # result['F_accuracy'] → 汇入扣带回
      # result['PE_total']   → 总预测误差
    """

    def __init__(self, image_size: int = 64, grid_size: int = 4):
        self.image_size = image_size
        self.grid_size = grid_size

        # ---- 所有视觉脑区模块 ----
        self.lgn = LGN()
        self.v1 = V1(image_size=image_size, grid_size=grid_size)
        self.v2 = V2()
        self.mt = MT()
        self.mst = MST()
        self.v4 = V4()
        self.it = ITCortex()
        self.sc = SuperiorColliculus()
        self.pulvinar = Pulvinar()
        self.binding = VisualBinding()

    def process(self, image: np.ndarray,
                brainstem_arousal: float = 0.8,
                fpn: Optional[object] = None,
                learn: bool = False) -> dict:
        """单步视觉处理 (前馈 + 反馈 + PE + 绑定).

        Args:
            image: (H, W, 3) uint8 图像
            brainstem_arousal: 脑干唤醒度 [0, 1]
            fpn: FPN 模块 (用于注意力调制和绑定, 可选)
            learn: 是否更新 Hebb 权重

        Returns:
            dict with 'sensory' (percept D_V5), 'F_accuracy', 'PE_total',
                 'diagnostics'
        """
        # ==== Phase 1: 自下而上 (前馈) ====

        # LGN + V1 (V1 直接从图像编码)
        lgn_out = self.lgn.relay(
            M_signal=np.zeros(1024, dtype=np.float32),
            P_signal=np.zeros(1024, dtype=np.float32),
            K_signal=np.zeros(1024, dtype=np.float32),
            brainstem_arousal=brainstem_arousal,
        )
        v1_out = self.v1.feedforward(lgn_out, image=image, learn=learn)

        # V2 三类条纹
        v2_out = self.v2.feedforward(v1_out)

        # 背侧通路: V1→MT→MST
        mt_out = self.mt.feedforward(v1_out['M_V1'], v2_out['thick'])
        mst_out = self.mst.feedforward(mt_out)

        # 腹侧通路: V2→V4→IT
        v4_out = self.v4.feedforward(v2_out)
        it_out = self.it.feedforward(v4_out)

        # 第二条通路: SC → Pulvinar
        sc_out = self.sc.feedforward(v1_out['SC'])
        pulvinar_out = self.pulvinar.relay(sc_out)

        # IT 学习
        if learn:
            self.it.learn(v4_out['convergence'])

        # ==== Phase 2: 自上而下 (反馈预测) ====

        # IT → V4
        it_pred = self.it.predict_to_V4(it_out)
        self.v4.receive_feedback_from_IT(it_pred)

        # V4 → V2
        v4_pred = self.v4.predict_to_V2(v4_out)
        self.v2.receive_feedback_from_V4(v4_pred)

        # MST → MT
        mst_pred = self.mst.predict_to_MT(mst_out)
        self.mt.receive_feedback_from_MST(mst_pred)

        # MT → V2 (共同命运律)
        mt_pred_v2 = self.mt.predict_to_V2(mt_out)
        self.v2.receive_feedback_from_MT(mt_pred_v2)

        # V2 → V1
        v2_pred = self.v2.predict_to_V1(v2_out)
        self.v1.receive_feedback(v2_pred)

        # ==== Phase 3: 预测误差 ====

        pe_v1 = self.v1.compute_prediction_error(v1_out)
        pe_v2 = self.v2.compute_prediction_error(v2_out)
        pe_v4 = self.v4.compute_prediction_error(v4_out)
        pe_mt = self.mt.compute_prediction_error(mt_out)
        pe_mst = self.mst.compute_prediction_error(mst_out, mt_out)
        pe_it = self.it.compute_prediction_error(it_out)

        # F_accuracy: 所有层级 PE 的加权和
        F_accuracy = (
            float(np.mean(np.abs(pe_v1['M'])) + np.mean(np.abs(pe_v1['P'])) +
                  np.mean(np.abs(pe_v1['K']))) * 0.3 +
            float(np.mean(np.abs(pe_v2['thick'])) + np.mean(np.abs(pe_v2['pale'])) +
                  np.mean(np.abs(pe_v2['thin']))) * 0.5 +
            float(np.mean(np.abs(pe_v4['shape'])) + np.mean(np.abs(pe_v4['color']))) * 0.7 +
            float(np.mean(np.abs(pe_mt))) * 0.4 +
            float(np.mean(np.abs(pe_mst))) * 0.3 +
            float(np.mean(np.abs(pe_it))) * 1.0
        )

        # ==== Phase 4: FPN 绑定 ====

        if fpn is not None and hasattr(fpn, 'compute_spatial_focus'):
            v1_concat = np.concatenate([
                v1_out['M_V1'][:32], v1_out['P_V1'][:32]
            ])
            fpn_focus = fpn.compute_spatial_focus(v1_concat)
            binding_vec = self.binding.bind(fpn_focus, {
                'M': v1_out['M_V1'],
                'P': v1_out['P_V1'],
                'K': v1_out['K_V1'],
            })
        else:
            binding_vec = np.zeros(BINDING_WIDTH, dtype=np.float32)

        # ==== Phase 5: 构建感知向量 (D_V5=372) ====

        sensory = np.zeros(D_V5, dtype=np.float32)

        # text[0:64] 由上层填充 (此处留空)

        # M 通路
        _place(sensory, M_V1_START, _trunc(v1_out['M_V1'], M_V1_WIDTH))
        _place(sensory, M_V2_START, _trunc(v2_out['thick'], M_V2_WIDTH))
        _place(sensory, MT_START, _trunc(mt_out['direction_energy'], MT_WIDTH))
        _place(sensory, MST_START, _trunc(mst_out['flow_patterns'], MST_WIDTH))

        # P 通路
        _place(sensory, P_V1_START, _trunc(v1_out['P_V1'], P_V1_WIDTH))
        _place(sensory, P_V2_START, _trunc(v2_out['pale'], P_V2_WIDTH))
        _place(sensory, V4_SHAPE_START, _trunc(v4_out['shape'], V4_SHAPE_WIDTH))

        # K 通路
        _place(sensory, K_V1_START, _trunc(v1_out['K_V1'], K_V1_WIDTH))
        _place(sensory, K_V2_START, _trunc(v2_out['thin'], K_V2_WIDTH))
        _place(sensory, V4_COLOR_START, _trunc(v4_out['color'], V4_COLOR_WIDTH))

        # IT + SC + Pulvinar + Binding
        _place(sensory, IT_START, _trunc(it_out['object_code'], IT_WIDTH))
        _place(sensory, SC_START, _trunc(sc_out['saliency_map'], SC_WIDTH))
        _place(sensory, PULVINAR_START, _trunc(pulvinar_out, PULVINAR_WIDTH))
        _place(sensory, BINDING_START, _trunc(binding_vec, BINDING_WIDTH))

        return {
            'sensory': sensory,
            'F_accuracy': F_accuracy,
            'PE_total': float(
                np.mean(np.abs(pe_v1['M'])) + np.mean(np.abs(pe_v1['P'])) +
                np.mean(np.abs(pe_v2['thick'])) + np.mean(np.abs(pe_v2['pale'])) +
                np.mean(np.abs(pe_v4['shape'])) + np.mean(np.abs(pe_it))
            ),
            'diagnostics': {
                'v1_pe': {k: float(np.mean(np.abs(v))) for k, v in pe_v1.items()
                          if v is not None and len(v) > 0},
                'v2_pe': {k: float(np.mean(np.abs(v))) for k, v in pe_v2.items()
                          if v is not None and len(v) > 0},
                'v4_pe': {k: float(np.mean(np.abs(v))) for k, v in pe_v4.items()
                          if v is not None and len(v) > 0},
                'mt_pe': float(np.mean(np.abs(pe_mt))),
                'it_pe': float(np.mean(np.abs(pe_it))),
                'sc_novelty': float(sc_out['novelty']),
            },
        }


def _trunc(vec: np.ndarray, target_len: int) -> np.ndarray:
    """Truncate or zero-pad a vector to target length."""
    if len(vec) >= target_len:
        return vec[:target_len]
    out = np.zeros(target_len, dtype=np.float32)
    out[:len(vec)] = vec
    return out
