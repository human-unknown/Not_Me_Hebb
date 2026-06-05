"""
auditory_hierarchy.py — 听觉层级管线编排 (Auditory Hierarchy Orchestrator) [v5.2]

组装完整听觉管线:
  CochlearNucleus → SuperiorOlive → LateralLemniscus →
  InferiorColliculus → MGB → AuditoryCortex (A1→Belt→Parabelt)

提供单步 process() 接口供 agent.step() 调用.

v5.2 数据流:
  Phase 1: 前馈 (自下而上) — 频谱 → 各脑区编码
  Phase 2: 反馈 (自上而下) — AuditoryCortex→MGB→IC
  Phase 3: 预测误差 — 各层 PE 汇总 → F_accuracy
  Phase 4: 构建感知向量 (D_V52=468)

双模式支持:
  - 真实频谱输入: 当音频数据可用时
  - 语义代理模式: 从语义向量合成伪频谱 (无音频时)
"""

import numpy as np
from typing import Optional
from brainstem_cerebellum.pons.cochlear_nucleus import CochlearNucleus
from brainstem_cerebellum.pons.superior_olivary import SuperiorOlive
from brainstem_cerebellum.pons.lateral_lemniscus import LateralLemniscus
from brainstem_cerebellum.midbrain.inferior_colliculus import InferiorColliculus
from cerebrum.thalamus.mgb import MedialGeniculateBody
from cerebrum.temporal_lobe.auditory_cortex import AuditoryCortex
from cns.data_types import (
    D_V52, D_AUDIO,
    CN_WIDTH, CN_START, CN_END,
    SOC_WIDTH, SOC_START, SOC_END,
    IC_WIDTH, IC_START, IC_END,
    AC_WIDTH, AC_START, AC_END,
)


def _place(arr, start, values):
    """Place values into array at start index."""
    arr = np.asarray(arr, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32).ravel()
    end = min(start + len(values), len(arr))
    arr[start:end] = values[:end - start]
    return arr


class AuditoryHierarchy:
    """v5.2 听觉层级管线 — 编排全流程.

    用法:
      ah = AuditoryHierarchy()
      result = ah.process(spectrum, arousal=0.8, fpn=fpn)
      # result['sensory']   → 感知向量 (D_V52=468,)
      # result['F_accuracy'] → 汇入扣带回
      # result['PE_total']   → 总预测误差
      # result['what_stream'] → Wernicke区 (语义理解增强)
    """

    def __init__(self):
        # ---- 所有听觉脑区模块 ----
        self.cn_left = CochlearNucleus()
        self.cn_right = CochlearNucleus()
        self.soc = SuperiorOlive()
        self.ll_left = LateralLemniscus()
        self.ll_right = LateralLemniscus()
        self.ic = InferiorColliculus()
        self.mgb = MedialGeniculateBody()
        self.ac = AuditoryCortex()

    def process(self,
                spectrum: Optional[np.ndarray] = None,
                left_spectrum: Optional[np.ndarray] = None,
                right_spectrum: Optional[np.ndarray] = None,
                semantic_vec: Optional[np.ndarray] = None,
                azimuth_hint: Optional[float] = None,
                arousal: float = 0.8,
                fpn: Optional[object] = None,
                visual_spatial: Optional[np.ndarray] = None,
                learn: bool = False) -> dict:
        """单步听觉处理 (前馈 + 反馈 + PE + 双流分离).

        Args:
            spectrum: 单耳频谱 (无空间信息时使用)
            left_spectrum: 左耳频谱
            right_spectrum: 右耳频谱
            semantic_vec: 语义向量 (无音频时的代理模式)
            azimuth_hint: 方位角提示 (度, -90=左, +90=右)
            arousal: 脑干唤醒度 [0, 1]
            fpn: FPN模块 (用于注意力调制)
            visual_spatial: 视觉空间信息 (用于ICx多感官整合)
            learn: 是否更新Hebb权重

        Returns:
            dict with:
              'sensory': 听觉感知向量段 (D_AUDIO,)
              'F_accuracy': 汇入扣带回的预测误差
              'PE_total': 总预测误差
              'what_stream': "What"流 → Wernicke
              'where_stream': "Where"流 → 顶叶/FPN
              'novelty': IC新颖性 → ACC
              'azimuth': SOC方位角估计
              'diagnostics': 调试信息
        """
        # ---- 输入处理: 真实频谱 或 语义代理 ----
        if spectrum is None and left_spectrum is None and right_spectrum is None:
            # 语义代理模式: 从语义向量合成伪频谱
            if semantic_vec is not None:
                spectrum = CochlearNucleus.semantic_to_pseudo_spectrum(
                    semantic_vec)
            else:
                spectrum = np.zeros(32, dtype=np.float32)

        # 确定左右耳输入
        if left_spectrum is not None and right_spectrum is not None:
            left_spec = left_spectrum
            right_spec = right_spectrum
        elif spectrum is not None:
            # 单耳模式: 根据azimuth创建双耳差异
            left_spec = spectrum.copy()
            right_spec = spectrum.copy()
            if azimuth_hint is not None:
                # 模拟ILD: 近侧耳声音更大
                az = float(azimuth_hint)
                az_rad = az * np.pi / 180.0
                ild_factor = float(np.sin(az_rad))  # [-1, +1]
                # 左侧声源 → 左耳更强
                left_gain = 1.0 + 0.3 * max(0.0, -ild_factor)
                right_gain = 1.0 + 0.3 * max(0.0, ild_factor)
                left_spec = left_spec * left_gain
                right_spec = right_spec * right_gain
        else:
            left_spec = np.zeros(32, dtype=np.float32)
            right_spec = np.zeros(32, dtype=np.float32)

        # ==== Phase 1: 自下而上 (前馈) ====

        # 耳蜗核: 频谱分解 → tonotopic
        cn_left_out = self.cn_left.process(left_spec)
        cn_right_out = self.cn_right.process(right_spec)

        # SOC: 双耳空间信息 (ITD + ILD)
        soc_out = self.soc.process(cn_left_out, cn_right_out,
                                   azimuth_hint=azimuth_hint)

        # LL: 时间增强 + 双耳抑制
        ll_left_out = self.ll_left.process(cn_left_out, soc_out,
                                           contralateral_cn=cn_right_out)
        ll_right_out = self.ll_right.process(cn_right_out, soc_out,
                                             contralateral_cn=cn_left_out)

        # IC: 频率×空间×时间 三维整合
        ic_out = self.ic.process(cn_left_out, soc_out,
                                 ll_output=ll_left_out,
                                 visual_spatial=visual_spatial)

        # MGB: 丘脑中继 + 门控
        mgb_out = self.mgb.process(ic_out, arousal=arousal, fpn=fpn)

        # AuditoryCortex: A1→Belt→Parabelt
        ac_out = self.ac.process(mgb_out, soc_spatial=soc_out['spatial'],
                                 ic_output=ic_out, arousal=arousal)

        # ==== Phase 2: 自上而下 (反馈预测) ====

        # AC Parabelt ← 在AC内部已完成 Belt/A1 反馈
        # AC → MGB 反馈
        mgb_pe = self.mgb.compute_prediction_error(mgb_out['relay'])
        # MGB → IC 反馈
        ic_pe = self.ic.compute_prediction_error()
        self.ic.receive_feedback(mgb_pe, lr=0.05)
        # IC → CN 反馈 (CN有预测编码接口)
        cn_pe = cn_left_out['adapted'] - self.cn_left.get_prediction()[:len(cn_left_out['adapted'])]
        self.cn_left.receive_feedback(cn_pe, lr=0.03)

        # ==== Phase 3: 预测误差汇总 ====

        pe_cn = np.abs(self.cn_left.compute_prediction_error()['tonotopic'])
        pe_ic = np.abs(ic_pe)
        pe_mgb = np.abs(mgb_pe)

        # 总 F_accuracy: 听觉层级各层PE加权
        F_accuracy_auditory = (
            float(np.mean(pe_cn)) * 0.2 +
            float(np.mean(pe_ic)) * 0.4 +
            float(np.mean(pe_mgb)) * 0.5 +
            ac_out['F_accuracy'] * 0.8
        )

        PE_total = (
            float(np.mean(pe_cn)) +
            float(np.mean(pe_ic)) +
            float(np.mean(pe_mgb)) +
            ac_out['PE_total']
        )

        # ==== Phase 4: 构建听觉感知向量 (D_AUDIO=96) ====

        sensory_audio = np.zeros(D_AUDIO, dtype=np.float32)

        # 耳蜗核段: tonotopic
        _place(sensory_audio, 0,
               cn_left_out['tonotopic'][:CN_WIDTH])

        # SOC段: 双耳空间 (ITD + ILD)
        _place(sensory_audio, CN_WIDTH,
               soc_out['spatial'][:SOC_WIDTH])

        # IC段: 整合特征
        _place(sensory_audio, CN_WIDTH + SOC_WIDTH,
               ic_out['integrated'][:IC_WIDTH])

        # AC段: 听觉对象/场景
        ac_features = np.concatenate([
            ac_out['what_stream'][:8],
            ac_out['where_stream'][:8],
        ])
        _place(sensory_audio, CN_WIDTH + SOC_WIDTH + IC_WIDTH,
               ac_features[:AC_WIDTH])

        return {
            'sensory': sensory_audio.astype(np.float32),
            'F_accuracy': F_accuracy_auditory,
            'PE_total': PE_total,
            'what_stream': ac_out['what_stream'],
            'where_stream': ac_out['where_stream'],
            'novelty': ic_out['novelty'],
            'azimuth': soc_out['azimuth'],
            'cn_output': cn_left_out,
            'soc_output': soc_out,
            'ic_output': ic_out,
            'mgb_output': mgb_out,
            'ac_output': ac_out,
            'diagnostics': {
                'cn_pe': float(np.mean(pe_cn)),
                'ic_pe': float(np.mean(pe_ic)),
                'mgb_pe': float(np.mean(pe_mgb)),
                'ac_pe': ac_out['F_accuracy'],
                'n_asa_streams': ac_out['scene']['n_active_streams'],
            },
        }
