"""
nociception_hierarchy.py — 痛觉层级管线编排 (Nociception Hierarchy) [v5.4]

组装完整痛觉管线:
  DorsalHorn (闸门控制) → Spinothalamic (双通路) →
  Thalamus (痛觉中继) → S1/S2 (感觉辨别) + Insula (内感受) + ACC (情感动机)
  ← PAG → RVM → DorsalHorn (下行调控闭环)

v5.4 数据流:
  Phase 1: 前馈 (自下而上) — 伤害性输入 → 背角闸门 → 上行双通路 → 丘脑 → 皮层
  Phase 2: 皮层处理 — S1/S2感觉辨别 + 岛叶内感受 + ACC情感评估
  Phase 3: 下行调控 — ACC/岛叶/PFC → PAG → RVM → 脊髓背角 (镇痛/易化)
  Phase 4: 预测误差 — 各层PE汇总 → F_accuracy
  Phase 5: 构建感知向量 (D_V54=516)

7条知觉规律:
  1. 闸门控制律 — Aβ关闭闸门, Aδ/C打开闸门
  2. 痛觉非适应律 — 痛觉极少适应 (生存需要)
  3. 双重通路律 — 外侧(感觉辨别) + 内侧(情感动机)
  4. 下行调控律 — PAG→RVM→脊髓, 内啡肽/5-HT/NE释放
  5. 中枢致敏律 — wind-up → allodynia/hyperalgesia
  6. 痛觉情感律 — 岛叶+ACC将强度转化为"难受"
  7. 韦伯定律(变式) — 疼痛JND与基线成比例
"""

import numpy as np
from typing import Optional
from spinal.dorsal_horn import DorsalHorn
from brainstem_cerebellum.midbrain.pag import PeriaqueductalGray
from brainstem_cerebellum.medulla.rvm import RostralVentromedialMedulla
from cerebrum.limbic_system.insula import Insula
from cns.data_types import (
    D_V54, D_PAIN,
    PAIN_DH_START, PAIN_DH_END,
    PAIN_LATERAL_START, PAIN_LATERAL_END,
    PAIN_MEDIAL_START, PAIN_MEDIAL_END,
    PAIN_THALAMIC_START, PAIN_THALAMIC_END,
    PAIN_DH_WIDTH, PAIN_LATERAL_WIDTH,
    PAIN_MEDIAL_WIDTH, PAIN_THALAMIC_WIDTH,
)


def _place(arr, start, values):
    """Place values into array at start index."""
    arr = np.asarray(arr, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32).ravel()
    end = min(start + len(values), len(arr))
    arr[start:end] = values[:end - start]
    return arr


class SpinothalamicTract:
    """脊髓丘脑束 — 痛觉上行双通路模型.

    外侧支 (新脊髓丘脑束): 精确定位 + 感觉辨别 → VPL → S1/S2
    内侧支 (旧脊髓丘脑束): 弥散情感 + 动机 → CM-Pf/MD → ACC/岛叶
    """

    def __init__(self, lateral_dim: int = PAIN_LATERAL_WIDTH,
                 medial_dim: int = PAIN_MEDIAL_WIDTH):
        self.lateral_dim = lateral_dim
        self.medial_dim = medial_dim

        # 外侧通路状态 (感觉-辨别)
        self._lateral_signal: np.ndarray = np.zeros(lateral_dim, dtype=np.float32)
        # 内侧通路状态 (情感-动机)
        self._medial_signal: np.ndarray = np.zeros(medial_dim, dtype=np.float32)

    def process(self, dorsal_horn_output: dict) -> dict:
        """双通路分离.

        Args:
            dorsal_horn_output: DorsalHorn.process() 的输出

        Returns:
            dict with lateral (感觉), medial (情感)
        """
        pain_signal = dorsal_horn_output['pain_signal']
        fast_pain = dorsal_horn_output['fast_pain']
        slow_pain = dorsal_horn_output['slow_pain']

        # ---- 外侧通路: Aδ快痛 → 精确定位 ----
        # 编码疼痛位置、强度、时间特征
        lateral_raw = np.zeros(self.lateral_dim, dtype=np.float32)
        # 前4维: 快痛强度 (Aδ → VPL → S1)
        lateral_raw[0] = float(fast_pain)
        # 中4维: 疼痛信号的精细化表征
        pain_sig = np.asarray(pain_signal, dtype=np.float32).ravel()
        n_copy = min(len(pain_sig), 8)
        lateral_raw[4:4 + n_copy] = pain_sig[:n_copy]
        # 后4维: 感觉辨别特征

        self._lateral_signal = (0.6 * self._lateral_signal
                                + 0.4 * lateral_raw)

        # ---- 内侧通路: C慢痛 → 情感-动机 ----
        medial_raw = np.zeros(self.medial_dim, dtype=np.float32)
        # 前4维: 慢痛强度 (C → CM-Pf/MD → ACC/岛叶)
        medial_raw[0] = float(slow_pain)
        # 中4维: 弥散情感成分
        pain_affective = np.asarray(pain_signal, dtype=np.float32).ravel()
        n_copy = min(len(pain_affective), 8)
        medial_raw[4:4 + n_copy] = pain_affective[:n_copy]

        self._medial_signal = (0.6 * self._medial_signal
                               + 0.4 * medial_raw)

        return {
            'lateral': self._lateral_signal.copy(),
            'medial': self._medial_signal.copy(),
            'fast_pain': float(fast_pain),
            'slow_pain': float(slow_pain),
        }


class NociceptionHierarchy:
    """v5.4 痛觉层级管线 — 编排全流程.

    用法:
      nh = NociceptionHierarchy()
      result = nh.process(
          nociceptive_input=0.7,
          tissue_damage=0.3,
          abeta_input=0.1,
          valence=-0.2,
          arousal=0.6,
          body_vector=body.b,
          acc_affect=0.5,
          pfc_cognitive=0.4,
          placebo_expectation=0.1,
      )
      # result['sensory'] → 痛觉感知向量段 (D_PAIN,)
      # result['pain_intensity'] → 综合疼痛强度
      # result['unpleasantness'] → 疼痛不愉快度
    """

    def __init__(self):
        # ---- 痛觉脑区模块 ----
        self.dorsal_horn = DorsalHorn()
        self.spinothalamic = SpinothalamicTract()
        self.pag = PeriaqueductalGray()
        self.rvm = RostralVentromedialMedulla()
        self.insula = Insula()

        # 下行调控状态
        self._descending_history: list[float] = []
        # 慢性疼痛追踪
        self._chronic_pain_ema: float = 0.0
        self._chronic_pain_steps: int = 0

        # 预测编码
        self._prediction: Optional[np.ndarray] = None
        self._pe_ema: float = 0.0

    def process(self,
                nociceptive_input: float = 0.0,
                tissue_damage: float = 0.0,
                abeta_input: float = 0.0,
                valence: float = 0.0,
                arousal: float = 0.5,
                body_vector: Optional[np.ndarray] = None,
                acc_affect: float = 0.0,
                insula_intero: float = 0.0,
                amygdala_fear: float = 0.0,
                pfc_cognitive: float = 0.0,
                stress_level: float = 0.0,
                placebo_expectation: float = 0.0,
                fpn: Optional[object] = None,
                learn: bool = False) -> dict:
        """单步痛觉处理 (前馈 + 下行调控闭环 + 皮层评估).

        Args:
            nociceptive_input: 伤害性信号 [0, 1] (模拟组织损伤→C纤维)
            tissue_damage: 组织损伤程度 [0, 1]
            abeta_input: Aβ触觉输入 [0, 1] (按摩/触摸 → 关闭闸门)
            valence: 当前效价 [-1, 1]
            arousal: 当前唤醒 [0, 1]
            body_vector: 身体状态向量 (来自BodyVector)
            acc_affect: ACC情感-动机信号 [0, 1]
            insula_intero: 岛叶内感受信号 (外部传入, 可选)
            amygdala_fear: 杏仁核恐惧/焦虑 [0, 1]
            pfc_cognitive: 前额叶认知调控 [0, 1]
            stress_level: 应激水平 [0, 1]
            placebo_expectation: 安慰剂预期 [0, 1]
            fpn: FPN模块 (注意力调制, 可选)
            learn: 是否启用学习

        Returns:
            dict with:
              'sensory': 痛觉感知向量段 (D_PAIN,)
              'pain_intensity': 综合疼痛强度
              'unpleasantness': 疼痛不愉快度
              'gate_state': 闸门状态
              'sensitization': 中枢敏化水平
              'descending_signal': 下行调控信号
              'F_accuracy': 预测误差
              'PE_total': 总预测误差
              'allodynia': 触诱发痛标志
              'hyperalgesia': 痛觉过敏标志
              'diagnostics': 调试信息
        """
        # ---- 慢性疼痛追踪 ----
        if nociceptive_input > 0.5:
            self._chronic_pain_steps += 1
        else:
            self._chronic_pain_steps = max(0, self._chronic_pain_steps - 1)
        self._chronic_pain_ema = float(np.clip(
            0.995 * self._chronic_pain_ema
            + 0.005 * (1.0 if self._chronic_pain_steps > 50 else 0.0),
            0.0, 1.0))

        # ---- Phase 1: 下行调控 (上一步的PAG→RVM→背角) ----
        # 使用上一步累积的下行信号
        prev_descending = (self._descending_history[-1]
                           if self._descending_history else 0.0)
        prev_opioid = self.pag._endorphin_level

        # ---- Phase 2: 脊髓背角闸门 ----
        dh_output = self.dorsal_horn.process(
            nociceptive_input=nociceptive_input,
            abeta_input=abeta_input,
            tissue_damage=tissue_damage,
            descending_signal=prev_descending,
            endorphin_level=prev_opioid,
        )

        # ---- Phase 3: 上行双通路 ----
        stt_output = self.spinothalamic.process(dh_output)

        # ---- Phase 4: 皮层处理 ----
        # 岛叶: 内感受 + 情感评估
        pain_intensity = float(np.mean(dh_output['pain_signal']))
        insula_output = self.insula.process(
            pain_signal=dh_output['pain_signal'],
            body_vector=body_vector,
            thalamic_relay=stt_output['lateral'],
            pain_intensity=pain_intensity,
            valence=valence,
            arousal=arousal,
        )

        # 岛叶内感受突显 (用于PAG)
        insula_salience = insula_output['interoceptive_salience']
        unpleasantness = insula_output['unpleasantness']

        # ---- Phase 5: PAG下行调控枢纽 ----
        # 输入: ACC情感 + 岛叶内感受 + 杏仁核恐惧 + PFC认知
        pag_output = self.pag.process(
            acc_affect=acc_affect,
            insula_intero=max(insula_intero, insula_salience),
            amygdala_fear=amygdala_fear,
            pfc_cognitive=pfc_cognitive,
            stress_level=stress_level,
            placebo_expectation=placebo_expectation,
            valence=valence,
            arousal=arousal,
        )

        # ---- Phase 6: RVM中继 ----
        rvm_output = self.rvm.process(
            pag_signal=pag_output['descending_signal'],
            opioid_level=pag_output['total_opioid_tone'],
            chronic_pain=self._chronic_pain_ema,
            pain_intensity=pain_intensity,
            arousal=arousal,
        )

        # 下行调控信号记录
        self._descending_history.append(rvm_output['descending_signal'])
        if len(self._descending_history) > 100:
            self._descending_history = self._descending_history[-100:]

        # ---- Phase 7: 预测编码 ----
        # 构建痛觉感知段
        pain_sensory = np.concatenate([
            dh_output['pain_signal'][:PAIN_DH_WIDTH],
            stt_output['lateral'][:PAIN_LATERAL_WIDTH],
            stt_output['medial'][:PAIN_MEDIAL_WIDTH],
            # 丘脑痛觉中继 (简化: lateral + medial 的混合)
            np.concatenate([
                stt_output['lateral'][:4],
                stt_output['medial'][:4],
            ])[:PAIN_THALAMIC_WIDTH],
        ]).astype(np.float32)

        # 填充到 D_PAIN
        pain_segment = np.zeros(D_PAIN, dtype=np.float32)
        n = min(len(pain_sensory), D_PAIN)
        pain_segment[:n] = pain_sensory[:n]

        # 预测误差
        if self._prediction is None:
            self._prediction = pain_segment.copy()
            F_accuracy = 0.5 * float(np.sum(pain_segment ** 2))
        else:
            pe = pain_segment - self._prediction
            F_accuracy = float(np.sum(pe ** 2))
            self._prediction = 0.9 * self._prediction + 0.1 * pain_segment

        self._pe_ema = 0.9 * self._pe_ema + 0.1 * F_accuracy

        # ---- FPN注意力调制 (可选) ----
        if fpn is not None and hasattr(fpn, 'attention_template'):
            # 疼痛突显 → 吸引注意力
            pain_salience = max(pain_intensity, unpleasantness)
            # 增强痛觉通道的注意力增益
            pass  # FPN调制在agent.step()中统一处理

        return {
            'sensory': pain_segment,
            'pain_intensity': pain_intensity,
            'unpleasantness': unpleasantness,
            'gate_state': dh_output['gate_state'],
            'sensitization': dh_output['sensitization'],
            'allodynia': dh_output['allodynia'],
            'hyperalgesia': dh_output['hyperalgesia'],
            'fast_pain': dh_output['fast_pain'],
            'slow_pain': dh_output['slow_pain'],
            'descending_signal': rvm_output['descending_signal'],
            'off_on_ratio': rvm_output['off_on_ratio'],
            'endorphin_level': pag_output['endorphin_release'],
            'sia_active': pag_output['sia_active'],
            'placebo_active': pag_output['placebo_active'],
            'insula_salience': insula_salience,
            'interoceptive_pe': insula_output['body_prediction_error'],
            'F_accuracy': F_accuracy,
            'PE_total': self._pe_ema,
            'chronic_pain': self._chronic_pain_ema,
            'diagnostics': {
                'dh_output': dh_output,
                'stt_output': stt_output,
                'pag_output': pag_output,
                'rvm_output': rvm_output,
                'insula_output': insula_output,
            },
        }

    def reset(self):
        """重置痛觉管线."""
        self.dorsal_horn = DorsalHorn()
        self.spinothalamic = SpinothalamicTract()
        self.pag.reset()
        self.rvm.reset()
        self.insula = Insula()
        self._prediction = None
        self._pe_ema = 0.0
        self._chronic_pain_ema = 0.0
        self._chronic_pain_steps = 0
        self._descending_history = []

    # ================================================================
    # 预测编码接口
    # ================================================================

    def get_prediction(self) -> np.ndarray:
        """返回痛觉预测."""
        if self._prediction is None:
            return np.zeros(D_PAIN, dtype=np.float32)
        return self._prediction.copy()

    def compute_prediction_error(self) -> float:
        """返回痛觉预测误差."""
        return self._pe_ema
