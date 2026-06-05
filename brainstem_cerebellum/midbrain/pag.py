"""
pag.py — 中脑导水管周围灰质 (Periaqueductal Gray) [v5.4]

对应脑区: 中脑导水管周围灰质 (PAG)
所属层级: 脑干 → 中脑 (Level 3)
脑区标签: PAG (vlPAG · lPAG · dlPAG · dmPAG)

功能职责:
  - 下行痛觉调节的高级整合中枢
  - 内源性镇痛系统的核心中继站
  - 接收前脑输入 (PFC, ACC, 岛叶, 杏仁核, 下丘脑)
  - 激活RVM → 下行抑制/易化脊髓背角
  - 富含内源性阿片肽 (脑啡肽, β-内啡肽)

四大柱状亚区:
  - dlPAG (背外侧): 主动应对策略 (fight/flight + 非阿片镇痛)
  - dmPAG (背内侧): 被动应对策略 (freezing + 心血管调节)
  - lPAG (外侧): 防御行为 + 心血管反应
  - vlPAG (腹外侧): 被动应对 + 内源性阿片镇痛 (关键——μ受体密集)

去抑制模型 (经典):
  1. PAG传出神经元受GABA能中间神经元持续抑制 ("被关住")
  2. 内源性阿片肽 (或外源性吗啡) 激活μ受体 → 抑制GABA中间神经元
  3. GABA抑制解除 (去抑制) → PAG传出神经元兴奋
  4. → 激活RVM OFF细胞 → 下行抑制脊髓背角

知觉规律:
  1. 下行调控律 — 前脑→PAG→RVM→脊髓, 三层调控
  2. 应激诱导镇痛 (SIA) — 急性应激 → 内源性阿片释放
  3. 安慰剂效应 — 正面预期 → PFC→PAG→内啡肽释放

在 NotMe 中的应用:
  - 接收ACC/岛叶/前额叶的认知-情感输入
  - 计算下行镇痛/易化信号
  - 内源性阿片释放模型
  - 输出到 RVM → 脊髓背角
"""

import numpy as np
from typing import Optional

# ============================================================
# 常量
# ============================================================

N_PAG_OUTPUT = 8              # PAG输出维度 (下行信号)
N_PAG_COLUMNS = 4             # 四个功能柱 (dl/l/dm/vl)


class PAGColumn:
    """PAG功能柱 — 不同应对策略.

    四个柱体处理不同的痛觉调节策略.
    vlPAG: 内源性阿片镇痛 (μ受体) — 本模块重点
    dlPAG: 非阿片镇痛 (NA/5-HT) — 主动应对
    """

    def __init__(self, column_type: str = 'vl'):
        self.col_type = column_type
        # 各柱体特性
        if column_type == 'vl':
            self._opioid_sensitivity = 0.9   # 高μ受体密度
            self._active_coping = 0.2        # 低主动应对
        elif column_type == 'dl':
            self._opioid_sensitivity = 0.3
            self._active_coping = 0.9        # 高主动应对
        elif column_type == 'dm':
            self._opioid_sensitivity = 0.4
            self._active_coping = 0.3
        else:  # lPAG
            self._opioid_sensitivity = 0.5
            self._active_coping = 0.7

        self.activation: float = 0.1

    def process(self, acc_input: float, insula_input: float,
                amygdala_input: float, pfc_input: float,
                endogenous_opioid: float) -> float:
        """单柱体处理.

        Args:
            acc_input: ACC情感-动机输入
            insula_input: 岛叶内感受输入
            amygdala_input: 杏仁核恐惧/焦虑输入
            pfc_input: 前额叶认知调控输入
            endogenous_opioid: 当前内源性阿片水平

        Returns:
            柱体激活 [0, 1]
        """
        # 情感-动机驱动
        affective_drive = 0.35 * acc_input + 0.25 * insula_input
        # 恐惧驱动 (杏仁核 → 激活PAG → 应激镇痛)
        fear_drive = 0.25 * amygdala_input
        # 认知调控 (PFC → 激活PAG → 安慰剂效应)
        cognitive_drive = 0.15 * pfc_input

        # 阿片调制: vlPAG对阿片最敏感 (去抑制效应)
        opioid_gain = 1.0 + self._opioid_sensitivity * endogenous_opioid

        # 综合驱动
        total_drive = (affective_drive + fear_drive + cognitive_drive)
        total_drive *= opioid_gain

        # 应对风格调制
        if self._active_coping > 0.5:
            total_drive *= (1.0 + 0.5 * fear_drive)  # 恐惧增强主动应对

        self.activation = float(np.clip(
            0.8 * self.activation + 0.2 * total_drive, 0.0, 1.0))
        return self.activation


class PeriaqueductalGray:
    """中脑导水管周围灰质 — 下行痛觉调节总枢纽.

    组装四个功能柱, 接收前脑输入, 输出下行调控信号.

    用法:
      pag = PeriaqueductalGray()
      output = pag.process(
          acc_affect=0.7,       # ACC痛觉情感
          insula_intero=0.6,    # 岛叶内感受
          amygdala_fear=0.3,    # 杏仁核恐惧
          pfc_cognitive=0.5,    # 前额叶认知调控
          placebo_expectation=0.2,  # 安慰剂预期
      )
      # output['descending_signal'] → RVM
      # output['endorphin_release'] → 内源性阿片释放
    """

    def __init__(self):
        # 四个功能柱
        self.vl_column = PAGColumn('vl')     # 内源性阿片镇痛
        self.dl_column = PAGColumn('dl')     # 非阿片镇痛 (主动应对)
        self.dm_column = PAGColumn('dm')     # 被动应对
        self.l_column = PAGColumn('l')       # 防御行为

        # 内源性阿片系统状态
        self._endorphin_level: float = 0.1     # β-内啡肽水平 [0, 1]
        self._enkephalin_level: float = 0.1    # 脑啡肽水平 [0, 1]
        self._dynorphin_level: float = 0.05    # 强啡肽水平 [0, 1]

        # 下行信号输出
        self._descending_signal: float = 0.0   # [-1, 1] 正=镇痛, 负=易化
        self._descending_history: list[float] = []

        # 应激状态
        self._stress_level: float = 0.0
        self._stress_induced_analgesia: float = 0.0  # SIA 水平

        # 安慰剂效应追踪
        self._placebo_expectation: float = 0.0
        self._placebo_analgesia: float = 0.0

        # 纳洛酮可逆性标记 (用于区分阿片/非阿片镇痛)
        self._naloxone_reversible: float = 0.0

    def process(self,
                acc_affect: float = 0.0,
                insula_intero: float = 0.0,
                amygdala_fear: float = 0.0,
                pfc_cognitive: float = 0.0,
                stress_level: float = 0.0,
                placebo_expectation: float = 0.0,
                valence: float = 0.0,
                arousal: float = 0.5) -> dict:
        """PAG单步处理 — 整合前脑输入, 输出下行调控.

        Args:
            acc_affect: ACC痛觉情感-动机信号 [0, 1]
            insula_intero: 岛叶内感受信号 [0, 1]
            amygdala_fear: 杏仁核恐惧/焦虑 [0, 1]
            pfc_cognitive: 前额叶认知评估/注意力 [0, 1]
            stress_level: 应激水平 [0, 1]
            placebo_expectation: 安慰剂预期强度 [0, 1]
            valence: 当前效价 [-1, 1]
            arousal: 当前唤醒 [0, 1]

        Returns:
            dict with:
              'descending_signal': 下行调控信号 [-1, 1]
              'endorphin_release': β-内啡肽释放量 [0, 1]
              'enkephalin_release': 脑啡肽释放量 [0, 1]
              'total_opioid_tone': 总阿片基调 [0, 1]
              'sia_active': 应激诱导镇痛是否激活
              'placebo_active': 安慰剂镇痛是否激活
              'column_activations': 四个功能柱激活
        """
        # ---- 内源性阿片更新 ----
        # 1. β-内啡肽: 应激 + 正效价 + 安慰剂预期 → 释放
        endorphin_drive = (
            0.4 * stress_level +
            0.3 * max(0.0, valence) +     # 正效价促进释放
            0.3 * placebo_expectation
        )
        self._endorphin_level = float(np.clip(
            0.95 * self._endorphin_level + 0.05 * endorphin_drive, 0.0, 1.0))

        # 2. 脑啡肽: PAG局部释放, 响应伤害性输入
        enkephalin_drive = 0.6 * insula_intero + 0.4 * acc_affect
        self._enkephalin_level = float(np.clip(
            0.9 * self._enkephalin_level + 0.1 * enkephalin_drive, 0.0, 1.0))

        # 3. 强啡肽: 慢性应激 → 强啡肽↑ (κ受体 → 负性情感)
        if stress_level > 0.5:
            self._dynorphin_level = float(np.clip(
                self._dynorphin_level + 0.005 * (stress_level - 0.5), 0.0, 0.5))
        else:
            self._dynorphin_level *= 0.99

        # ---- 功能柱处理 ----
        total_opioid = 0.6 * self._endorphin_level + 0.4 * self._enkephalin_level

        vl_act = self.vl_column.process(
            acc_affect, insula_intero, amygdala_fear,
            pfc_cognitive, total_opioid)
        dl_act = self.dl_column.process(
            acc_affect, insula_intero, amygdala_fear,
            pfc_cognitive, total_opioid)
        dm_act = self.dm_column.process(
            acc_affect, insula_intero, amygdala_fear,
            pfc_cognitive, total_opioid)
        l_act = self.l_column.process(
            acc_affect, insula_intero, amygdala_fear,
            pfc_cognitive, total_opioid)

        # ---- 应激诱导镇痛 (SIA) ----
        # 急性应激 → 内源性阿片释放 → SIA
        # 慢性应激 → 痛觉敏化 (非SIA)
        self._stress_level = 0.85 * self._stress_level + 0.15 * stress_level
        if stress_level > 0.6 and self._stress_level < 0.8:
            # 急性应激: SIA激活
            self._stress_induced_analgesia = float(np.clip(
                self._stress_induced_analgesia + 0.1, 0.0, 0.8))
            self._naloxone_reversible = 0.7  # 大部分纳洛酮可逆 (阿片介导)
        elif self._stress_level > 0.8:
            # 慢性应激: SIA衰减 + 转为易化 (强啡肽↑)
            self._stress_induced_analgesia *= 0.95
            self._naloxone_reversible *= 0.8
        else:
            self._stress_induced_analgesia *= 0.9
            self._naloxone_reversible *= 0.95

        # ---- 安慰剂镇痛 ----
        self._placebo_expectation = (0.8 * self._placebo_expectation
                                     + 0.2 * placebo_expectation)
        # 安慰剂效应: PFC→PAG→释放内啡肽 (纳洛酮可逆)
        self._placebo_analgesia = float(np.clip(
            self._placebo_expectation * pfc_cognitive * 0.8, 0.0, 1.0))

        # ---- 综合下行信号 ----
        # 抑制性 (镇痛) 成分:
        analgesic = (
            0.40 * vl_act +       # vlPAG: 主要阿片镇痛
            0.15 * dl_act +       # dlPAG: 非阿片镇痛
            0.10 * dm_act +
            0.10 * self._stress_induced_analgesia +
            0.15 * self._placebo_analgesia +
            0.10 * max(0.0, pfc_cognitive - 0.5)  # PFC主动抑制
        )

        # 易化性 (加重疼痛) 成分:
        pro_nociceptive = (
            0.35 * amygdala_fear * (1.0 - self._stress_induced_analgesia) +
            0.25 * self._dynorphin_level * 2.0 +   # 强啡肽→κ受体→负性情感
            0.20 * acc_affect * (1.0 - vl_act) +   # ACC情感未被抑制 → 易化
            0.20 * insula_intero * (1.0 - vl_act)
        )

        # 下行信号: [-1, 1], 正=镇痛, 负=易化
        self._descending_signal = float(np.clip(
            analgesic - pro_nociceptive, -1.0, 1.0))

        # 唤醒调制: 低唤醒 → 镇痛减弱
        arousal_mod = np.clip(arousal * 2.0, 0.5, 1.5)
        self._descending_signal *= arousal_mod

        self._descending_history.append(self._descending_signal)
        if len(self._descending_history) > 100:
            self._descending_history = self._descending_history[-100:]

        return {
            'descending_signal': self._descending_signal,
            'endorphin_release': self._endorphin_level,
            'enkephalin_release': self._enkephalin_level,
            'dynorphin_level': self._dynorphin_level,
            'total_opioid_tone': total_opioid,
            'sia_active': self._stress_induced_analgesia > 0.3,
            'sia_level': self._stress_induced_analgesia,
            'placebo_active': self._placebo_analgesia > 0.2,
            'placebo_analgesia': self._placebo_analgesia,
            'naloxone_reversible': self._naloxone_reversible,
            'column_activations': {
                'vlPAG': vl_act,
                'dlPAG': dl_act,
                'dmPAG': dm_act,
                'lPAG': l_act,
            },
            'analgesic_component': analgesic,
            'pro_nociceptive_component': pro_nociceptive,
        }

    def reset(self):
        """重置PAG状态."""
        self._endorphin_level = 0.1
        self._enkephalin_level = 0.1
        self._dynorphin_level = 0.05
        self._descending_signal = 0.0
        self._stress_level = 0.0
        self._stress_induced_analgesia = 0.0
        self._placebo_expectation = 0.0
        self._placebo_analgesia = 0.0
