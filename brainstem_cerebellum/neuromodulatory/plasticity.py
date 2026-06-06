"""
plasticity.py — 可塑性调节 (Plasticity Regulation)  [v6.1 已实现]

对应脑区: 神经调节系统 (多巴胺/NE/5-HT/ACh 的 plasticity 效应)
所属层级: 脑干+小脑 → 神经调节 → 可塑性

功能职责:
  - 关键期 (Critical Period): GluN2B→GluN2A 连续发育轨迹
  - 事件驱动可塑性 (Event-Driven Plasticity): RPE/新颖性→临时可塑性窗口
  - 稳态可塑性 (Homeostatic Plasticity): 维持网络活性在目标范围
  - 异突触可塑性 (Heterosynaptic Plasticity): 神经调质门控 LTP/LTD
  - 可塑性衰减 (Plasticity Decay): 随年龄/经验降低

神经调质对可塑性的影响:
  多巴胺 (DA):   D1 → LTP 增强; D2 → LTD 增强
  NE:             β-receptor → LTP 增强
  ACh:            mAChR → LTP 门控 (注意→学习)
  5-HT:           5-HT2A → LTP 调制

在 NotMe 中的实现 (v6.1):
  1. PlasticityRegulator: 整合 GluN2B + 事件驱动 + 稳态 + 神经调质
  2. 为 agent.step() 提供统一的 plasticity_factor 输出

设计参考:
  - Hensch, T. K. (2005). Critical period plasticity in local cortical circuits.
  - Turrigiano, G. G., & Nelson, S. B. (2004). Homeostatic plasticity.
  - Bear, M. F., Connors, B. W., & Paradiso, M. A. (2016). Neuroscience.
"""

import numpy as np
from typing import Optional


# ============================================================
# 常量
# ============================================================

# 稳态目标
HOMEOSTATIC_TARGET_ACTIVATION = 0.15   # 目标平均激活值
HOMEOSTATIC_SCALE_TAU = 0.99           # 稳态缩放 EMA 时间常数
HOMEOSTATIC_SCALE_RANGE = (0.5, 2.0)   # 缩放因子范围

# 事件驱动可塑性
EVENT_LR_BOOST_MAX = 3.0               # 最大事件学习率增强
EVENT_LR_BOOST_DECAY = 0.5             # 事件增强衰减率
EVENT_RPE_THRESHOLD = 0.3              # RPE 阈值 (超过才触发)

# 神经调质 LTP/LTD 门控
DA_D1_LTP_BOOST = 1.5                  # D1 激活 → LTP 增强
DA_D2_LTD_BOOST = 1.3                  # D2 激活 → LTD 增强
NE_BETA_LTP_BOOST = 1.3                # NE β-receptor → LTP 增强
ACH_LTP_GATE = 1.4                     # ACh → 注意力门控 LTP


# ============================================================
# 可塑性调节器
# ============================================================

class PlasticityRegulator:
    """v6.1: 统一可塑性调节器.

    整合四个维度:
      1. 发育可塑性 (GluN2B 轨迹)
      2. 事件驱动可塑性 (RPE → 临时学习率变化)
      3. 稳态可塑性 (firing rate homeostasis)
      4. 神经调质门控 (DA/NE/ACh → LTP/LTD 偏向)

    用法:
      pr = PlasticityRegulator()
      result = pr.process(
          glun2b_ratio=0.6, rpe=0.2, novelty=0.3,
          da_tonic=0.3, da_phasic=0.1, ne_tonic=0.2,
          mean_activation=0.15,
      )
      # result['plasticity_factor'] → 应用到学习率
      # result['ltp_bias'] → LTP vs LTD 偏向
    """

    def __init__(self):
        # 事件驱动可塑性状态
        self._event_boost: float = 1.0
        self._event_boost_decay = EVENT_LR_BOOST_DECAY

    def process(self,
                glun2b_ratio: float = 0.5,
                rpe: float = 0.0,
                novelty: float = 0.0,
                da_tonic: float = 0.3,
                da_phasic: float = 0.0,
                ne_tonic: float = 0.2,
                mean_activation: float = 0.15,
                ) -> dict:
        """单步可塑性调节.

        Args:
            glun2b_ratio: 当前 GluN2B 占比 [0.05, 1.0]
            rpe: 奖赏预测误差 [-1, 1]
            novelty: 新颖性信号 [0, 1]
            da_tonic: 紧张性 DA [0, 1]
            da_phasic: 时相性 DA [0, 1]
            ne_tonic: 紧张性 NE [0, 1]
            mean_activation: 网络平均激活值

        Returns:
            dict with:
              plasticity_factor: 学习率总调制因子
              developmental_factor: GluN2B 发育因子
              event_factor: 事件驱动因子
              homeostatic_factor: 稳态缩放因子
              ltp_bias: LTP vs LTD 偏向 (>1 = LTP 偏向, <1 = LTD 偏向)
              is_event: 当前是否为事件触发状态
        """
        # ---- 1. 发育因子 (GluN2B) ----
        # GluN2B 高 → 高可塑性 (婴儿期)
        developmental_factor = 0.5 + 0.5 * glun2b_ratio  # [0.525, 1.0]

        # ---- 2. 事件驱动可塑性 ----
        # RPE 高 + 新颖性 → 临时学习率增强
        is_event = abs(rpe) > EVENT_RPE_THRESHOLD
        if is_event:
            # 事件触发 → 临时增强
            event_boost = 1.0 + (EVENT_LR_BOOST_MAX - 1.0) * abs(rpe)
            # 新颖性增强事件效应
            event_boost *= (1.0 + 0.3 * novelty)
            self._event_boost = float(np.clip(
                max(self._event_boost, event_boost),
                1.0, EVENT_LR_BOOST_MAX * 1.5))
        else:
            # 事件消退 → 衰减
            self._event_boost = 1.0 + (self._event_boost - 1.0) * self._event_boost_decay

        event_factor = float(np.clip(self._event_boost, 1.0, EVENT_LR_BOOST_MAX * 1.5))

        # ---- 3. 稳态可塑性 ----
        # 平均激活偏离目标 → 全局缩放
        if mean_activation > 0:
            ratio = mean_activation / max(HOMEOSTATIC_TARGET_ACTIVATION, 0.01)
            # 激活过高 → 降低全局学习率 (防止 runaway)
            # 激活过低 → 提高全局学习率 (促进新学习)
            homeostatic_factor = float(np.clip(
                1.0 / np.sqrt(ratio),
                HOMEOSTATIC_SCALE_RANGE[0],
                HOMEOSTATIC_SCALE_RANGE[1]))
        else:
            homeostatic_factor = 1.0

        # ---- 4. 神经调质 LTP/LTD 偏向 ----
        # D1 (Go) → LTP 增强
        # D2 (No-Go) → LTD 增强
        # NE β-receptor → LTP 增强
        # ACh (通过 attention/novelty 代理) → LTP 门控

        # DA 偏向: D1-D2 平衡
        # 简化: tonic DA = D1 tone, 低 DA = D2 占优
        da_d1_effect = da_tonic * DA_D1_LTP_BOOST  # D1 → LTP
        da_d2_effect = (1.0 - da_tonic) * DA_D2_LTD_BOOST  # D2 → LTD

        # NE 偏向: 高 NE → LTP 增强
        ne_effect = ne_tonic * NE_BETA_LTP_BOOST  # NE → LTP

        # ACh 门控: attention/novelty → LTP
        ach_effect = novelty * ACH_LTP_GATE  # novelty → ACh → LTP

        # 净 LTP 偏向
        ltp_bias = float(np.clip(
            da_d1_effect + ne_effect + ach_effect - da_d2_effect,
            0.3, 3.0))

        # ---- 综合 plasticity factor ----
        # 发育 × 事件 × 稳态 = 总调制
        plasticity_factor = float(np.clip(
            developmental_factor * event_factor * homeostatic_factor,
            0.3, 6.0))

        return {
            'plasticity_factor': plasticity_factor,
            'developmental_factor': developmental_factor,
            'event_factor': event_factor,
            'homeostatic_factor': homeostatic_factor,
            'ltp_bias': ltp_bias,
            'is_event': is_event,
            'rpe': rpe,
            'novelty': novelty,
        }

    def reset(self):
        self._event_boost = 1.0
