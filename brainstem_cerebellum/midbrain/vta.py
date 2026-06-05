"""
vta.py — 腹侧被盖区 (Ventral Tegmental Area) [v5.5]

对应脑区: VTA (腹侧被盖区, A10 细胞群)
所属层级: 脑干 → 中脑 → VTA

功能职责:
  - 多巴胺释放 → 伏隔核(NAc)、前额叶(PFC)、杏仁核
  - 奖赏预测误差 (RPE): δ = R_actual - R_predicted
  - Incentive salience (wanting): 将中性刺激转化为"想要"的目标
  - 动机驱动: 调节行为激活水平
  - 社会奖赏: 社交互动也触发 VTA 多巴胺

在 NotMe 中的实现 (v5.5):
  1. RPEModel: 奖赏预测误差 (δ = R - R_pred), R 来自 Δvalence + ΔF_body + 社会
  2. DopamineDynamics: 时相性(burst) + 紧张性(baseline) 多巴胺
  3. VTA: 顶层 — RPE → DA → 事件驱动学习率调制

设计参考:
  - Schultz, W., Dayan, P., & Montague, P. R. (1997). A neural substrate of
    prediction and reward.
  - Berridge, K. C., & Robinson, T. E. (2003). Parsing reward.
"""

import numpy as np
from typing import Optional

# ============================================================
# 常量
# ============================================================

# 多巴胺动力学
DA_TONIC_BASELINE = 0.30      # 紧张性DA基线
DA_PHASIC_DECAY = 0.50        # 时相性DA衰减率 (快速回落, ~200ms 尺度)
DA_PHASIC_MAX = 1.0           # 时相性DA上限
DA_TONIC_TAU = 0.95           # 紧张性DA EMA时间常数

# 学习率调制范围
LR_MULT_MIN = 0.3             # 负RPE → 最低学习率压制
LR_MULT_MAX = 3.0             # 正RPE → 最高学习率增强


# ============================================================
# 奖赏预测误差模型
# ============================================================

class RPEModel:
    """奖赏预测误差: δ = R_actual - R_predicted.

    R_actual 是复合奖赏信号:
      - Δvalence: 效价改善 = 正奖赏
      - ΔF_body:  身体自由能下降 = 正奖赏 (生存状态改善)
      - 社会奖赏: 正效价互动 → 正奖赏
      - 新颖性:   意外事件 → 探索性奖赏 (novelty bonus)

    R_predicted 是过去奖赏的 EMA.

    用法:
      rpe = RPEModel()
      delta, pred = rpe.compute_rpe(valence_delta=0.1, fbody_delta=-0.05,
                                     social_reward=0.3, novelty=0.1)
    """

    def __init__(self, tau: float = 0.9):
        self.tau = tau                         # 预测 EMA 时间常数
        self._predicted_reward: float = 0.0     # EMA 预测
        self._rpe_history: list[float] = []     # RPE 历史 (最近100步)
        self._reward_history: list[float] = []  # 实际奖赏历史

    def compute_rpe(self,
                    valence_delta: float = 0.0,
                    fbody_delta: float = 0.0,
                    social_reward: float = 0.0,
                    novelty: float = 0.0) -> tuple[float, float]:
        """计算奖赏预测误差.

        Args:
            valence_delta: Δvalence (当前 - 上一步), 正=情绪改善
            fbody_delta: ΔF_body (上一步 - 当前), 正=身体状态改善
            social_reward: 社会奖赏 [0, 1]
            novelty: 新颖性信号 [0, 1]

        Returns:
            (rpe: float [-1, 1], predicted_reward: float)
        """
        # 复合实际奖赏
        # Δvalence × 0.4: 情绪改善权重
        # ΔF_body × 0.3:  身体稳态改善权重
        # social × 0.2:    社会奖赏权重
        # novelty × 0.1:   探索奖赏权重
        actual_reward = float(np.clip(
            0.4 * valence_delta +
            0.3 * fbody_delta +
            0.2 * social_reward +
            0.1 * novelty,
            -1.0, 1.0))

        # 奖赏预测误差
        rpe = actual_reward - self._predicted_reward

        # 更新预测 (EMA)
        self._predicted_reward = (
            self.tau * self._predicted_reward
            + (1.0 - self.tau) * actual_reward
        )

        # 记录历史
        self._rpe_history.append(rpe)
        self._reward_history.append(actual_reward)
        if len(self._rpe_history) > 100:
            self._rpe_history = self._rpe_history[-100:]
            self._reward_history = self._reward_history[-100:]

        return float(rpe), float(self._predicted_reward)

    @property
    def predicted_reward(self) -> float:
        return self._predicted_reward

    def reset(self):
        self._predicted_reward = 0.0
        self._rpe_history = []
        self._reward_history = []


# ============================================================
# 多巴胺动力学
# ============================================================

class DopamineDynamics:
    """时相性(Phasic) vs 紧张性(Tonic) 多巴胺动力学.

    时相性 DA: 事件驱动爆发 — 正 RPE → 快速 DA 释放 → 快速衰减
    紧张性 DA: 基线水平 — 慢速调制, 受唤醒度和长期奖赏历史影响

    用法:
      da = DopamineDynamics()
      result = da.process(rpe=0.3, novelty=0.2, arousal=0.5)
    """

    def __init__(self):
        self._tonic_da: float = DA_TONIC_BASELINE
        self._phasic_da: float = 0.0

        # 累积奖赏追踪 (紧张性DA的慢速调节基础)
        self._cumulative_reward_ema: float = 0.0

    def process(self,
                rpe: float = 0.0,
                novelty: float = 0.0,
                arousal: float = 0.5) -> dict:
        """更新多巴胺动态.

        Args:
            rpe: 奖赏预测误差 [-1, 1]
            novelty: 新颖性信号 [0, 1]
            arousal: 唤醒度 [0, 1]

        Returns:
            dict with tonic_da, phasic_da, total_da
        """
        # ---- 时相性 DA: 正RPE触发爆发 ----
        # 正 RPE → phasic DA 爆发; 负 RPE → phasic DA 下降 (dip)
        if rpe > 0:
            # 正 RPE: DA 爆发 (burst firing)
            phasic_burst = rpe * 0.8  # scale to [0, 0.8]
            # 新颖性增强爆发 (unexpected reward → bigger burst)
            novelty_boost = 1.0 + 0.5 * novelty
            self._phasic_da = float(np.clip(
                self._phasic_da + phasic_burst * novelty_boost,
                0.0, DA_PHASIC_MAX))
        else:
            # 负 RPE: DA 暂停 (pause / dip)
            phasic_dip = abs(rpe) * 0.3
            self._phasic_da = float(max(0.0, self._phasic_da - phasic_dip))

        # 时相性DA快速衰减
        self._phasic_da *= DA_PHASIC_DECAY

        # ---- 紧张性 DA: 慢速基线调制 ----
        # 累积奖赏 EMA
        self._cumulative_reward_ema = float(np.clip(
            0.99 * self._cumulative_reward_ema + 0.01 * rpe,
            -1.0, 1.0))

        # 紧张性DA目标: 基线 + 累积奖赏偏移 + 唤醒度调制
        tonic_target = float(np.clip(
            DA_TONIC_BASELINE
            + 0.1 * self._cumulative_reward_ema   # 长期奖赏 ↑ → tonic ↑
            + 0.05 * (arousal - 0.5),              # 高唤醒 → tonic ↑
            0.05, 0.7))

        # 慢速EMA向目标
        self._tonic_da = float(np.clip(
            DA_TONIC_TAU * self._tonic_da
            + (1.0 - DA_TONIC_TAU) * tonic_target,
            0.05, 0.8))

        # ---- 总DA = tonic + phasic ----
        total_da = float(np.clip(self._tonic_da + self._phasic_da, 0.0, 1.0))

        return {
            'tonic_da': self._tonic_da,
            'phasic_da': self._phasic_da,
            'total_da': total_da,
        }

    def reset(self):
        self._tonic_da = DA_TONIC_BASELINE
        self._phasic_da = 0.0
        self._cumulative_reward_ema = 0.0


# ============================================================
# VTA — 顶层奖赏系统
# ============================================================

class VTA:
    """VTA 奖赏系统 — RPE → DA 释放 → 学习率调制 + 动机调节.

    用法:
      vta = VTA()
      result = vta.process(
          valence=0.2, delta_valence=0.05,
          F_body=0.3, delta_F_body=-0.02,
          social_reward=0.5, novelty=0.1, arousal=0.5,
          base_learn_rate=0.05,
      )
      # result['learn_rate_multiplier'] → 应用到海马学习率
      # result['motivation'] → 行为激活水平
    """

    def __init__(self):
        self.rpe_model = RPEModel()
        self.da_dynamics = DopamineDynamics()

        # 动机信号 EMA (DA驱动的行为激活)
        self._motivation_ema: float = 0.5

        # 追踪
        self._rpe_history: list[float] = []
        self._da_history: list[float] = []
        self._lr_mult_history: list[float] = []

    def process(self,
                valence: float = 0.0,
                delta_valence: float = 0.0,
                F_body: float = 0.0,
                delta_F_body: float = 0.0,
                social_reward: float = 0.0,
                novelty: float = 0.0,
                arousal: float = 0.5,
                base_learn_rate: float = 0.05) -> dict:
        """单步 VTA 处理.

        Args:
            valence: 当前效价 [-1, 1]
            delta_valence: Δvalence (当前 - 上一步)
            F_body: 当前身体自由能
            delta_F_body: ΔF_body (上一步 - 当前, 正=改善)
            social_reward: 社会奖赏 [0, 1] (互动质量)
            novelty: 新颖性信号 [0, 1]
            arousal: 唤醒度 [0, 1]
            base_learn_rate: 基础学习率 (theta.learn_rate_l0)

        Returns:
            dict with:
              'rpe': 奖赏预测误差
              'predicted_reward': 预测奖赏
              'tonic_da': 紧张性DA
              'phasic_da': 时相性DA
              'total_da': 总DA
              'learn_rate_multiplier': 学习率乘数 [0.3, 3.0]
              'motivation': 行为激活 [0, 1]
              'is_rewarding': 当前为奖赏事件
              'is_punishing': 当前为惩罚事件
        """
        # ---- 1. RPE 计算 ----
        rpe, predicted = self.rpe_model.compute_rpe(
            valence_delta=delta_valence,
            fbody_delta=delta_F_body,
            social_reward=social_reward,
            novelty=novelty,
        )

        # ---- 2. DA 动力学 ----
        da_result = self.da_dynamics.process(
            rpe=rpe,
            novelty=novelty,
            arousal=arousal,
        )

        # ---- 3. 学习率调制 ----
        # DA 偏离基线 → 学习率缩放
        da_deviation = da_result['total_da'] - DA_TONIC_BASELINE

        if da_deviation > 0:
            # 正 RPE → 增强学习 (DA > baseline)
            # 映射: deviation [0, 0.7] → multiplier [1.0, 3.0]
            lr_mult = 1.0 + (LR_MULT_MAX - 1.0) * (da_deviation / 0.7)
        else:
            # 负 RPE → 抑制学习 (DA < baseline)
            # 映射: deviation [-0.25, 0] → multiplier [0.3, 1.0]
            lr_mult = 1.0 - (1.0 - LR_MULT_MIN) * (abs(da_deviation) / 0.25)

        lr_mult = float(np.clip(lr_mult, LR_MULT_MIN, LR_MULT_MAX))

        # 考虑时相性DA的特殊效应: 正phasic burst → 临时学习率大幅提升
        if da_result['phasic_da'] > 0.3:
            # 强时相性爆发 → 额外增强 (模拟 LTP 诱导)
            phasic_boost = 1.0 + 0.5 * (da_result['phasic_da'] - 0.3)
            lr_mult *= phasic_boost

        lr_mult = float(np.clip(lr_mult, LR_MULT_MIN, LR_MULT_MAX * 1.5))

        # ---- 4. 动机信号 ----
        # DA 驱动 motivation (wanting, 非 liking)
        motivation_target = float(np.clip(
            0.3 + 0.5 * da_result['total_da'] + 0.2 * novelty,
            0.0, 1.0))
        self._motivation_ema = float(np.clip(
            0.9 * self._motivation_ema + 0.1 * motivation_target,
            0.1, 1.0))

        # ---- 追踪 ----
        self._rpe_history.append(rpe)
        self._da_history.append(da_result['total_da'])
        self._lr_mult_history.append(lr_mult)
        if len(self._rpe_history) > 100:
            self._rpe_history = self._rpe_history[-100:]
            self._da_history = self._da_history[-100:]
            self._lr_mult_history = self._lr_mult_history[-100:]

        return {
            'rpe': rpe,
            'predicted_reward': predicted,
            'tonic_da': da_result['tonic_da'],
            'phasic_da': da_result['phasic_da'],
            'total_da': da_result['total_da'],
            'learn_rate_multiplier': lr_mult,
            'motivation': self._motivation_ema,
            'is_rewarding': rpe > 0.2,
            'is_punishing': rpe < -0.2,
        }

    def reset(self):
        self.rpe_model.reset()
        self.da_dynamics.reset()
        self._motivation_ema = 0.5
        self._rpe_history = []
        self._da_history = []
        self._lr_mult_history = []
