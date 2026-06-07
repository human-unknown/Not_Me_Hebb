"""
vlpo.py — 视前区腹外侧核 (Ventrolateral Preoptic Area) [v6.3]

对应脑区: 下丘脑前部 — VLPO (视前区腹外侧核)
所属层级: 脑干 → 脑桥 → VLPO (促睡眠核心中枢)

功能职责:
  - NREM 睡眠启动和维持的核心中枢
  - Saper 触发器开关: VLPO ⇄ 觉醒系统 (LC/TMN/DR) 相互抑制
  - 接收 Process S (睡眠压力) + 昼夜节律信号 → 触发睡眠
  - NREM/REM 周期振荡 (~90min 等价, ~30 steps)
  - REM 守门人/产生者控制: REM-off (vlPAG) ↔ REM-on (SLD Glu)
  - REM 各特征控制: 肌张力消失 + 快速眼动 + NE→0 + PGO波

在 NotMe 中的实现 (v6.3):
  1. FlipFlopSwitch: 双稳态触发器 — VLPO ↔ 觉醒系统
  2. NREM_REM_Oscillator: NREM/REM 周期振荡器
  3. VLPO: 顶层编排 — 睡眠阶段判定 + NE/ACh 环境调制

设计参考:
  - Saper, C. B., Scammell, T. E., & Lu, J. (2005).
    Hypothalamic regulation of sleep and circadian rhythms. Nature.
  - Saper, C. B., Fuller, P. M., Pedersen, N. P., Lu, J., & Scammell, T. E. (2010).
    Sleep state switching. Neuron.
  - Scammell, T. E., Arrigoni, E., & Lipton, J. O. (2017).
    Neural circuitry of wakefulness and sleep. Neuron.
"""

import numpy as np
from cns.data_types import SleepState

# ============================================================
# 常量
# ============================================================

# 触发器开关参数
FLIP_FLOP_HYSTERESIS = 0.15    # 迟滞 — 防止频繁切换 (睡眠需要更强信号)
VLPO_ACTIVATION_RATE = 0.3     # VLPO 激活速度
AROUSAL_CENTER_DECAY = 0.85    # 觉醒中枢在睡眠中的衰减因子

# NREM/REM 周期参数
NREM_REM_CYCLE_STEPS = 30      # 一个 NREM→REM 周期 (≈90min 等价)
MIN_NREM_DURATION = 8          # 最短 NREM 持续步数 (防止过早 REM)
MIN_REM_DURATION = 4           # 最短 REM 持续步数
MAX_REM_DURATION = 12          # 最长 REM 持续步数 (REM 不应过长)

# REM 振荡器参数
REM_ON_GROWTH = 0.08           # REM-on 神经元在 NREM 中的增长速率
REM_OFF_DECAY_IN_NREM = 0.95   # REM-off 在 NREM 中的衰减
REM_ON_DECAY_IN_REM = 0.90     # REM-on 在 REM 中逐渐衰减 (终止REM)
REM_OFF_GROWTH_IN_REM = 0.12   # REM-off 在 REM 中重新积累

# NE/ACh 调制参数
NE_SLEEP_LEVEL = 0.05          # NREM 睡眠中 NE 基线
NE_REM_LEVEL = 0.001           # REM 睡眠中 NE → 0 (去甲肾上腺素能静默)
ACH_SLEEP_LEVEL = 0.15         # NREM 中 ACh 基线
ACH_REM_LEVEL = 0.70           # REM 中 ACh 显著升高


class FlipFlopSwitch:
    """Saper 触发器开关 — VLPO ⇄ 觉醒系统 相互抑制.

    双稳态:
      State A (清醒): VLPO 低, 觉醒中枢高
      State B (睡眠): VLPO 高, 觉醒中枢低

    触发睡眠需要 sleep_propensity > threshold (含迟滞).
    唤醒需要 sleep_propensity < threshold - hysteresis (出睡眠更难).

    用法:
      ffs = FlipFlopSwitch()
      is_asleep, vlpo_act, arousal_act = ffs.update(
          sleep_propensity=0.7, threshold=0.65)
    """

    def __init__(self):
        self.vlpo_activation: float = 0.0      # VLPO 激活度 [0, 1]
        self.arousal_center_activity: float = 0.6  # 觉醒中枢活动 [0, 1]
        self._is_asleep: bool = False
        self._stable_counter: int = 20          # 初始稳定在清醒

    def update(self, sleep_propensity: float,
               threshold: float = 0.65) -> tuple[bool, float, float]:
        """单步更新触发器开关状态.

        Args:
            sleep_propensity: 当前综合睡眠倾向 [0, 1]
            threshold: 睡眠触发阈值

        Returns:
            (is_asleep, vlpo_activation, arousal_center_activity)
        """
        hysteresis = FLIP_FLOP_HYSTERESIS

        if not self._is_asleep:
            # 清醒 → 睡眠: 需要超过阈值
            if sleep_propensity > threshold:
                self._is_asleep = True
                self._stable_counter = 0

            # VLPO 受睡眠倾向驱动
            self.vlpo_activation = float(np.clip(
                self.vlpo_activation * 0.9 +
                VLPO_ACTIVATION_RATE * sleep_propensity * 0.5,
                0.0, 1.0))

            # 觉醒中枢保持高 (清醒)
            self.arousal_center_activity = float(np.clip(
                self.arousal_center_activity * 0.95 +
                0.05 * (1.0 - sleep_propensity) * 1.5,
                0.1, 1.0))
        else:
            # 睡眠 → 清醒: 需要低于 (threshold - hysteresis)
            if sleep_propensity < (threshold - hysteresis):
                self._is_asleep = False
                self._stable_counter = 0

            # VLPO 在睡眠中保持高
            self.vlpo_activation = float(np.clip(
                self.vlpo_activation * 0.95 +
                0.05 * 0.8,  # 缓慢维持
                0.0, 1.0))

            # 觉醒中枢被 VLPO 抑制
            arousal_target = 0.05  # 睡眠中极低
            self.arousal_center_activity = float(np.clip(
                self.arousal_center_activity * AROUSAL_CENTER_DECAY +
                (1.0 - AROUSAL_CENTER_DECAY) * arousal_target,
                0.01, 1.0))

        # 稳定性追踪
        if self._is_asleep:
            self._stable_counter += 1

        return (self._is_asleep,
                self.vlpo_activation,
                self.arousal_center_activity)

    @property
    def is_asleep(self) -> bool:
        return self._is_asleep

    @property
    def is_stable(self) -> bool:
        """已在当前状态稳定 ≥5 步."""
        return self._stable_counter > 5

    def reset(self):
        self.__init__()


class NREM_REM_Oscillator:
    """NREM/REM 周期振荡器 — 脑桥 REM-on/REM-off 相互抑制.

    模型:
      REM-on (SLD Glu): 在 NREM 中缓慢积累 → 达到阈值触发 REM
      REM-off (vlPAG GABA): 在 NREM 中衰减, 在 REM 中被重新激活

    周期: ~30 steps (~90min 等价)
    前半夜: NREM 主导 (长 NREM, 短 REM)
    后半夜: REM 主导 (短 NREM, 长 REM)

    用法:
      osc = NREM_REM_Oscillator()
      phase, rem_on, rem_off = osc.update(
          is_asleep=True, nrem_ratio=0.65, cycle_position=0.3)
    """

    def __init__(self):
        self.rem_on: float = 0.0           # REM-on 神经元活动 [0, 1]
        self.rem_off: float = 0.6          # REM-off 神经元活动 [0, 1]
        self.in_rem: bool = False
        self.phase: str = 'nrem'           # 'nrem' | 'rem'
        self._cycle_step: int = 0          # 当前周期内的步数
        self._total_cycles: int = 0        # 总周期数
        self._nrem_steps_this_cycle: int = 0
        self._rem_steps_this_cycle: int = 0
        self._time_of_night: float = 0.0   # 0=入睡, 1=将醒

    def update(self, is_asleep: bool,
               nrem_ratio: float = 0.65,
               time_in_sleep: int = 0) -> dict:
        """单步更新 NREM/REM 周期.

        Args:
            is_asleep: 是否在睡眠中
            nrem_ratio: NREM 占比 (前半夜高, 后半夜低 — 由 circadian 位置决定)
            time_in_sleep: 本次睡眠已持续步数

        Returns:
            dict with phase, rem_on, rem_off, nrem_steps, rem_steps
        """
        if not is_asleep:
            # 觉醒 → 重置 REM 振荡器
            self.rem_on = 0.0
            self.rem_off = 0.6
            self.in_rem = False
            self.phase = 'nrem'
            self._cycle_step = 0
            self._nrem_steps_this_cycle = 0
            self._rem_steps_this_cycle = 0
            return self._get_state_dict()

        self._cycle_step += 1
        self._time_of_night = min(1.0, time_in_sleep / 200.0)  # 约 200 步睡眠 = 充满

        if self.phase == 'nrem':
            self._nrem_steps_this_cycle += 1
            # REM-off 在 NREM 中衰减 (解除对 REM-on 的抑制)
            self.rem_off *= REM_OFF_DECAY_IN_NREM

            # REM-on 在 NREM 中缓慢增长
            # 后半夜增长更快 (nrem_ratio 降低)
            growth_rate = REM_ON_GROWTH * (1.0 + 0.5 * (1.0 - nrem_ratio))
            self.rem_on = min(1.0, self.rem_on + growth_rate)

            # 触发 REM 条件:
            # 1. REM-on > 0.7
            # 2. NREM 已持续足够久 (≥ MIN_NREM_DURATION)
            # 3. REM-off 已充分衰减
            if (self.rem_on > 0.7 and
                self._nrem_steps_this_cycle >= MIN_NREM_DURATION and
                self.rem_off < 0.4):
                self.phase = 'rem'
                self.in_rem = True
                self._rem_steps_this_cycle = 0

        elif self.phase == 'rem':
            self._rem_steps_this_cycle += 1
            # REM-on 在 REM 中逐渐衰减 (终止REM)
            self.rem_on *= REM_ON_DECAY_IN_REM

            # REM-off 在 REM 中重新积累 (准备终止 REM)
            self.rem_off = min(1.0, self.rem_off + REM_OFF_GROWTH_IN_REM)

            # 退出 REM 条件:
            # 1. REM-on 已衰减 (< 0.3)
            # 2. REM 持续时间足够 (≥ MIN_REM_DURATION)
            # 3. REM 不应过长 (MAX_REM_DURATION) 或 REM-off 恢复充分
            exit_rem = False
            if self.rem_on < 0.3 and self._rem_steps_this_cycle >= MIN_REM_DURATION:
                exit_rem = True
            if self._rem_steps_this_cycle >= MAX_REM_DURATION:
                exit_rem = True
            if self.rem_off > 0.7 and self._rem_steps_this_cycle >= MIN_REM_DURATION:
                exit_rem = True

            if exit_rem:
                self.phase = 'nrem'
                self.in_rem = False
                self._nrem_steps_this_cycle = 0
                self._total_cycles += 1
                # 新周期: REM-on 重置, REM-off 开始衰减
                self.rem_on = 0.15
                # rem_off 保持当前值 (逐渐衰减)

        return self._get_state_dict()

    def _get_state_dict(self) -> dict:
        return {
            'phase': self.phase,
            'is_rem': self.in_rem,
            'rem_on': float(self.rem_on),
            'rem_off': float(self.rem_off),
            'nrem_steps': self._nrem_steps_this_cycle,
            'rem_steps': self._rem_steps_this_cycle,
            'total_cycles': self._total_cycles,
        }

    @property
    def total_cycles(self) -> int:
        return self._total_cycles

    def reset(self):
        self.__init__()


class VLPO:
    """视前区腹外侧核 — 睡眠-觉醒调控中心 (v6.3).

    整合:
      - FlipFlopSwitch: VLPO⇄觉醒系统 双稳态触发器
      - NREM_REM_Oscillator: NREM/REM 周期振荡
      - NE/ACh 环境调制: 各阶段特征性神经化学环境

    输出:
      - sleep_state: SleepState dataclass
      - ne_level: NE 水平 (供 LC 模块)
      - ach_level: ACh 水平 (供认知模块)
      - sleep_stage: N1/N2/N3/REM 判定

    用法:
      vlpo = VLPO()
      state: SleepState = vlpo.update(
          sleep_propensity=0.7, sleep_pressure=0.65,
          threshold=0.65, nrem_ratio=0.65,
      )
    """

    def __init__(self):
        self.flip_flop = FlipFlopSwitch()
        self.oscillator = NREM_REM_Oscillator()
        self.sleep_state = SleepState()

        # 追踪
        self._total_sleep_steps: int = 0
        self._total_nrem_steps: int = 0
        self._total_rem_steps: int = 0
        self._n_sleep_episodes: int = 0
        self._was_asleep: bool = False

    def update(self,
               sleep_propensity: float = 0.0,
               sleep_pressure: float = 0.0,
               threshold: float = 0.65,
               nrem_ratio: float = 0.65,
               ) -> SleepState:
        """单步 VLPO 处理 — 判定睡眠阶段.

        Args:
            sleep_propensity: 综合睡眠倾向 [0, 1] (来自 SCN)
            sleep_pressure: Process S 压力 [0, 1]
            threshold: 睡眠触发阈值
            nrem_ratio: NREM 占睡眠时间比例 (前半夜高)

        Returns:
            SleepState — 完整睡眠-觉醒状态
        """
        # ---- 1. 触发器开关 ----
        is_asleep, vlpo_act, arousal_act = self.flip_flop.update(
            sleep_propensity=sleep_propensity, threshold=threshold)

        # 追踪睡眠片段
        if is_asleep and not self._was_asleep:
            self._n_sleep_episodes += 1
        self._was_asleep = is_asleep

        if is_asleep:
            self._total_sleep_steps += 1
        else:
            # 清醒时振荡器也更新 (但不触发 REM)
            self.oscillator.update(is_asleep=False)

        # ---- 2. NREM/REM 振荡器 ----
        osc_result = self.oscillator.update(
            is_asleep=is_asleep,
            nrem_ratio=nrem_ratio,
            time_in_sleep=self._total_sleep_steps if is_asleep else 0,
        )

        # ---- 3. 睡眠阶段判定 ----
        if not is_asleep:
            state = 'awake'
            phase = 'none'
            # 清醒期 NE/ACh
            ne_level = 0.20
            ach_level = 0.40
        elif osc_result['phase'] == 'rem':
            phase = 'rem'
            state = 'rem'
            ne_level = NE_REM_LEVEL  # REM → NE 归零
            ach_level = ACH_REM_LEVEL  # REM → ACh 飙升
            self._total_rem_steps += 1
        else:
            phase = 'nrem'
            self._total_nrem_steps += 1
            # NREM 深度判定 (基于睡眠压力 + VLPO 激活度)
            # 高睡眠压力+高VLPO → N3 深睡; 否则 N1/N2
            nrem_depth = sleep_pressure * 0.6 + vlpo_act * 0.4
            if nrem_depth > 0.7:
                state = 'nrem_n3'   # 深睡/慢波睡眠
            elif nrem_depth > 0.45:
                state = 'nrem_n2'   # 中睡 (纺锤波+K复合波)
            else:
                state = 'nrem_n1'   # 浅睡
            ne_level = NE_SLEEP_LEVEL
            ach_level = ACH_SLEEP_LEVEL

        # ---- 4. 周期位置 ----
        cycle_position = 0.0
        if is_asleep:
            osc_phase = osc_result['phase']
            if osc_phase == 'nrem':
                cycle_position = (osc_result['nrem_steps'] /
                                  max(NREM_REM_CYCLE_STEPS, 1))
            else:
                cycle_position = (NREM_REM_CYCLE_STEPS +
                                  osc_result['rem_steps']) / (2.0 * NREM_REM_CYCLE_STEPS)
            cycle_position = float(np.clip(cycle_position, 0.0, 1.0))

        # ---- 5. 构建 SleepState ----
        self.sleep_state = SleepState(
            state=state,
            phase=phase,
            sleep_cycle_count=self.oscillator.total_cycles,
            time_in_state=(self._total_sleep_steps
                          if is_asleep else 0),
            time_in_phase=(osc_result['nrem_steps']
                          if phase == 'nrem' else osc_result['rem_steps']),
            cycle_position=cycle_position,
            vlpo_activation=vlpo_act,
            arousal_center_activity=arousal_act,
            flip_flop_stable=self.flip_flop.is_stable,
            rem_on_activity=osc_result['rem_on'],
            rem_off_activity=osc_result['rem_off'],
            total_sleep_steps=self._total_sleep_steps,
            total_nrem_steps=self._total_nrem_steps,
            total_rem_steps=self._total_rem_steps,
            n_sleep_episodes=self._n_sleep_episodes,
        )

        return self.sleep_state

    def get_ne_level(self) -> float:
        """返回当前 NE 水平 — 供 LC 模块在睡眠期调制."""
        if self.sleep_state.state == 'rem':
            return NE_REM_LEVEL
        elif self.sleep_state.state == 'awake':
            return 0.20  # 清醒基线 → LC 自行计算
        else:
            return NE_SLEEP_LEVEL

    def get_ach_level(self) -> float:
        """返回当前 ACh 水平 — 供认知模块."""
        if self.sleep_state.state == 'rem':
            return ACH_REM_LEVEL
        elif self.sleep_state.state == 'awake':
            return 0.40
        else:
            return ACH_SLEEP_LEVEL

    @property
    def is_asleep(self) -> bool:
        return self.flip_flop.is_asleep

    @property
    def is_in_rem(self) -> bool:
        return self.oscillator.in_rem

    @property
    def is_in_nrem(self) -> bool:
        return self.is_asleep and not self.is_in_rem

    def reset(self):
        self.flip_flop.reset()
        self.oscillator.reset()
        self.sleep_state = SleepState()
        self._total_sleep_steps = 0
        self._total_nrem_steps = 0
        self._total_rem_steps = 0
        self._n_sleep_episodes = 0
        self._was_asleep = False
