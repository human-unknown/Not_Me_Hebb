"""
scn.py — 视交叉上核 (Suprachiasmatic Nucleus) [v6.3, doc v6.6]

对应脑区: 下丘脑前部 — 视交叉上核 (SCN)
所属层级: 大脑 → 边缘系统 → SCN

功能职责:
  - 昼夜节律中枢起搏器 — ~24h 内源性振荡
  - TTFL 分子钟: CLOCK/BMAL1 → PER/CRY → 反馈抑制
  - 光同步 (photoentrainment): ipRGC → 视网膜下丘脑束 → SCN 核心区
  - 输出节律: 褪黑素 (夜间) + 皮质醇 (晨峰)
  - Process S: 睡眠压力 (腺苷) 累积与清除

在 NotMe 中的实现 (v6.3):
  1. TTFL: 4-variable 简化 ODE (Per_mRNA, PER, Cry_mRNA, CRY)
  2. Process S: 觉醒期累积 + 睡眠期清除
  3. Photoentrainment: 光照 → PER 上调 → 相位重置
  4. 输出: circadian_phase, melatonin, cortisol, sleep_propensity

═══ SCN 时间 — 用户友好说明 (v6.6) ═══

SCN 是 Agent 的内置生物钟, 模拟人类 ~24h 昼夜节律:
  • 2880 Agent Steps = 1 昼夜 (1 step ≈ 30s)
  • 相位 0 (0h/24h) = 主观午夜 — 褪黑素高峰 😴
  • 相位 π/2 (6h)    = 主观清晨 — 皮质醇高峰, 自然醒来 🌅
  • 相位 π (12h)     = 主观正午 — 最高警觉 ☀
  • 相位 3π/2 (18h)  = 主观傍晚 — 开始困倦 🌆

节律输出:
  - melatonin (褪黑素):     夜间↑ → 促睡眠, 促 VLPO 激活
  - cortisol (皮质醇):      清晨↑ → 促觉醒, 促 HPA 轴
  - sleep_pressure (Process S): 觉醒累积 → 睡眠清除
  - sleep_propensity:       综合睡眠倾向 (高→VLPO 易翻转→睡眠)

对 Agent 行为的影响:
  - 睡眠期间: Agent 只做内部巩固 (记忆回放+修剪), 不对外输出
  - 高皮质醇 + 低褪黑素: 学习率高, 探索倾向强
  - 高褪黑素: 学习率低, 倾向保守/休息
  - 睡眠剥夺 (高 Process S + 强光): 认知疲劳 → F_cognitive↑

UI 显示 (Header):
  SCN 0h🌙 = 午夜 | SCN 6h🌅 = 清晨 | SCN 12h☀ = 正午 | SCN 18h🌆 = 傍晚

设计参考:
  - Takahashi, J. S. (2017). Transcriptional architecture of the mammalian circadian clock.
  - Hastings, M. H., Maywood, E. S., & Brancaccio, M. (2018).
  - Borbély, A. A. (1982). A two process model of sleep regulation.
  - 2017 Nobel Prize in Physiology or Medicine (Hall, Rosbash, Young)
"""

import numpy as np
from typing import Optional
from cns.data_types import CircadianState

# ============================================================
# 常量
# ============================================================

# TTFL 动力学参数
TTFL_K1 = 0.8          # 转录激活速率 (CLOCK:BMAL1 → Per/Cry mRNA)
TTFL_K2 = 0.15         # mRNA 降解速率
TTFL_K3 = 0.6          # 翻译速率 (mRNA → 蛋白)
TTFL_K4 = 0.12         # 蛋白降解速率 (周期长度关键决定因素)
TTFL_KD = 0.3          # PER:CRY 抑制 CLOCK:BMAL1 的半最大浓度
TTFL_HILL = 2.0        # 抑制协同系数 (Hill coefficient)
TTFL_DT = 0.05         # ODE 步长 (每 step 积分此步长)

# 光信号参数
LIGHT_PER_INDUCTION = 0.8   # 光照 → Per mRNA 最大诱导幅度
LIGHT_THRESHOLD = 0.1       # 触发 Per 诱导的最低光照

# Process S 参数
PROCESS_S_WAKE_ACCUM = 0.012    # 觉醒期每步累积速率
PROCESS_S_NREM_CLEAR = 0.94     # NREM 睡眠每步清除因子 (<1 = 清除)
PROCESS_S_REM_CLEAR = 0.975     # REM 睡眠每步清除因子 (慢于 NREM)
ALLOSTATIC_LOAD_FACTOR = 0.5    # 异稳态负荷对 Process S 累积的调制

# 输出节律参数
MELATONIN_PHASE_OFFSET = np.pi      # 褪黑素峰值在主观午夜 (相移 π)
CORTISOL_PHASE_OFFSET = 0.0         # 皮质醇峰值在主观清晨 (相移 0)
MELATONIN_SHARPNESS = 3.0           # 褪黑素节律陡峭度
CORTISOL_SHARPNESS = 2.5            # 皮质醇节律陡峭度

# 步数→昼夜时间映射
STEPS_PER_CIRCADIAN_DAY = 2880     # 1 step ≈ 30s → 2880 steps = 24h
STEPS_PER_HOUR = STEPS_PER_CIRCADIAN_DAY / 24  # 120 steps = 1 hour


class TTFL:
    """转录-翻译反馈环路 (Transcription-Translation Feedback Loop).

    简化 4-variable ODE:
      d(Pm)/dt = k1 * f(BMAL1_activity) - k2 * Pm
      d(P)/dt  = k3 * Pm - k4 * P
      d(Cm)/dt = k1 * f(BMAL1_activity) - k2 * Cm
      d(C)/dt  = k3 * Cm - k4 * C

      BMAL1_activity = 1 / (1 + ((P+C)/Kd)^n)  — 被 PER:CRY 异二聚体抑制

    用法:
      ttfl = TTFL()
      state = ttfl.step(light_level=0.3, dt=0.05)
    """

    def __init__(self):
        # 初始值: 主观清晨 (~6am), PER/CRY 开始上升
        self.per_mrna: float = 0.3
        self.per_protein: float = 0.2
        self.cry_mrna: float = 0.3
        self.cry_protein: float = 0.2

        # 自由运行周期追踪
        self._phase_estimate: float = 0.0   # 相位估计 [0, 2π]

    def _bmal1_activity(self, per: float, cry: float) -> float:
        """CLOCK:BMAL1 活性 — 被 PER:CRY 异二聚体抑制."""
        total = per + cry
        # Hill 函数: 抑制 = 1 / (1 + ([inhibitor]/Kd)^n)
        activity = 1.0 / (1.0 + (total / TTFL_KD) ** TTFL_HILL)
        return float(np.clip(activity, 0.001, 1.0))

    def step(self, light_level: float = 0.0, dt: float = TTFL_DT,
             k4_override: float = None) -> dict:
        """单步积分 TTFL ODE (Euler method).

        Args:
            light_level: 当前光照强度 [0, 1]
            dt: 积分步长
            k4_override: 蛋白降解速率覆盖 (None=使用默认 K4), 用于周期调制

        Returns:
            dict with per_mrna, per_protein, cry_mrna, cry_protein, bmal1_activity
        """
        pm, p, cm, c = self.per_mrna, self.per_protein, self.cry_mrna, self.cry_protein
        k4 = k4_override if k4_override is not None else TTFL_K4

        # 当前 BMAL1 活性
        bmal1_act = self._bmal1_activity(p, c)

        # ---- ODE 积分 (Euler) ----
        trans_activation = TTFL_K1 * bmal1_act

        # Per mRNA: 基础转录 + 光诱导
        light_induction = 0.0
        if light_level > LIGHT_THRESHOLD:
            light_induction = LIGHT_PER_INDUCTION * light_level
        d_pm = trans_activation + light_induction - TTFL_K2 * pm
        d_cm = trans_activation - TTFL_K2 * cm

        # 蛋白
        d_p = TTFL_K3 * pm - k4 * p
        d_c = TTFL_K3 * cm - k4 * c

        # 更新
        pm_new = float(np.clip(pm + d_pm * dt, 0.0, 2.0))
        p_new = float(np.clip(p + d_p * dt, 0.0, 2.0))
        cm_new = float(np.clip(cm + d_cm * dt, 0.0, 2.0))
        c_new = float(np.clip(c + d_c * dt, 0.0, 2.0))

        self.per_mrna, self.per_protein = pm_new, p_new
        self.cry_mrna, self.cry_protein = cm_new, c_new
        new_bmal1 = self._bmal1_activity(p_new, c_new)

        # 相位估计: 基于 PER 蛋白峰谷追踪
        # PER蛋白正弦近似相位
        per_mean = 0.5  # PER 蛋白大致均值
        per_amp = max(0.05, abs(p_new - per_mean))
        per_phase_shift = (p_new - per_mean) / per_amp
        self._phase_estimate = float(np.arctan2(
            per_phase_shift, bmal1_act - 0.5))

        return {
            'per_mrna': pm_new,
            'per_protein': p_new,
            'cry_mrna': cm_new,
            'cry_protein': c_new,
            'bmal1_activity': new_bmal1,
            'light_induction': light_induction,
        }

    def get_circadian_phase(self) -> float:
        """返回归一化昼夜相位 [0, 2π]."""
        # 使用 PER 蛋白水平 + BMAL1 活性 计算相位
        # PER 高 + BMAL1 低 = 主观夜晚
        # PER 低 + BMAL1 高 = 主观清晨
        raw = float(np.arctan2(
            self.per_protein - 0.3,          # y: PER偏离基线
            self._bmal1_activity(self.per_protein, self.cry_protein) - 0.4  # x: BMAL1偏离
        ))
        # 映射到 [0, 2π]
        return float(raw % (2.0 * np.pi))

    def reset(self):
        self.__init__()


class ProcessS:
    """Process S — 睡眠压力 (腺苷类似物).

    觉醒期累积, 睡眠期清除. 与 Process C (昼夜节律) 共同决定睡眠倾向.

    Borbély 双过程模型:
      Process S: 觉醒 → 指数上升, 睡眠 → 指数衰减
      Process C: SCN 昼夜节律输出

    用法:
      ps = ProcessS()
      pressure = ps.update(is_asleep=False, allostatic_load=0.0, sleep_phase='nrem')
    """

    def __init__(self):
        self.pressure: float = 0.0         # 当前睡眠压力 [0, 1]
        self._wake_since: int = 0           # 自上次睡眠以来的步数

    def update(self, is_asleep: bool = False,
               allostatic_load: float = 0.0,
               sleep_phase: str = 'nrem') -> float:
        """单步更新 Process S.

        Args:
            is_asleep: 当前是否在睡眠中
            allostatic_load: 异稳态负荷 [0, 1] (高负荷 → 更快累积)
            sleep_phase: 睡眠阶段 ('nrem' | 'rem') — NREM 清除更快

        Returns:
            current_pressure: 更新后的睡眠压力 [0, 1]
        """
        if is_asleep:
            # 睡眠 → 腺苷清除
            self._wake_since = 0
            if sleep_phase == 'nrem':
                self.pressure *= PROCESS_S_NREM_CLEAR
            else:
                self.pressure *= PROCESS_S_REM_CLEAR
        else:
            # 觉醒 → 腺苷累积
            self._wake_since += 1
            accum = PROCESS_S_WAKE_ACCUM * (
                1.0 + ALLOSTATIC_LOAD_FACTOR * allostatic_load)
            self.pressure = min(1.0, self.pressure + accum)

        return self.pressure

    @property
    def wake_duration(self) -> int:
        return self._wake_since

    def reset(self):
        self.__init__()


class SCN:
    """视交叉上核 — 昼夜节律主时钟 (v6.3).

    整合:
      - TTFL 分子钟 → ~24h 内源性节律
      - Process S → 睡眠压力累积/清除
      - 光同步 → ipRGC 通路相位重置
      - 输出: circadian_phase, melatonin, cortisol, sleep_propensity

    输出节律:
      - 褪黑素 (melatonin): 傍晚开始 → 夜间峰值 → 清晨下降
        peak at 相位 ~π (主观午夜), 促睡眠
      - 皮质醇 (cortisol): 清晨峰值 → 白天下降 → 夜间最低
        peak at 相位 ~0 (主观清晨), 促觉醒

    用法:
      scn = SCN()
      state: CircadianState = scn.step(
          light_level=0.5, is_asleep=False,
          allostatic_load=0.0, sleep_phase='none',
          circa_tau=24.0, circa_light_sensitivity=0.3,
      )
    """

    def __init__(self):
        self.ttfl = TTFL()
        self.process_s = ProcessS()
        self._step_count: int = 0
        self._light_history: list[float] = []

    def step(self,
             light_level: float = 0.0,
             is_asleep: bool = False,
             allostatic_load: float = 0.0,
             sleep_phase: str = 'none',
             circa_tau: float = 24.0,
             circa_light_sensitivity: float = 0.3,
             ) -> CircadianState:
        """单步 SCN 处理.

        Args:
            light_level: 当前光照强度 [0, 1] (ipRGC 信号)
            is_asleep: 当前是否在睡眠中
            allostatic_load: 异稳态负荷 [0, 1]
            sleep_phase: 睡眠阶段 ('none' | 'nrem' | 'rem')
            circa_tau: 内源周期 (小时, 影响 TTFL_K4)
            circa_light_sensitivity: 光同步敏感度 [0, 1]

        Returns:
            CircadianState — 完整昼夜节律状态
        """
        self._step_count += 1
        self._light_history.append(light_level)
        if len(self._light_history) > 100:
            self._light_history = self._light_history[-100:]

        # ---- 1. TTFL 步进 ----
        # 周期调制: circa_tau → 调整蛋白降解速率 K4
        # 默认 K4=0.12 对应 ~24h; 更慢降解=更长周期
        tau_ratio = 24.0 / max(circa_tau, 1.0)
        effective_k4 = TTFL_K4 * tau_ratio

        # 光敏感度调制
        effective_light = float(np.clip(
            light_level * circa_light_sensitivity, 0.0, 1.0))

        ttfl_result = self.ttfl.step(
            light_level=effective_light, dt=TTFL_DT,
            k4_override=effective_k4)

        # ---- 2. Process S 更新 ----
        sleep_pressure = self.process_s.update(
            is_asleep=is_asleep,
            allostatic_load=allostatic_load,
            sleep_phase=sleep_phase if sleep_phase != 'none' else 'nrem',
        )

        # ---- 3. 节律输出 ----
        phase = self.ttfl.get_circadian_phase()

        # 褪黑素: 在相位 ~π (主观午夜) 达到峰值
        melatonin = float(np.clip(
            0.5 + 0.5 * np.cos(phase - MELATONIN_PHASE_OFFSET),
            0.0, 1.0))
        # 锐化 (夜间快速上升)
        melatonin = float(np.clip(melatonin ** MELATONIN_SHARPNESS, 0.0, 1.0))

        # 皮质醇: 在相位 ~0 (主观清晨 ~6am) 达到峰值
        cortisol = float(np.clip(
            0.5 + 0.5 * np.cos(phase - CORTISOL_PHASE_OFFSET),
            0.0, 1.0))
        cortisol = float(np.clip(cortisol ** CORTISOL_SHARPNESS, 0.0, 1.0))

        # ---- 4. 综合睡眠倾向 ----
        # 昼夜唤醒驱力 = 皮质醇 (高皮质醇 → 不易睡)
        circadian_wake_drive = cortisol * 0.6 + 0.2

        # 睡眠倾向 = Process S * 昼夜易睡性
        # 高 Process S + 低皮质醇 → 高睡眠倾向
        sleep_propensity = float(np.clip(
            0.6 * sleep_pressure + 0.4 * (1.0 - circadian_wake_drive),
            0.0, 1.0))

        return CircadianState(
            circadian_phase=phase,
            per_mrna=ttfl_result['per_mrna'],
            per_protein=ttfl_result['per_protein'],
            cry_mrna=ttfl_result['cry_mrna'],
            cry_protein=ttfl_result['cry_protein'],
            bmal1_activity=ttfl_result['bmal1_activity'],
            melatonin=melatonin,
            cortisol=cortisol,
            sleep_pressure=sleep_pressure,
            light_level=light_level,
            sleep_propensity=sleep_propensity,
        )

    def get_time_of_day(self) -> float:
        """返回归一化时间 [0, 1] — 0=主观午夜, 0.25=6am, 0.5=正午, 0.75=6pm."""
        phase = self.ttfl.get_circadian_phase()
        return float(phase / (2.0 * np.pi))

    def get_hour(self) -> float:
        """返回近似小时 [0, 24) — 基于 TTFL 分子钟相位."""
        return self.get_time_of_day() * 24.0

    @staticmethod
    def get_reliable_hour(total_steps: int) -> float:
        """返回可靠的 24 小时制时间 [0, 24) — 基于步数线性映射.

        2880 steps = 24 hours, 120 steps = 1 hour.
        不会卡住或倒计时, 适合 UI 显示.

        Args:
            total_steps: Agent 累计步数 (从 meta.step_count)

        Returns:
            小时 [0.0, 24.0) — 0=午夜, 6=清晨, 12=正午, 18=傍晚
        """
        steps_in_day = total_steps % STEPS_PER_CIRCADIAN_DAY
        return steps_in_day / STEPS_PER_HOUR

    def reset(self):
        self.ttfl.reset()
        self.process_s.reset()
        self._step_count = 0
        self._light_history = []
