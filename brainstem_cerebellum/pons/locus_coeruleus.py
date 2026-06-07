"""
locus_coeruleus.py — 蓝斑核 (Locus Coeruleus) [v5.5]

对应脑区: 蓝斑核 (A6 细胞群, 脑桥背侧)
所属层级: 脑干 → 脑桥 → 蓝斑核

功能职责:
  - 去甲肾上腺素 (NE) 的唯一来源 → 全脑投射 (除基底节外)
  - 唤醒度调控: 低 NE = 睡眠/放松, 中 NE = 专注, 高 NE = 应激/焦虑
  - 注意力: 增强信噪比 (signal-to-noise)
  - 新颖性检测: 新异刺激 → 阶段性 NE 爆发
  - 应激反应: 威胁 → 强直性 NE 升高
  - 决策: 高 NE → exploitation (利用已知), 低 NE → exploration (探索)

  Yerkes-Dodson 曲线 (Aston-Jones & Cohen 2005):
    低唤醒 → 低表现 (无聊/睡眠)
    中唤醒 → 最佳表现 (专注/flow)
    高唤醒 → 低表现 (应激/焦虑)

在 NotMe 中的实现 (v5.5):
  1. NEDynamics: 阶段性(phasic) vs 强直性(tonic) NE 动态
  2. SNREnhancer: NE → 感觉通道 SNR 增强
  3. LocusCoeruleus: 顶层 — NE 输出 → RVM 痛觉下行调制 + FPN 探索/利用

设计参考:
  - Aston-Jones, G., & Cohen, J. D. (2005). An integrative theory of locus
    coeruleus-norepinephrine function: adaptive gain and optimal performance.
  - Sara, S. J. (2009). The locus coeruleus and noradrenergic modulation of cognition.
"""

import numpy as np
from typing import Optional

# ============================================================
# 常量
# ============================================================

# NE 动力学
NE_TONIC_BASELINE = 0.20        # 强直性NE基线 (安静清醒)
NE_PHASIC_DECAY = 0.30          # 时相性NE快速衰减 (~100ms尺度)
NE_PHASIC_MAX = 1.0             # 时相性NE上限
NE_TONIC_TAU = 0.90             # 强直性NE EMA时间常数
NE_TONIC_MAX = 0.80             # 强直性NE上限 (避免过度兴奋)

# Yerkes-Dodson 倒U曲线参数
YD_OPTIMAL_NE = 0.40            # 最佳NE水平 (peak of inverted-U)
YD_WIDTH = 0.20                 # 倒U曲线宽度

# 探索/利用阈值
EXPLORE_NE_THRESHOLD = 0.30     # NE低于此 → 探索主导
EXPLOIT_NE_THRESHOLD = 0.55     # NE高于此 → 利用主导 (但过高=应激→无效)


# ============================================================
# NE 动力学
# ============================================================

class NEDynamics:
    """时相性(Phasic) vs 强直性(Tonic) NE 动态 (Aston-Jones & Cohen 2005).

    时相性 NE: 事件驱动爆发 — 新颖刺激 → 快速 NE 释放 → 快速衰减
    强直性 NE: 唤醒度驱动的基线 — 慢速调制, 受应激水平和任务参与影响

    用法:
      ne = NEDynamics()
      result = ne.process(arousal=0.5, novelty=0.3, stress=0.1, task_engagement=0.6)
    """

    def __init__(self):
        self._tonic_ne: float = NE_TONIC_BASELINE
        self._phasic_ne: float = 0.0

        # 应激累积 EMA (慢性应激 → 强直性NE持续升高)
        self._stress_ema: float = 0.0

        # 新颖性爆发追踪
        self._novelty_burst_counter: int = 0

    def process(self,
                arousal: float = 0.5,
                novelty: float = 0.0,
                stress: float = 0.0,
                task_engagement: float = 0.3) -> dict:
        """更新NE动态.

        Args:
            arousal: 当前唤醒度 [0, 1] (来自 cingulate.compute_valence_arousal)
            novelty: 新颖性信号 [0, 1] (来自 TPN salience section)
            stress: 应激水平 [0, 1] (高F_body + 高唤醒)
            task_engagement: 任务参与度 [0, 1] (来自 TPN activation)

        Returns:
            dict with tonic_ne, phasic_ne, total_ne, mode
        """
        # ---- 时相性 NE: 新颖性触发爆发 ----
        if novelty > 0.3:
            # 新颖刺激 → 阶段性NE爆发 (phasic response)
            phasic_burst = novelty * 0.6
            self._phasic_ne = float(np.clip(
                self._phasic_ne + phasic_burst,
                0.0, NE_PHASIC_MAX))
            self._novelty_burst_counter += 1
        else:
            self._novelty_burst_counter = max(0, self._novelty_burst_counter - 1)

        # 时相性NE快速衰减
        self._phasic_ne *= NE_PHASIC_DECAY

        # ---- 强直性 NE: 多因素慢速调制 ----
        # 应激累积EMA
        self._stress_ema = float(np.clip(
            0.95 * self._stress_ema + 0.05 * stress,
            0.0, 1.0))

        # 强直性NE目标:
        # - 唤醒度 ↑ → tonic NE ↑ (主要驱动力)
        # - 应激累积 ↑ → tonic NE ↑ (慢性应激升高基线)
        # - 任务参与 ↑ → tonic NE ↑ (专注状态需要适中NE)
        # - 但任务参与过高 (强迫) → tonic NE 略微回落 (防止burnout)
        task_modulation = task_engagement
        if task_engagement > 0.8:
            # 过高任务参与 → 轻微回落 (避免强直性过载)
            task_modulation = 0.8 - 0.3 * (task_engagement - 0.8)

        tonic_target = float(np.clip(
            NE_TONIC_BASELINE
            + 0.35 * (arousal - 0.3)          # 唤醒度主驱动
            + 0.15 * self._stress_ema           # 慢性应激偏移
            + 0.10 * task_modulation,           # 任务参与调制
            0.02, NE_TONIC_MAX))

        # 慢速EMA向目标 (强直性变化慢, τ~数分钟)
        self._tonic_ne = float(np.clip(
            NE_TONIC_TAU * self._tonic_ne
            + (1.0 - NE_TONIC_TAU) * tonic_target,
            0.01, NE_TONIC_MAX))

        # ---- 总 NE = tonic + phasic ----
        total_ne = float(np.clip(self._tonic_ne + self._phasic_ne, 0.0, 1.0))

        # ---- NE 模式 (Aston-Jones 分类) ----
        if self._tonic_ne < 0.15:
            mode = 'disengaged'    # 低NE → 脱离/睡眠
        elif self._phasic_ne > 0.3:
            mode = 'phasic'        # 阶段性NE → 注意重定向
        elif self._tonic_ne < 0.45:
            mode = 'tonic_optimal' # 中等强直性 → 最佳表现 (exploitation)
        else:
            mode = 'tonic_high'    # 高强直性 → 应激/焦虑 (扫描但不专注)

        return {
            'tonic_ne': self._tonic_ne,
            'phasic_ne': self._phasic_ne,
            'total_ne': total_ne,
            'mode': mode,
            'stress_ema': self._stress_ema,
        }

    def reset(self):
        self._tonic_ne = NE_TONIC_BASELINE
        self._phasic_ne = 0.0
        self._stress_ema = 0.0
        self._novelty_burst_counter = 0


# ============================================================
# SNR 增强器
# ============================================================

class SNREnhancer:
    """NE → 信噪比增强.

    NE 增强感觉通道的信号-噪声比:
      - 中NE (0.3-0.5): 最佳SNR增强 — 增强相关通道, 抑制噪声
      - 低NE (<0.2): SNR不增强 — 感觉通道均匀响应
      - 高NE (>0.6): SNR反而下降 — 过度兴奋 → 噪声放大

    这对应 Yerkes-Dodson 倒U曲线在感知层面的体现.

    用法:
      snr = SNREnhancer()
      result = snr.enhance(sensory_vector, ne_level=0.4, attention_mask=None)
    """

    def __init__(self):
        self._snr_gain_ema: float = 1.0

    def enhance(self,
                sensory: np.ndarray,
                ne_level: float,
                attention_mask: Optional[np.ndarray] = None) -> dict:
        """NE驱动的SNR增强.

        Args:
            sensory: 感知向量 (D,) 或 任意形状
            attention_mask: 注意力掩码 (D,) — 标记"信号"通道 (1=信号, 0=噪声)
                           如果为None, 使用全1掩码

        Returns:
            dict with enhanced_sensory, snr_gain, yd_performance
        """
        s = np.asarray(sensory, dtype=np.float32).copy()

        if attention_mask is None:
            attention_mask = np.ones_like(s)

        mask = np.asarray(attention_mask, dtype=np.float32)

        # ---- Yerkes-Dodson 倒U增益曲线 ----
        # 峰值在 YD_OPTIMAL_NE (0.4), 宽度由 YD_WIDTH 控制
        # gain(ne) = exp(-(ne - optimal)² / (2 * width²))
        yd_gain = float(np.exp(
            -((ne_level - YD_OPTIMAL_NE) ** 2) / (2.0 * YD_WIDTH ** 2)))

        # 性能水平归一化到 [0.3, 1.0] (即使在最差NE水平也有基础SNR)
        yd_performance = 0.3 + 0.7 * yd_gain

        # SNR增益: 信号通道增强, 噪声通道抑制
        snr_gain = 1.0 + 0.3 * (yd_performance - 0.5)
        noise_suppression = 1.0 - 0.15 * (1.0 - yd_performance)

        # 应用增强
        signal_boost = snr_gain * mask + noise_suppression * (1.0 - mask)
        enhanced = s * signal_boost.reshape(s.shape)

        # EMA追踪
        self._snr_gain_ema = 0.9 * self._snr_gain_ema + 0.1 * snr_gain

        return {
            'enhanced_sensory': enhanced.astype(np.float32),
            'snr_gain': float(snr_gain),
            'yd_performance': yd_performance,
            'yd_gain': yd_gain,
            'signal_boost': signal_boost,
        }

    def reset(self):
        self._snr_gain_ema = 1.0


# ============================================================
# 蓝斑核 — 顶层NE系统
# ============================================================

class LocusCoeruleus:
    """蓝斑核NE系统 — 唤醒度 + 注意力 + SNR + 探索/利用 + RVM下行痛觉调制.

    关键输出:
      - ne_tonic → RVM._norepinephrine_tone (v5.4 预留接口, v5.5 接入)
      - SNR增强 → 感觉通道信噪比优化
      - Yerkes-Dodson → 表现水平评估
      - Explore/Exploit bias → FPN 探索-利用平衡

    用法:
      lc = LocusCoeruleus()
      result = lc.process(
          arousal=0.5, novelty=0.1, stress=0.2,
          F_body=0.3, task_engagement=0.5,
      )
      # result['ne_tonic'] → 写入 rvm._norepinephrine_tone
      # result['snr_gain'] → 应用到 sensory
      # result['exploration_bias'] → 调制 explore/exploit
    """

    def __init__(self):
        self.ne_dynamics = NEDynamics()
        self.snr_enhancer = SNREnhancer()

        # 探索/利用追踪
        self._explore_exploit_ema: float = 0.0  # 正=利用, 负=探索
        self._yd_history: list[float] = []

    def process(self,
                arousal: float = 0.5,
                novelty: float = 0.0,
                stress: float = 0.0,
                F_body: float = 0.0,
                task_engagement: float = 0.3,
                sensory: Optional[np.ndarray] = None,
                attention_mask: Optional[np.ndarray] = None) -> dict:
        """单步 LC 处理.

        Args:
            arousal: 当前唤醒度 [0, 1]
            novelty: 新颖性信号 [0, 1]
            stress: 应激水平 [0, 1]
            F_body: 身体自由能
            task_engagement: 任务参与度 [0, 1] (TPN激活)
            sensory: 可选 — 当前感知向量 (用于SNR增强)
            attention_mask: 可选 — FPN注意力模板 (用于SNR增强)

        Returns:
            dict with:
              'tonic_ne': 强直性NE — 写入 RVM._norepinephrine_tone
              'phasic_ne': 时相性NE
              'total_ne': 总NE
              'ne_mode': NE模式 (disengaged/phasic/tonic_optimal/tonic_high)
              'snr_gain': SNR增益
              'yd_performance': Yerkes-Dodson 表现水平
              'exploration_bias': 探索偏移 [-1, 1] (负=探索, 正=利用)
              'enhanced_sensory': SNR增强后的感知 (如提供sensory)
        """
        # ---- 1. NE 动力学 ----
        ne_result = self.ne_dynamics.process(
            arousal=arousal,
            novelty=novelty,
            stress=stress,
            task_engagement=task_engagement,
        )

        # ---- 2. SNR 增强 ----
        snr_result = {'snr_gain': 1.0, 'yd_performance': 0.5, 'enhanced_sensory': sensory,
                      'yd_gain': 0.5}
        if sensory is not None:
            snr_result = self.snr_enhancer.enhance(
                sensory=sensory,
                ne_level=ne_result['total_ne'],
                attention_mask=attention_mask,
            )

        # ---- 3. Yerkes-Dodson 表现评估 ----
        yd_performance = snr_result['yd_performance']
        self._yd_history.append(yd_performance)
        if len(self._yd_history) > 100:
            self._yd_history = self._yd_history[-100:]

        # ---- 4. 探索/利用平衡 ----
        # Aston-Jones & Cohen (2005):
        #   低NE (tonic < 0.3): exploration — 扫描多种可能性
        #   中NE (0.3-0.5): exploitation — 专注当前任务
        #   高NE (>0.55): disorganized scanning — 应激, 无法专注
        tonic = ne_result['tonic_ne']

        if tonic < EXPLORE_NE_THRESHOLD:
            # 低NE → 探索主导
            explore_bias = -0.5 * (1.0 - tonic / EXPLORE_NE_THRESHOLD)
        elif tonic < EXPLOIT_NE_THRESHOLD:
            # 中NE → 利用主导 (最佳区间)
            mid = (EXPLORE_NE_THRESHOLD + EXPLOIT_NE_THRESHOLD) / 2.0
            half_width = (EXPLOIT_NE_THRESHOLD - EXPLORE_NE_THRESHOLD) / 2.0
            # 归一化到 [0, 1] 在利用区间内
            exploit_strength = (tonic - EXPLORE_NE_THRESHOLD) / (2.0 * half_width)
            explore_bias = exploit_strength  # [0, 1] → 利用
        else:
            # 高NE → 应激扫描 (伪探索, 但非有效)
            excess = (tonic - EXPLOIT_NE_THRESHOLD) / (NE_TONIC_MAX - EXPLOIT_NE_THRESHOLD)
            explore_bias = 1.0 - excess  # 逐渐回退到中性

        explore_bias = float(np.clip(explore_bias, -1.0, 1.0))
        self._explore_exploit_ema = float(np.clip(
            0.95 * self._explore_exploit_ema + 0.05 * explore_bias,
            -1.0, 1.0))

        # ---- 5. 构建结果 ----
        result = {
            'tonic_ne': ne_result['tonic_ne'],
            'phasic_ne': ne_result['phasic_ne'],
            'total_ne': ne_result['total_ne'],
            'ne_mode': ne_result['mode'],
            'stress_ema': ne_result['stress_ema'],
            'snr_gain': snr_result['snr_gain'],
            'yd_performance': yd_performance,
            'yd_gain': snr_result['yd_gain'],
            'exploration_bias': explore_bias,
            'explore_exploit_ema': self._explore_exploit_ema,
            'is_exploring': explore_bias < -0.3,
            'is_exploiting': explore_bias > 0.3,
        }

        if sensory is not None:
            result['enhanced_sensory'] = snr_result['enhanced_sensory']

        return result

    def set_sleep_ne(self, ne_level: float = 0.05, is_rem: bool = False):
        """v6.3: VLPO驱动的睡眠期NE调制.

        NREM: NE降至极低 (~0.05)
        REM:  NE归零 (~0.001) — 去甲肾上腺素能静默

        Args:
            ne_level: VLPO计算的NE水平
            is_rem: 是否在REM睡眠
        """
        self.ne_dynamics._tonic_ne = ne_level
        self.ne_dynamics._phasic_ne = 0.0
        self._explore_exploit_ema = -0.5 if is_rem else self._explore_exploit_ema

    def reset(self):
        self.ne_dynamics.reset()
        self.snr_enhancer.reset()
        self._explore_exploit_ema = 0.0
        self._yd_history = []
