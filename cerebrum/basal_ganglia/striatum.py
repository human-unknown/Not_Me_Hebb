"""
striatum.py — 纹状体 (Striatum)  [待实现]

对应脑区: 尾状核 (Caudate) + 壳核 (Putamen) + 伏隔核 (NAc)
所属层级: 大脑 → 基底神经节 → 纹状体

功能职责:
  背侧纹状体 (Dorsal Striatum = Caudate + Putamen):
    - 感觉运动 (Putamen): 习惯形成、动作 chunking
    - 联合 (Caudate): 目标导向学习、认知灵活性
  腹侧纹状体 (Ventral Striatum = NAc):
    - 奖赏处理、动机、wanting (incentive salience)
    - 多巴胺 D1/D2 受体平衡 → 直接/间接通路

  D1 中型多棘神经元 (MSN) → 直接通路 (Go: 促进动作)
  D2 MSN → 间接通路 (No-Go: 抑制动作)

在 NotMe 中的待实现功能:
  1. 习惯学习: 经常重复的动作序列 → 自动化 (程序性记忆)
  2. 直接/间接通路: Go/NoGo 动作选择的基底节实现
  3. 奖赏预测: NAc 的 wanting vs OFC 的 liking
  4. 动作 chunking: 多步骤动作固化为单个 chunk

当前状态:
  动作选择目前由 MoEGate (action_gating.py) 实现，但没有明确的 D1/D2 通路。
  习惯学习完全缺失——没有动作序列自动化的机制。

接口设计 (预留):
  class Striatum:
      def d1_direct_pathway(motor_cortex_input, dopamine) -> go_signal
      def d2_indirect_pathway(motor_cortex_input, dopamine) -> nogo_signal
      def habit_learn(action_sequence, reward) -> habit_strength
      def incentive_salience(stimulus, dopamine_state) -> wanting_signal

参考:
  - Gerfen, C. R., & Surmeier, D. J. (2011). Modulation of striatal projection
    systems by dopamine.
  - Graybiel, A. M. (2008). Habits, rituals, and the evaluative brain.

TODO 清单:
  [ ] D1_D2_MSN: D1/D2 中型多棘神经元模型
  [ ] DirectPathway: 直接通路 (Go)
  [ ] IndirectPathway: 间接通路 (No-Go)
  [ ] HabitLearning: 习惯学习 (动作→自动化)
  [ ] NAc: 伏隔核 (incentive salience)
  [ ] ActionChunking: 动作序列 chunking
"""

# 占位: 纹状体将在未来版本实现
# 当前动作选择由 cerebrum.basal_ganglia.action_gating.MoEGate 实现
