"""
orbitofrontal.py — 眶额皮层 (Orbitofrontal Cortex, OFC)  [待实现]

对应脑区: BA11, BA12, BA13 (部分), BA14, BA47 (部分)
所属层级: 大脑 → 额叶 → 眶额皮层

功能职责:
  - 价值评估 — 为感知和行动分配主观价值
  - 奖赏预测 — 预期奖赏 vs 实际奖赏 → 预测误差
  - 决策权重 — 影响前额叶行动选择的效用计算
  - 刺激-奖赏关联 — 学习什么刺激预示奖赏/惩罚
  - 反转学习 — 当 contingency 改变时快速调整
  - 冲动控制 — 延迟满足 vs 即时奖赏

在 NotMe 中的待实现功能:
  1. 价值函数 V(s): 对感知状态的主观价值评估
  2. 奖赏预测误差 RPE: 与 VTA 多巴胺信号协同
  3. 效用调制: 影响 EFE (期望自由能) 中的 pragmatic value
  4. 反转学习: contingency 改变时的适应速度

当前状态:
  价值评估目前隐含在 L2 compute_G() 的期望自由能计算中。
  OFC 将作为独立模块，分离"价值"和"决策"的神经计算。

接口设计 (预留):
  class OrbitofrontalCortex:
      def evaluate_state(sensory_state, body_state) -> value
      def predict_reward(stimulus, context) -> expected_reward
      def compute_rpe(expected, actual) -> reward_prediction_error
      def update_value(stimulus, rpe, learn_rate) -> None
      def modulate_utility(efe_pragmatic, value) -> modulated_utility

参考:
  - Rolls, E. T. (2000). The orbitofrontal cortex and reward.
  - Schoenbaum, G. et al. (2009). A new perspective on the role of the
    orbitofrontal cortex in adaptive behavior.

TODO 清单:
  [ ] ValueFunction: 状态→主观价值映射
  [ ] RewardPredictor: 奖赏预测模型
  [ ] RPE: 奖赏预测误差计算
  [ ] ReversalLearning: 反转学习机制
  [ ] DelayDiscounting: 延迟折扣模型
"""

# 占位: 眶额皮层将在未来版本实现
# 当前价值评估隐含在 cerebrum.frontal_lobe.prefrontal.compute_G() 中
