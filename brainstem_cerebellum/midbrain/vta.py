"""
vta.py — 腹侧被盖区 (Ventral Tegmental Area)  [待实现]

对应脑区: VTA (腹侧被盖区, A10 细胞群)
所属层级: 脑干 → 中脑 → VTA

功能职责:
  - 多巴胺释放 → 伏隔核(NAc)、前额叶(PFC)、杏仁核
  - 奖赏预测误差 (RPE): δ = R_actual - R_predicted
  - incentive salience (wanting): 将中性刺激转化为"想要"的目标
  - 动机驱动: 调节行为激活水平
  - 社会奖赏: 社交互动也触发 VTA 多巴胺

在 NotMe 中的待实现功能:
  1. RPE 计算: 与 OFC 的 value 预测配合 → δ = V(s') - V(s)
  2. 多巴胺调制: RPE → DA 释放 → 调节 L0 学习率 + L2 行动选择
  3. 动机系统: DA 水平 → 行为激活/探索倾向
  4. 社会奖赏: 正效价社交互动 → DA → 增强社交倾向

当前状态:
  奖赏/动机系统目前完全缺失。
  F_body 驱动效价，但没有专门的奖赏预测误差通路。
  学习率目前是固定的 (learn_rate_l0)，缺乏事件驱动的自适应调制。

接口设计 (预留):
  class VTA:
      def compute_rpe(predicted_reward, actual_outcome) -> dopamine_signal
      def modulate_learning(dopamine, base_lr) -> effective_lr
      def modulate_motivation(dopamine, arousal) -> behavioral_activation
      def social_reward(social_valence, trust_level) -> social_dopamine

参考:
  - Schultz, W., Dayan, P., & Montague, P. R. (1997). A neural substrate of
    prediction and reward.
  - Berridge, K. C., & Robinson, T. E. (2003). Parsing reward.

TODO 清单:
  [ ] RPEModel: 奖赏预测误差 (δ = R - R_pred)
  [ ] DopamineRelease: 时相性/紧张性多巴胺
  [ ] LearningModulation: RPE → 学习率调制
  [ ] IncentiveSalience: "wanting" (非 liking)
  [ ] SocialReward: 社会奖赏通路
"""

# 占位: VTA 将在未来版本实现
# 当前学习率由 Theta 参数 (meta_learning.py) 全局调控
