"""
tpj.py — 颞顶联合区 (Temporoparietal Junction)  [待实现]

对应脑区: BA39 (角回) + BA40 (缘上回) + 颞上沟后部(pSTS)
所属层级: 大脑 → 顶叶 → TPJ

功能职责:
  - 心理理论 (Theory of Mind) — 推断他人信念/意图/情绪
  - 自我/他人区分 — 区分自己的和别人的心理状态
  - 视角采择 (Perspective Taking) — 从他人视角看世界
  - 道德判断 — 意图评估、道德推理
  - 社会注意力 — 共同注意(joint attention)、注视线索(gaze cueing)

在 NotMe 中的待实现功能:
  1. 二阶信念: "我认为你认为..." → 用于多智能体博弈
  2. 心理理论: 对其他 Agent 的内部状态建模
  3. 视角采择: 从其他 Agent 的视角评估同一场景
  4. 社会推理: 意图归因、信任更新

当前状态:
  社会信念更新目前由 L2 update_social_beliefs() 实现，仅为一阶信念。
  TPJ 将引入二阶及以上的递归社会推理。

接口设计 (预留):
  class TPJ:
      def infer_others_belief(observation, self_model) -> belief_model
      def perspective_take(scene, other_agent_state) -> other_view
      def second_order_belief(my_belief, other_model) -> "I think you think..."
      def moral_evaluation(intention, outcome) -> moral_judgment

参考:
  - Saxe, R., & Kanwisher, N. (2003). People thinking about thinking people:
    The role of the temporo-parietal junction in "theory of mind".
  - Frith, C. D., & Frith, U. (2006). The neural basis of mentalizing.

TODO 清单:
  [ ] TheoryOfMind: 心理理论 (1st/2nd order)
  [ ] SelfOtherDistinction: 自我/他人区分
  [ ] PerspectiveTaking: 视角采择
  [ ] MoralJudgment: 道德判断
  [ ] JointAttention: 共同注意力
"""

# 占位: TPJ 将在未来版本实现
# 当前社会推理由 cerebrum.frontal_lobe.prefrontal.update_social_beliefs() 实现
