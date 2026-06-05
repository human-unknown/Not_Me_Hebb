"""
subthalamic.py — 底丘脑核 (Subthalamic Nucleus, STN)  [待实现]

对应脑区: 底丘脑核 (丘脑底部)
所属层级: 大脑 → 基底神经节 → STN

功能职责:
  - 超直接通路 — 皮层→STN→GPi (最快通路，全局动作抑制)
  - 动作取消 — 当需要突然停止正在进行的动作
  - 冲突检测 — 多个竞争动作 → STN 激活 → 暂停+重新评估
  - 冲动控制 — 延缓响应以允许更多皮层处理
  - 与 GPe 构成中央起搏器 — beta 节律 (13-30 Hz) 与运动障碍相关

超直接通路: 皮层 → STN → GPi (全抑制，~5ms)
  - 被比喻为"紧急刹车"——在运动的最后时刻取消

在 NotMe 中的待实现功能:
  1. 动作取消: 当新的紧急事件出现时，中断当前动作
  2. 冲突检测: 多个动作竞争时 → 延迟决策，收集更多信息
  3. 冲动控制: 高 STN 激活 → 高决策阈值 (更谨慎)

当前状态:
  动作取消和冲突检测完全缺失。
  MoEGate 的疲劳机制是唯一接近"动作抑制"的机制。

接口设计 (预留):
  class SubthalamicNucleus:
      def hyperdirect_pathway(cortical_command) -> global_inhibition
      def conflict_detect(competing_actions) -> conflict_signal
      def impulse_control(decision_urgency) -> delayed_response
      def action_cancel(ongoing_action, new_priority) -> cancel_signal

参考:
  - Nambu, A., Tokuno, H., & Takada, M. (2002). Functional significance of
    the cortico-subthalamo-pallidal 'hyperdirect' pathway.
  - Frank, M. J. (2006). Hold your horses: A dynamic computational role for
    the subthalamic nucleus in decision making.

TODO 清单:
  [ ] HyperdirectPathway: 超直接通路
  [ ] ConflictDetection: 冲突检测
  [ ] ActionCancel: 动作取消 (紧急刹车)
  [ ] DecisionThreshold: 决策阈值调制
  [ ] BetaOscillation: GPe-STN beta 振荡
"""

# 占位: 底丘脑核将在未来版本实现
# 当前动作抑制仅由 MoEGate 的疲劳预算轮替机制实现
