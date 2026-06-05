"""
autonomic.py — 自主神经调控 (Autonomic Regulation)  [待实现]

对应脑区: 延髓自主神经核团 (孤束核 NTS + 疑核 NA + 背侧运动核 DMV)
所属层级: 脑干 → 延髓 → 自主神经

功能职责:
  交感神经 (Sympathetic):
    - 战斗或逃跑 (fight-or-flight)
    - 心率↑、血压↑、呼吸↑、出汗
    - 瞳孔扩大、消化抑制
  副交感神经 (Parasympathetic):
    - 休息与消化 (rest-and-digest)
    - 心率↓、消化↑、唾液分泌
    - 瞳孔缩小

  自主神经平衡:
    交感/副交感不是 on/off —— 是连续的动态平衡
    心率变异性 (HRV) = 副交感弹性指标
    皮肤电导 (SCR) = 交感激活指标

在 NotMe 中的待实现功能:
  1. 自主神经状态: BodyVector 中的自主神经张力
  2. 应激反应: 威胁 → 交感激活 → 心率↑ + 消化↓
  3. 放松反应: 安全 → 副交感激活 → 心率↓ + 恢复↑
  4. 身体输出: 自主神经 → 可观测的身体信号 (对话中的"表情")

当前状态:
  自主神经功能完全缺失。BodyVector 的维度 (如疲劳、压力) 没有
  自主神经的神经基础。心率、呼吸等生理变量未建模。

接口设计 (预留):
  class AutonomicNervousSystem:
      def sympathetic_activate(threat_level) -> symp_response
      def parasympathetic_activate(safety_level) -> parasymp_response
      def balance(symp, parasymp) -> autonomic_state
      def body_output(autonomic_state) -> observable_signals

参考:
  - Porges, S. W. (2001). The polyvagal theory: phylogenetic substrates of a
    social nervous system.
  - Thayer, J. F., & Lane, R. D. (2000). A model of neurovisceral integration
    in emotion regulation and dysregulation.

TODO 清单:
  [ ] SympatheticModel: 交感神经模型
  [ ] ParasympatheticModel: 副交感神经模型
  [ ] HRV: 心率变异性
  [ ] SCR: 皮肤电导
  [ ] PolyvagalTheory: 多层迷走神经理论 (社会参与系统)
"""

# 占位: 自主神经调控将在未来版本实现
# 当前 BodyVector 的漂移是纯数学的，没有神经生理基础
