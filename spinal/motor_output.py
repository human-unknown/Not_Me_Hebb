"""
motor_output.py — 运动输出 (Motor Output / Spinal Cord)  [待实现]

对应脑区: 脊髓运动神经元 + 中枢模式发生器 (CPG)
所属层级: CNS → 脊髓 → 运动输出

功能职责:
  - 运动指令执行: 皮层/脑干运动指令 → 脊髓 → 肌肉激活
  - 中枢模式发生器 (CPG): 节奏性运动 (行走、呼吸、咀嚼)
  - 牵张反射: 肌梭→脊髓→运动神经元 (最快反射, ~30ms)
  - 屈曲反射: 疼痛→快速回缩
  - 低级协调: 不经过脑的简单运动模式

在 NotMe 中的待实现功能:
  1. 运动输出抽象: action → 环境中的具体效果
  2. 反射弧: 紧急情况 → 不经皮层 → 快速动作
  3. 动作平滑: 离散 action → 连续执行
  4. 本体感觉反馈: 肌肉状态 → 脊髓 → 上传

当前状态:
  Action 直接从 L2 select_action() → 环境执行，没有运动输出层。

接口设计 (预留):
  class MotorOutput:
      def execute(action, body_state) -> motor_commands
      def cpg_rhythm(gait_pattern, speed) -> rhythmic_output
      def stretch_reflex(muscle_stretch) -> reflex_contraction
      def withdrawal_reflex(painful_stimulus) -> rapid_withdrawal

参考:
  - Grillner, S. (2003). The motor infrastructure: from ion channels to
    neuronal networks.
  - Bizzi, E., et al. (2008). Combining modules for movement.

TODO 清单:
  [ ] MotorNeuronPool: 运动神经元池 (肌群激活)
  [ ] CPG: 中枢模式发生器
  [ ] StretchReflex: 牵张反射
  [ ] WithdrawalReflex: 屈曲反射
  [ ] MotorSmoothing: 动作平滑
"""

# 占位: 脊髓运动输出将在未来版本实现
# 当前 action 直接作用于环境，无中间输出层
