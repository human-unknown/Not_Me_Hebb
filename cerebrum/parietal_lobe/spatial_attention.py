"""
spatial_attention.py — 空间注意力网络 (Spatial Attention)  [待实现]

对应脑区: 顶内沟(IPS)、上顶叶(SPL)、额叶眼区(FEF)
所属层级: 大脑 → 顶叶 → 空间注意力

功能职责:
  - 空间注意定向 (overt + covert attention)
  - 显著图 (saliency map) — 整合自下而上和自上而下注意
  - 特征注意 — 增强特定特征的神经元响应
  - 空间工作记忆 — 保持空间位置信息
  - 眼动规划 — 与 FEF/上丘协作

在 NotMe 中的待实现功能:
  1. 注意力探照灯: 增强环境中相关区域/特征的信号
  2. 显著图: 计算感知输入各维度的显著度
  3. 注意力调制: 调制 L0 感知编码的 gain
  4. 空间导航: GridWorld 的空间位置编码和路径规划

当前状态:
  注意力目前仅通过 L1 的习惯化追踪 (HabituationTracker) 实现。
  空间注意力网络将显著增强 Agent 对环境的感知选择能力。

接口设计 (预留):
  class SpatialAttention:
      def compute_saliency(sensory_input, task_goal) -> saliency_map
      def attend(sensory, saliency_map, focus) -> attended_sensory
      def spatial_working_memory(locations) -> spatial_buffer
      def orient_attention(current_focus, saliency) -> new_focus

参考:
  - Itti, L., & Koch, C. (2001). Computational modelling of visual attention.
  - Corbetta, M., & Shulman, G. L. (2002). Control of goal-directed and
    stimulus-driven attention in the brain.

TODO 清单:
  [ ] SaliencyMap: 显著图计算
  [ ] FeatureAttention: 特征注意力
  [ ] SpatialAttention: 空间注意定向
  [ ] Neglect: 半侧忽略模拟 (右顶叶损伤)
  [ ] DorsalStream: 背侧视觉通路 (where/how)
"""

# 占位: 空间注意力网络将在未来版本实现
# 当前注意调制由 cerebrum.limbic_system.cingulate.HabituationTracker 提供
