"""
substantia_nigra.py — 黑质 (Substantia Nigra)  [待实现]

对应脑区: 黑质致密部 (SNc) + 黑质网状部 (SNr)
所属层级: 脑干 → 中脑 → 黑质

功能职责:
  SNc (致密部, A9 细胞群):
    - 多巴胺供给背侧纹状体 (nigrostriatal pathway)
    - 运动启动和 vigor (运动力度)
    - 习惯学习的 DA 信号
    - 退行性变 → 帕金森病 (运动迟缓、僵硬)
  SNr (网状部):
    - 基底节输出核 (与 GPi 并列)
    - GABA 抑制 → 上丘/丘脑
    - 眼动控制 (saccade 门控)

在 NotMe 中的待实现功能:
  1. SNc DA 供给: 背侧纹状体的多巴胺调控 (习惯学习)
  2. 运动 vigor: DA 水平 → 运动速度/力度
  3. SNr 输出: 动作门控的第二个出口 (与 GPi 并行)
  4. 黑质-纹状体退化: 模拟帕金森症状 (低 DA → 运动减少)

当前状态:
  黑质-纹状体通路完全缺失。没有运动 vigor 的概念。

接口设计 (预留):
  class SubstantiaNigra:
      # SNc
      def dopamine_supply(striatal_needs) -> nigrostriatal_da
      def vigor_modulation(da_level) -> movement_vigor
      # SNr
      def snr_output(direct_pathway_input) -> thalamic_collicular_inhibition

参考:
  - Schultz, W. (1998). Predictive reward signal of dopamine neurons.
  - Haber, S. N. (2003). The primate basal ganglia: parallel and integrative
    networks.

TODO 清单:
  [ ] NigrostriatalDA: 黑质-纹状体多巴胺通路
  [ ] VigorModel: 运动 vigor 调制
  [ ] SnrOutput: SNr 输出门控
  [ ] ParkinsonModel: 帕金森病理模拟
"""

# 占位: 黑质将在未来版本实现
# 当前无多巴胺通路实现
