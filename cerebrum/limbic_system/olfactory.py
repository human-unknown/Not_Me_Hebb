"""
olfactory.py — 嗅皮层 (Olfactory Cortex)  [待实现]

对应脑区: 嗅球 + 前嗅核 + 梨状皮层 + 内嗅皮层 (BA34 部分)
所属层级: 大脑 → 边缘系统 → 嗅皮层

功能职责:
  - 气味检测 — 嗅上皮 → 嗅球 → 梨状皮层
  - 气味识别 — 模式分离 (梨状皮层)
  - 气味-记忆关联 — 嗅皮层→海马的直接投射 (普鲁斯特效应)
  - 气味-情绪关联 — 嗅皮层→杏仁核
  - 信息素处理 — 犁鼻器→副嗅球→杏仁核 (社会化学信号)

特殊地位:
  嗅觉是唯一不经过丘脑中继的感觉——
  嗅球直接投射到梨状皮层和杏仁核。
  这也是为什么气味能如此强烈地触发记忆和情绪。

在 NotMe 中的待实现功能:
  1. 嗅觉编码: 气味特征 → 嗅觉向量 (作为感知维度扩展)
  2. 气味-记忆桥接: 嗅输入 → 海马自动检索 (Proust effect)
  3. 情感条件: 气味+情感事件 → 条件性情感反应

当前状态:
  嗅觉通道完全缺失。相比视觉和听觉，嗅觉在文本对话场景中优先级较低。

接口设计 (预留):
  class OlfactoryCortex:
      def encode_odor(chemical_features) -> odor_vector
      def odor_memory(odor_vector) -> associated_memories
      def odor_emotion(odor_vector) -> emotional_response

参考:
  - Wilson, D. A., & Sullivan, R. M. (2011). Cortical processing of odor objects.
  - Gottfried, J. A. (2010). Central mechanisms of odour object perception.

TODO 清单:
  [ ] OlfactoryBulb: 嗅球编码
  [ ] PiriformCortex: 梨状皮层 (气味识别)
  [ ] OdorMemory: 气味-记忆关联
  [ ] PheromonePathway: 信息素通路
"""

# 占位: 嗅皮层将在未来版本实现 (低优先级)
