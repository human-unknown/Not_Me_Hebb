"""
interoception.py — 内感受通路 (Interoceptive Pathway)  [待实现]

对应脑区: 岛叶前部 (Anterior Insula, AI)
所属层级: 身体模型 → 内感受

功能职责:
  - 身体状态读出: BodyVector → 内感受表征
  - 内感受精度: 对身体状态的感知噪声/精度 (meta-awareness)
  - interoceptive inference: 预测内感受 vs 实际内感受 → 误差
  - 情绪基础: Craig 理论——情绪 = 内感受状态的高阶表征
  - 岛叶层级: 后岛叶(初级内感受) → 中岛叶(整合) → 前岛叶(意识)

在 NotMe 中的关键角色:
  按照 CLAUDE.md 的核心原则:
  "身体是基础——没有身体稳态就没有情感，没有情感就没有意识"
  "Valence/Arousal 是 F_body 的数学函数，不是手工规则"

  内感受通路是身体与情感之间的桥梁:
    BodyVector → 内感受编码 → F_body 计算 → Valence/Arousal

当前状态:
  Valence/Arousal 目前直接在 L1 compute_free_energy() 中从 F_body 计算，
  没有经过内感受编码层。缺少内感受精度/噪声的概念。

接口设计 (预留):
  class Interoception:
      def encode(body_vector) -> interoceptive_state
      def precision(interoceptive, attention) -> precision_weight
      def interoceptive_inference(predicted, actual, precision) -> pe
      def feeling(interoceptive_state) -> core_affect  # valence + arousal
      def insula_hierarchy(body) -> (posterior, mid, anterior)

参考:
  - Craig, A. D. (2009). How do you feel — now? The anterior insula and human
    awareness.
  - Seth, A. K. (2013). Interoceptive inference, emotion, and the embodied self.
  - Barrett, L. F. (2017). The theory of constructed emotion: an active
    inference account of interoception and categorization.

TODO 清单:
  [ ] InteroceptiveEncoder: 身体→内感受编码
  [ ] PrecisionWeighting: 内感受精度
  [ ] InteroceptiveInference: 内感受推理
  [ ] InsulaHierarchy: 岛叶三层级
  [ ] CoreAffect: 核心情感 (valence×arousal 二维)
"""

# 占位: 内感受通路将在未来版本实现
# 当前 Valence/Arousal 直接由 cerebrum.limbic_system.cingulate.compute_free_energy() 计算
