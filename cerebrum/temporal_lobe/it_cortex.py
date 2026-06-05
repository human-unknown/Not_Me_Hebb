"""
it_cortex.py — 下颞皮层 (Inferotemporal Cortex, IT)  [待实现]

对应脑区: BA20, BA21, BA37 (部分)
所属层级: 大脑 → 颞叶 → IT 皮层

功能职责:
  - 物体识别 — 腹侧视觉通路终点
  - 形状编码 — 从 V4 的局部特征到整体物体表征
  - 视角不变性 — 识别不同角度/光照/大小的同一物体
  - 语义关联 — 连接视觉特征与语义概念
  - 面孔/场景 patch — 特定类别的高度选择性神经元

在 NotMe 中的待实现功能:
  1. 物体识别: 视觉特征 → "这是什么物体"
  2. 视角不变编码: Gabor 特征 → 物体身份 (不变表示)
  3. 语义桥接: 视觉物体 → 文本语义 (跨模态对齐)
  4. 类别学习: 无监督物体类别形成

当前状态:
  物体"识别"目前由 stage2_crossmodal.py 的跨模态检索实现。
  IT 皮层将作为视觉通路 (V1→V2→V4→IT) 的最终阶段，输出物体级表示。

接口设计 (预留):
  class ITCortex:
      def object_encode(visual_features) -> object_vector
      def invariant_recognize(object_vector, viewpoint) -> identity
      def category_form(object_vectors) -> categories
      def semantic_link(object_vector) -> semantic_embedding

参考:
  - Logothetis, N. K., & Sheinberg, D. L. (1996). Visual object recognition.
  - Tanaka, K. (1996). Inferotemporal cortex and object vision.
  - DiCarlo, J. J., Zoccolan, D., & Rust, N. C. (2012). How does the brain
    solve visual object recognition?

TODO 清单:
  [ ] ObjectEncoder: 物体编码 (pooled V4 features)
  [ ] ViewInvariance: 视角不变性
  [ ] CategoryLearning: 无监督类别学习
  [ ] SemanticGrounding: 视觉→语义 grounding
  [ ] FacePatches: 面孔选择性区域
"""

# 占位: IT 皮层将在未来版本实现
# 当前物体识别由 cerebrum.association.crossmodal 的跨模态检索实现
