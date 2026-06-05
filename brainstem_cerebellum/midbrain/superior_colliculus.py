"""
superior_colliculus.py — 上丘 (Superior Colliculus)  [待实现]

对应脑区: 上丘 (视顶盖, Optic Tectum)
所属层级: 脑干 → 中脑 → 上丘

功能职责:
  浅层 (Superficial layers):
    - 视觉输入 (视网膜 + V1) —  retinotopic map
    - 运动检测、新颖性检测
  深层 (Deep layers):
    - 多感官整合 (视觉 + 听觉 + 体感) — spatiotopic map
    - 眼动控制 (saccade): 快速注视定向
    - 注意定向: 隐性注意力转移

在 NotMe 中的待实现功能:
  1. 快速定向: 新颖刺激 → 上丘 → 快速注意转移 (不经皮层)
  2. 多感官地图: 视觉+听觉+体感的空间对齐
  3. Saccade 生成: 眼跳的向量编码
  4. 防御反射:  looming stimulus → 快速躲避

当前状态:
  所有感知处理经过皮层 (L0)，没有皮层下快速通路的实现。

接口设计 (预留):
  class SuperiorColliculus:
      def novelty_detect(visual_input) -> novelty_signal
      def orient_attention(novelty_location) -> attention_shift
      def multisensory_map(visual, auditory, somato) -> spatiotopic
      def saccade_generate(target_location) -> eye_movement_command

参考:
  - Sparks, D. L. (1986). Translation of sensory signals into commands for
    control of saccadic eye movements: role of primate superior colliculus.
  - Stein, B. E., & Stanford, T. R. (2008). Multisensory integration: current
    issues from the perspective of the single neuron.

TODO 清单:
  [ ] NoveltyDetector: 新颖性检测
  [ ] OrientingResponse: 注意定向
  [ ] MultisensoryMap: 多感官空间地图
  [ ] SaccadeGenerator: 扫视生成
  [ ] DefenseReflex: 防御反射 (looming)
"""

# 占位: 上丘将在未来版本实现
# 当前无皮层下快速视觉通路
