"""
somatosensory.py — 体感皮层 (Somatosensory Cortex)  [待实现]

对应脑区: BA3, BA1, BA2 (初级体感皮层 S1) + BA5, BA7 (后顶叶)
所属层级: 大脑 → 顶叶 → 体感皮层

功能职责:
  BA3a  肌梭输入 (本体感觉)
  BA3b  皮肤慢适应触觉 (纹理、形状)
  BA1   皮肤快适应触觉 (振动、滑动)
  BA2   深部组织 (关节位置、压力)
  BA5/7 多感官整合 (视觉+体感)、身体图式(body schema)

在 NotMe 中的待实现功能:
  1. 触觉编码: 环境反馈中的触觉成分 → 体感向量
  2. 本体感觉: BodyVector 的身体状态读出 → 身体图式
  3. 身体图式: 将身体状态映射到空间参考系
  4. 疼痛编码: 组织损伤信号 → 岛叶/ACC 传递

当前状态:
  身体状态目前由 BodyVector 直接维护，没有独立的体感编码层。
  体感皮层将作为 BodyVector 和环境反馈之间的编码/解码接口。

接口设计 (预留):
  class SomatosensoryCortex:
      def encode_touch(contact_info) -> touch_vector
      def encode_proprioception(body_state, action) -> proprio_vector
      def body_schema(body_vector) -> spatial_body_map
      def multisensory_integration(touch, vision, proprio) -> unified

参考:
  - Kaas, J. H. (2004). Evolution of somatosensory and motor cortex in primates.
  - Maravita, A., & Iriki, A. (2004). Tools for the body (schema).

TODO 清单:
  [ ] TouchEncoder: 触觉向量编码
  [ ] Proprioception: 本体感觉读出
  [ ] BodySchema: 身体图式 (空间参考系)
  [ ] PainPathway: 疼痛编码通路
  [ ] MultisensoryIntegration: 多感官整合 (VIP, AIP)
"""

# 占位: 体感皮层将在未来版本实现
# 当前身体状态由 cns.data_types.BodyVector 直接维护
