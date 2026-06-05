"""
v1.py — 初级视皮层 V1 (Primary Visual Cortex / Striate Cortex)  [待实现]

对应脑区: BA17 (纹状皮层, V1)
所属层级: 大脑 → 枕叶 → V1

功能职责:
  - 边缘检测: 方向选择性 (orientation selectivity)
  - 空间频率: 不同尺度的细节
  - 双眼视差: 深度感知
  - 方向选择性柱: 皮层柱组织的功能架构
  - 简单细胞/复杂细胞: Hubel & Wiesel 经典发现

V1 是视觉皮层第一站——来自 LGN 的信号首次在这里被皮层处理。

在 NotMe 中的待实现功能:
  1. Gabor 滤波器组 (已实现于 visual_pathway.py)
  2. 方向选择性柱: 柱状组织 → 侧抑制竞争
  3. 对比度归一化: 局部亮度调整
  4. 双眼视差: 如果有立体图像输入

当前状态:
  V1 的 Gabor 编码当前在 cerebrum.occipital_lobe.visual_pathway.py 中实现。
  此文件预留为独立的 V1 模块，含方向柱和对比度归一化。

接口设计 (预留):
  class V1:
      def gabor_encode(image) -> orientation_map
      def orientation_columns(orient_map) -> column_response
      def contrast_normalize(response) -> normalized
      def simple_cells(lgn_input) -> edge_response
      def complex_cells(simple_output) -> phase_invariant

参考:
  - Hubel, D. H., & Wiesel, T. N. (1962). Receptive fields, binocular
    interaction and functional architecture in the cat's visual cortex.
  - Olshausen, B. A., & Field, D. J. (1996). Emergence of simple-cell
    receptive field properties by learning a sparse code for natural images.
"""

# 占位: 独立 V1 模块将在未来版本从 visual_pathway.py 解耦
# 当前 V1 Gabor 编码集成在 cerebrum.occipital_lobe.visual_pathway.py 中
