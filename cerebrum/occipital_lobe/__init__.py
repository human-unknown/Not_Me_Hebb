"""
枕叶 (Occipital Lobe)  [Level 3]

功能：视觉处理 — 从边缘检测到物体识别

布罗德曼分区备注：
  BA17      初级视皮层 (V1) — 边缘、方向、空间频率
  BA18      次级视皮层 (V2) — 轮廓整合、深度
  BA19      第三视皮层 (V3/V4/V5) — 颜色、运动、形状
  (注: V1→V2→V4→IT 为腹侧通路"what"; V1→V2→V5/MT→顶叶 为背侧通路"where")

视觉通路 (6 条子通路):
  V1 → V2 → V4 → IT  (腹侧通路: 物体识别)
  V1 → V2 → MT → 顶叶 (背侧通路: 空间位置/运动)
  Pulvinar 捷径        (低空间频率快速通路)
  Color Opponent        (色拮抗: 红绿 + 蓝黄)

子模块:
├── visual_pathway.py   视觉编码器整合 — Gabor V1+V2+V4+Color+Pulvinar+Dorsal
├── v1.py               BA17 初级视皮层 — Gabor 滤波器组 [待实现]
├── v2.py               V2 — 轮廓整合、方向交互 [待实现]
├── v4.py               V4 — 全局形状、曲率、颜色 [待实现]
├── retina_lgn.py       视网膜→LGN 编码 — 图像 Gabor 多尺度编码
└── gestalt.py          Gestalt 知觉分组 — 邻近/共线/相似/对称/图底分离 (v4.1)
"""

# 视觉通路
from cerebrum.occipital_lobe.visual_pathway import (
    GaborFilterBank,
)
from cerebrum.occipital_lobe.retina_lgn import (
    ImageEncoder, build_visual_sensory,
)
from cerebrum.occipital_lobe.gestalt import (
    GestaltGrouping, compute_gestalt_from_image,
)
from cerebrum.occipital_lobe.v1 import V1
from cerebrum.occipital_lobe.v2 import V2
from cerebrum.occipital_lobe.v4 import V4

__all__ = [
    'GaborFilterBank',
    'ImageEncoder', 'build_visual_sensory',
    'GestaltGrouping', 'compute_gestalt_from_image',
    'V1', 'V2', 'V4',
]
