"""
基底神经节 (Basal Ganglia)  [Level 3]

功能：习惯动作 · 程序性记忆 · 动作选择/抑制 · 奖赏强化(多巴胺)

核心结构:
├── 纹状体 (Striatum)       — 尾状核 + 壳核：习惯学习、奖赏预测
│   ├── 背侧纹状体 (Dorsal)  — 感觉运动/联合
│   └── 腹侧纹状体 (Ventral) — 奖赏/动机 (含伏隔核 NAc)
├── 苍白球 (Globus Pallidus) — 动作门控输出
│   ├── 外侧苍白球 (GPe)     — 间接通路
│   └── 内侧苍白球 (GPi)     — 直接通路输出
├── 底丘脑核 (STN)           — 动作抑制/冲动控制
└── 黑质 (Substantia Nigra)  — (细胞体位于中脑，功能属基底节回路)
    ├── 黑质致密部 (SNc)     — 多巴胺供给
    └── 黑质网状部 (SNr)     — 输出核

经典通路:
  直接通路: 皮层 → 纹状体 → GPi/SNr (抑制) → 丘脑(去抑制) → 皮层 (促进动作)
  间接通路: 皮层 → 纹状体 → GPe → STN → GPi/SNr (增强抑制) → 丘脑(抑制) → 皮层 (抑制动作)
  超直接通路: 皮层 → STN → GPi/SNr (快速动作取消)

子模块:
├── action_gating.py   MoE 门控 — 疲劳预算轮替、动作选择/抑制 (L2.5)
├── striatum.py        纹状体 — 习惯学习、直接/间接通路 [待实现]
├── pallidum.py        苍白球 — 动作门控输出 [待实现]
└── subthalamic.py     底丘脑核 — 动作抑制、冲动控制 [待实现]
"""

# 动作门控
from cerebrum.basal_ganglia.action_gating import MoEGate

__all__ = ['MoEGate']
