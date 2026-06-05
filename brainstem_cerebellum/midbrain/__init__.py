"""
中脑 (Midbrain / Mesencephalon)  [Level 3]

功能：视觉/听觉反射整合 · 多巴胺奖赏通路 · 运动调节

核心核团:
├── 上丘 (Superior Colliculus)    — 视觉反射、眼动控制、多感官整合
├── 下丘 (Inferior Colliculus)    — v5.2: 听觉中脑整合, 频率×空间×时间
├── 黑质致密部 (SNc)              — 多巴胺供给基底节 (运动调节)
├── 黑质网状部 (SNr)              — 基底节输出核 (动作门控)
├── 腹侧被盖区 (VTA)              — 多巴胺奖赏预测误差、动机
├── 导水管周围灰质 (PAG)          — 疼痛调节、防御行为
└── 红核 (Red Nucleus)            — 运动协调

子模块:
├── vta.py                    VTA — 多巴胺奖赏通路、RPE、学习率调制 ★v5.5
├── substantia_nigra.py       黑质 — SNc 多巴胺供给、SNr 输出门控 [待实现]
├── superior_colliculus.py    上丘 — 视觉反射、快速定向 [待实现]
└── inferor_colliculus.py     下丘 — 听觉整合 + 新颖性检测 ★v5.2
"""
