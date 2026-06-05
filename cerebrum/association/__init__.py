"""
联合皮层 + 默认模式网络 (Association Cortex + DMN)  [Level 3]

功能：自我参照思维 · 跨模态整合 · 情景未来模拟 · 社会认知

核心结构:
├── 默认模式网络 (DMN)      — 自我参照、走神、自传体记忆、情景模拟
│   ├── vmPFC (腹内侧前额叶) — 自我参照评估
│   ├── PCC (后扣带回)      — DMN 核心节点
│   ├── 角回 (Angular Gyrus) — 语义整合、跨模态
│   └── 内侧颞叶 (MTL)       — 情景记忆
├── 突显网络 (Salience)      — 检测显著刺激，切换 DMN↔TPN
│   ├── 前岛叶 (AI)          — 内感受意识
│   └── 背侧前扣带 (dACC)    — 冲突/错误监测
└── 中央执行网络 (CEN/FPN)   — 工作记忆、任务执行
    ├── dlPFC (背外侧前额叶)  — 执行控制
    └── PPC (后顶叶皮层)      — 注意力定向

关键动态:
  DMN ↔ TPN 跷跷板: DMN (走神) 与 TPN (任务) 互相抑制，由突显网络切换

子模块:
├── dmn.py         默认模式网络 — 自我模型、"我是谁"的 Hebb 表征 (SelfModel)
└── crossmodal.py  跨模态整合 — COCO Visual↔Text Hebb 关联学习 (Stage 2)
"""

# 默认模式网络
from cerebrum.association.dmn import SelfModel

# 跨模态整合
# (stage2_crossmodal.py 作为独立脚本运行，不在此导入)

__all__ = ['SelfModel']
