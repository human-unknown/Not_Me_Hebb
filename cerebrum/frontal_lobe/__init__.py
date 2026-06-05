"""
额叶 (Frontal Lobe)  [Level 3]

功能：执行功能 · 决策判断 · 计划组织 · 语言产出 · 运动规划

布罗德曼分区备注：
  BA4      初级运动皮层 — 执行随意运动
  BA6      前运动皮层 + 辅助运动区 — 运动规划/序列
  BA8      额叶眼区 — 眼动控制
  BA9/46   背外侧前额叶 (dlPFC) — 工作记忆、执行功能
  BA10     额极 — 认知分支、 multitasking
  BA11     眶额皮层 (OFC) — 价值评估、奖赏处理
  BA44/45  布罗卡区 — 语言产出 (语法、语音)
  BA47     额下回眶部 — 语义处理

子模块:
├── prefrontal.py     前额叶皮层 — EFE 行动选择、递归多层次 G (L2)
├── broca.py          BA44/45 布罗卡区 — Hebb 词序链语言生成
├── motor_cortex.py   BA4/6 运动皮层 — 运动规划与执行 [待实现]
└── orbitofrontal.py  BA11 眶额皮层 — 价值评估与奖赏预测 [待实现]
"""

# 前额叶皮层 —— 主动推理核心
from cerebrum.frontal_lobe.prefrontal import (
    compute_G, select_action, predict_next_state,
    update_social_beliefs,
)

# 布罗卡区 —— 语言产出
from cerebrum.frontal_lobe.broca import Broca

__all__ = [
    'compute_G', 'select_action', 'predict_next_state',
    'update_social_beliefs',
    'Broca',
]
