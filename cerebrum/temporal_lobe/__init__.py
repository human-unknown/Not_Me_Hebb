"""
颞叶 (Temporal Lobe)  [Level 3]

功能：听觉处理 · 语言理解 · 情景记忆编码 · 物体/面孔识别 · 情绪判断

布罗德曼分区备注：
  BA20      颞下回 — 视觉物体识别
  BA21      颞中回 — 语义记忆
  BA22      韦尼克区 — 语言理解 (后部)
  BA37      梭状回 — 面孔/文字/物体识别
  BA38      颞极 — 社会-情绪处理
  BA41/42   初级听皮层 (Heschl's gyrus) — 听觉输入
  BA52      副听区

子模块:
├── wernicke.py         BA22 韦尼克区 — 语言理解回路、对话记忆 (L2.5)
├── auditory_cortex.py  BA41/42 听皮层 — 听觉频谱编码 [待实现]
├── it_cortex.py        IT 皮层 — 物体识别 (形状/颜色/语义) [待实现]
└── fusiform.py         梭状回 — 面孔识别 (FFA) [待实现]
"""

# 韦尼克区 —— 语言理解
from cerebrum.temporal_lobe.wernicke import (
    DialogueContext, comprehend, evaluate_response,
    consolidate_dialogue_memory, micro_consolidation,
)

__all__ = [
    'DialogueContext', 'comprehend', 'evaluate_response',
    'consolidate_dialogue_memory', 'micro_consolidation',
]
