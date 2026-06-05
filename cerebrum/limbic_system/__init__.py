"""
边缘系统 (Limbic System)  [Level 3]

功能：记忆编码/巩固/提取 · 情绪评估 · 稳态调节 · 感觉中继

核心结构:
├── 海马 (Hippocampus)    — 情景记忆编码、模式补全/分离、睡眠巩固
├── 杏仁核 (Amygdala)      — 恐惧/情绪评估、情感学习
├── 下丘脑 (Hypothalamus)  — 饥饿/体温/内分泌/昼夜节律
├── 丘脑 (Thalamus)        — 感觉中继站/门控、意识与注意力调节
├── 扣带回 (Cingulate)     — ACC: 冲突监测/共情/疼痛; PCC: DMN节点
├── 嗅皮层 (Olfactory)     — 嗅觉处理、气味记忆
├── 隔区 (Septum)          — 奖赏中枢
└── 乳头体 (Mammillary)    — 情景记忆回路中转站

布罗德曼分区备注：
  BA23-24  扣带回 (ACC 为 BA24/25/32/33)
  BA25     膝下扣带 — 情绪调节
  BA26-30  压后皮层 — 空间导航/情景记忆
  BA34-36  内嗅皮层/旁海马 — 空间导航/记忆
  BA13-16  岛叶 — 内感受/自我意识 (注: 岛叶常被归入旁边缘系统)

子模块:
├── hippocampus.py    海马 — Hebb 集群记忆、模式补全、睡眠回放巩固 (L0)
├── amygdala.py       杏仁核 — Hebb 情感词汇网络、从 F_body 学习词的情感效应
├── cingulate.py      扣带回 — 自由能计算、效价/唤醒、习惯化、社会上下文 (L1)
├── hypothalamus.py   下丘脑 — 身体稳态调节 (BodyVector ODE) [待实现]
├── thalamus.py       丘脑 — 感觉中继/门控、注意力调制 [待实现]
└── olfactory.py      嗅皮层 — 嗅觉编码 [待实现]
"""

# 海马 —— 记忆核心
from cerebrum.limbic_system.hippocampus import (
    predict_sensations, ClusterNetwork, sleep_cycle,
    _masked_cosine, _auto_mask,
)

# 杏仁核 —— 情感学习
from cerebrum.limbic_system.amygdala import (
    analyze_sentiment, sentiment_to_social_signal,
    get_emotional_lexicon,
)

# 扣带回 —— 自由能计算
from cerebrum.limbic_system.cingulate import (
    compute_free_energy, HabituationTracker,
    SocialContext, compute_F_body,
)

__all__ = [
    # Hippocampus
    'predict_sensations', 'ClusterNetwork', 'sleep_cycle',
    '_masked_cosine', '_auto_mask',
    # Amygdala
    'analyze_sentiment', 'sentiment_to_social_signal',
    'get_emotional_lexicon',
    # Cingulate
    'compute_free_energy', 'HabituationTracker',
    'SocialContext', 'compute_F_body',
]
