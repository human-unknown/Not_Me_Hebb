"""
神经调节系统 (Neuromodulatory Systems)  [Level 3]

功能：全局学习率调节 · 可塑性门控 · 发育关键期 · 神经递质模拟

四大 diffuse 投射系统:
├── 多巴胺 (DA)    VTA/SNc → 纹状体/前额叶  — 奖赏预测误差、动机、运动
├── 去甲肾上腺素 (NE)  蓝斑核 → 全脑      — 唤醒度、注意力、应激
├── 血清素 (5-HT)      中缝核 → 全脑      — 情绪、睡眠、食欲
├── 乙酰胆碱 (ACh)    基底前脑 → 皮层/海马 — 学习、记忆、注意力

在 NotMe 中映射到:
  MetaLearner — 有限差分梯度下降在线调整 Theta 参数
  GluN2B→GluN2A 发育轨迹 (v6.1)
  PlasticityRegulator — 整合可塑性调制 (v6.1)
  创伤模拟 (apply_trauma) — 永久修改社会参数

子模块:
├── meta_learning.py  元学习 — 有限差分梯度下降 (L3 MetaLearner)
└── plasticity.py     ★ v6.1 可塑性调节 — GluN2B + 事件驱动 + 稳态 + 神经调质门控
"""

# 元学习
from brainstem_cerebellum.neuromodulatory.meta_learning import (
    create_default_theta, MetaLearner,
)

# v6.1: 可塑性调节
from brainstem_cerebellum.neuromodulatory.plasticity import (
    PlasticityRegulator,
)

__all__ = ['create_default_theta', 'MetaLearner', 'PlasticityRegulator']
