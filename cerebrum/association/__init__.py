"""
联合皮层 + 三大核心网络 (Association Cortex + Triple Networks)  [Level 3]

功能：自我参照思维 · 跨模态整合 · 选择性注意 · 任务执行 · 情景未来模拟 · 社会认知

图3 规则4 (注意力瓶颈) 的核心结构 — 三大网络跷跷板:

├── 默认模式网络 (DMN)      — 自我参照、走神、自传体记忆、情景模拟 (图3 规则4)
│   ├── vmPFC (腹内侧前额叶) — 自我参照评估
│   ├── PCC (后扣带回)      — DMN 核心节点
│   ├── 角回 (Angular Gyrus) — 语义整合、跨模态
│   └── 内侧颞叶 (MTL)       — 情景记忆
│
├── 额顶网络 (FPN / CEN)    — 选择性注意"探照灯"、工作记忆、认知控制 (图3 规则4)
│   ├── dlPFC (背外侧前额叶 BA9/46) — 执行控制、工作记忆
│   ├── PPC (后顶叶皮层 BA7/40)     — 空间注意定向
│   └── FEF (额叶眼区 BA8)          — 注意转移
│
├── 任务正网络 (TPN)        — 任务执行、DMN 跷跷板对立面 (图3 规则4)
│   ├── dlPFC · dACC · PPC · AI · pre-SMA — 跨区域任务激活模式
│   └── 工作机制: TPN ↑ ⟹ DMN ↓ (任务中); DMN ↑ ⟹ TPN ↓ (走神)
│
├── 突显网络 (Salience)      — 检测显著刺激，切换 DMN↔TPN
│   ├── 前岛叶 (AI)          — 内感受意识
│   └── 背侧前扣带 (dACC)    — 冲突/错误监测
│
└── 跨模态联合皮层           — 多模态 Hebb 整合 (图3 规则3: 并行分布式)

关键动态 (v4.3 图3 规则整合):
  规则4 注意力瓶颈: DMN ↔ TPN 跷跷板 — 互相抑制，由突显网络切换
  规则3 并行分布式: "苹果" 分布存储在视觉 + 味觉 + 嗅觉 + 语言区域
  规则1 分层处理: 联合皮层位于皮层层级顶端，接收多级前馈输入
  规则2 双向加工: dlPFC → 感觉皮层的自上而下预期信号

子模块:
├── dmn.py         默认模式网络 — 自我模型、"我是谁"的 Hebb 表征 (SelfModel)
├── fpn.py         额顶网络 — 选择性注意探照灯 (FrontoparietalNetwork) ★ NEW v4.3
├── tpn.py         任务正网络 — TPN↔DMN 跷跷板动态 (TaskPositiveNetwork) ★ NEW v4.3
└── crossmodal.py  跨模态整合 — COCO Visual↔Text Hebb 关联学习 (Stage 2)
"""

# 默认模式网络
from cerebrum.association.dmn import SelfModel

# 额顶网络 (v4.3 — 图3 规则4)
from cerebrum.association.fpn import FrontoparietalNetwork

# 任务正网络 (v4.3 — 图3 规则4)
from cerebrum.association.tpn import TaskPositiveNetwork

# 跨模态整合
# (stage2_crossmodal.py 作为独立脚本运行，不在此导入)

__all__ = ['SelfModel', 'FrontoparietalNetwork', 'TaskPositiveNetwork']
