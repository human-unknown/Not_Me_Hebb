"""
cerebrum — 大脑 (Cerebrum)  [Level 2a]

人脑层级结构第二层：最高级中枢，负责认知、意识、决策。

包含四大叶、边缘系统、基底神经节、联合皮层：

子包 (Level 3):
├── frontal_lobe/    额叶 — 决策、计划、语言产出、运动
├── parietal_lobe/   顶叶 — 触觉、空间、本体感觉
├── temporal_lobe/   颞叶 — 听觉、记忆、语言理解
├── occipital_lobe/  枕叶 — 视觉处理
├── limbic_system/   边缘系统 — 记忆、情绪、稳态
├── basal_ganglia/   基底神经节 — 动作选择/门控、习惯、奖赏
├── thalamus/        丘脑 — 感觉中继/门控、注意力调制 (v4.1 独立)
└── association/     联合皮层 — DMN、跨模态整合

设计原则 (CLAUE.md 五条核心原则):
1. 自由能原理 — 一切行为统一在最小化自由能框架下
2. Hebb 网络 — fire together, wire together
3. 仿脑区结构 — 按进化形成的脑区功能分化组织计算
4. 无现象学 — 情感是 F_body 的数值动力学产物
5. 无 LLM — 认知核心不依赖 transformer
"""
