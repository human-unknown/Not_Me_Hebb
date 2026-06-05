"""
小脑 (Cerebellum)  [Level 3]

功能：运动协调 · 平衡维持 · 运动学习 · 时序预测 · 认知功能

包含约 80% 的脑神经元 (主要是颗粒细胞)。

核心结构:
├── 小脑皮层 (Cerebellar Cortex)  — 分子层 + 浦肯野层 + 颗粒层
│   ├── 浦肯野细胞 (Purkinje)     — 唯一输出 (抑制性 GABA)
│   ├── 颗粒细胞 (Granule)         — 苔藓纤维输入 (兴奋性 Glu)
│   └── 攀缘纤维 (Climbing Fiber)  — 下橄榄核→浦肯野 (误差信号)
├── 小脑深部核团:
│   ├── 齿状核 (Dentate)          — 运动规划/认知
│   ├── 栓状核 (Emboliform)       — 运动执行
│   ├── 球状核 (Globose)          — 运动调节
│   └── 顶核 (Fastigial)          — 平衡/前庭

功能分区 (纵轴):
  前庭小脑 (Vestibulocerebellum)  — 平衡、眼动
  脊髓小脑 (Spinocerebellum)       — 运动执行纠错
  皮层小脑 (Cerebrocerebellum)     — 运动规划、时序预测、认知

子模块:
├── motor_coordination.py  运动协调/纠错 — 内部模型: 预测→比较→纠正 [待实现]
└── predictive_timing.py   时序预测 — 前馈模型、节律生成 [待实现]
"""
