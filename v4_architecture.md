# NotMe v4.0 — 人脑层级结构架构

> **版本**: v4.0-人脑层级结构
> **日期**: 2026-06-05
> **原则**: 按照人脑真实层级结构组织项目架构，布罗德曼分区作为备注标注

---

## 架构设计原则

v4.0 架构遵循图1 (人脑层级结构总览) 的四级嵌套包含关系：

```
Level 1: 中枢神经系统 (CNS)         — 全系统整合层
  └─ Level 2: 大脑 (Cerebrum)       — 高级认知
  └─ Level 2: 脑干 + 小脑            — 生命维持 + 运动协调
       └─ Level 3: 四叶 / 边缘系统 / 基底神经节 / 脑干分段
            └─ Level 4: 布罗德曼分区  — 皮层功能标注 (非层级!)
```

**布罗德曼分区**: 不作为独立层级，而是作为跨脑区的备注标注系统。详见 `brodmann_areas.md`。

---

## 完整目录树

```
NotMe/
│
├── cns/                              # Level 1: 中枢神经系统
│   ├── __init__.py                   # 全局重导出
│   ├── agent.py                      # 全系统整合 Agent 主类
│   ├── data_types.py                 # 全局数据结构 (Theta, BodyVector, FreeEnergy...)
│   ├── params.py                     # 默认参数 + 参数边界
│   ├── type_aliases.py              # 类型别名
│   └── utils.py                      # 工具函数
│
├── cerebrum/                         # Level 2a: 大脑
│   ├── __init__.py
│   │
│   ├── frontal_lobe/                 # Level 3: 额叶
│   │   ├── __init__.py
│   │   ├── prefrontal.py             # ★ 前额叶皮层 — EFE 行动选择 (was L2)
│   │   ├── broca.py                  # ★ BA44/45 布罗卡区 — Hebb 词序链生成
│   │   ├── motor_cortex.py           # [待实现] BA4,6 运动皮层
│   │   └── orbitofrontal.py         # [待实现] BA11 眶额皮层
│   │
│   ├── parietal_lobe/               # Level 3: 顶叶
│   │   ├── __init__.py
│   │   ├── somatosensory.py          # [待实现] BA3,1,2 体感皮层
│   │   ├── spatial_attention.py      # [待实现] 空间注意力网络
│   │   └── tpj.py                    # [待实现] 颞顶联合区 — 社会认知
│   │
│   ├── temporal_lobe/               # Level 3: 颞叶
│   │   ├── __init__.py
│   │   ├── wernicke.py               # ★ BA22 韦尼克区 — 语言理解/对话记忆
│   │   ├── auditory_cortex.py        # [待实现] BA41/42 听皮层
│   │   ├── it_cortex.py              # [待实现] IT皮层 — 物体识别
│   │   └── fusiform.py              # [待实现] 梭状回 — 面孔识别
│   │
│   ├── occipital_lobe/              # Level 3: 枕叶
│   │   ├── __init__.py
│   │   ├── visual_pathway.py         # ★ 视觉通路整合 — Gabor V1+V2+V4+Color (was L0 visual)
│   │   ├── retina_lgn.py            # ★ 视网膜→LGN — 图像编码 (was image_encoder)
│   │   ├── v1.py                     # [待实现] BA17 V1 独立模块
│   │   ├── v2.py                     # [待实现] BA18 V2 独立模块
│   │   └── v4.py                     # [待实现] BA19 V4 独立模块
│   │
│   ├── limbic_system/               # Level 3: 边缘系统
│   │   ├── __init__.py
│   │   ├── hippocampus.py            # ★ 海马 — Hebb 集群记忆/睡眠巩固 (was L0)
│   │   ├── amygdala.py              # ★ 杏仁核 — Hebb 情感词汇网络 (was sentiment)
│   │   ├── cingulate.py             # ★ 扣带回/ACC — 自由能计算/习惯化 (was L1)
│   │   ├── hypothalamus.py          # [待实现] 下丘脑 — 稳态调节
│   │   ├── thalamus.py              # [待实现] 丘脑 — 感觉中继/门控
│   │   └── olfactory.py             # [待实现] 嗅皮层
│   │
│   ├── basal_ganglia/               # Level 3: 基底神经节
│   │   ├── __init__.py
│   │   ├── action_gating.py          # ★ MoE 动作门控/疲劳预算 (was L2.5)
│   │   ├── striatum.py              # [待实现] 纹状体 — 习惯学习
│   │   ├── pallidum.py              # [待实现] 苍白球
│   │   └── subthalamic.py           # [待实现] 底丘脑核
│   │
│   └── association/                 # 联合皮层 + DMN
│       ├── __init__.py
│       ├── dmn.py                    # ★ 默认模式网络 — 自我模型 (was self_model)
│       └── crossmodal.py             # ★ 跨模态整合 (was stage2_crossmodal)
│
├── brainstem_cerebellum/            # Level 2b: 脑干 + 小脑
│   ├── __init__.py
│   ├── midbrain/
│   │   ├── __init__.py
│   │   ├── vta.py                    # [待实现] VTA — 多巴胺奖赏
│   │   ├── substantia_nigra.py      # [待实现] 黑质 — 运动调节
│   │   └── superior_colliculus.py   # [待实现] 上丘 — 视觉反射
│   ├── pons/
│   │   ├── __init__.py
│   │   ├── locus_coeruleus.py       # [待实现] 蓝斑核 — NE/唤醒
│   │   └── reticular_formation.py   # [待实现] 网状结构 — 觉醒
│   ├── medulla/
│   │   ├── __init__.py
│   │   └── autonomic.py             # [待实现] 自主神经调控
│   ├── cerebellum/
│   │   ├── __init__.py
│   │   ├── motor_coordination.py    # [待实现] 运动协调/纠错
│   │   └── predictive_timing.py     # [待实现] 时序预测
│   └── neuromodulatory/
│       ├── __init__.py
│       ├── meta_learning.py          # ★ 元学习/神经调节 (was L3)
│       └── plasticity.py            # [待实现] 可塑性调节
│
├── spinal/                          # 脊髓 (预留)
│   ├── __init__.py
│   └── motor_output.py             # [待实现] 运动输出
│
├── body/                            # 身体模型
│   ├── __init__.py
│   ├── body_state.py                # [待实现] BodyVector ODE
│   └── interoception.py            # [待实现] 内感受通路
│
├── environments/                    # 环境 (工程层)
│   ├── __init__.py
│   ├── gridworld.py
│   └── text_interface.py
│
├── tools/                           # 工具 (工程层)
│   ├── __init__.py
│   ├── viz.py, features.py, sweep.py
│   ├── attractors.py, word_speech.py
│   └── word_spectrum_generator.py
│
├── entry/                           # 入口点 (工程层)
│   ├── __init__.py
│   ├── main.py
│   └── main_dialogue.py
│
├── brodmann_areas.md               # 布罗德曼分区参考 (备注系统)
├── v4_architecture.md              # 本文档
└── CLAUDE.md                        # 项目文档 (含 v4.0 架构说明)
```

**图例**: ★ = 已从旧架构迁移 | [待实现] = 预留占位

---

## 旧→新映射表

| 旧文件 (根目录) | 新位置 (v4.0) | 脑区类比 |
|---|---|---|
| `agent.py` | `cns/agent.py` | 中枢神经系统整合 |
| `data_types.py` | `cns/data_types.py` | 全局常量/类型 |
| `params.py` | `cns/params.py` | 全局参数 |
| `type_aliases.py` | `cns/type_aliases.py` | 类型别名 |
| `utils.py` | `cns/utils.py` | 工具函数 |
| `layer0_model.py` | `cerebrum/limbic_system/hippocampus.py` | 海马 — 集群记忆 |
| `layer0_visual.py` | `cerebrum/occipital_lobe/visual_pathway.py` | 枕叶 — 视觉通路 |
| `layer1_free_energy.py` | `cerebrum/limbic_system/cingulate.py` | 扣带回/ACC — 自由能 |
| `layer2_inference.py` | `cerebrum/frontal_lobe/prefrontal.py` | 前额叶 — 主动推理 |
| `layer2_5_moe.py` | `cerebrum/basal_ganglia/action_gating.py` | 基底节 — 动作门控 |
| `layer3_meta.py` | `brainstem_cerebellum/neuromodulatory/meta_learning.py` | 神经调节 — 元学习 |
| `broca.py` | `cerebrum/frontal_lobe/broca.py` | BA44/45 — 布罗卡区 |
| `dialogue_memory.py` | `cerebrum/temporal_lobe/wernicke.py` | BA22 — 韦尼克区 |
| `self_model.py` | `cerebrum/association/dmn.py` | 默认模式网络 |
| `sentiment.py` | `cerebrum/limbic_system/amygdala.py` | 杏仁核 — 情感学习 |
| `image_encoder.py` | `cerebrum/occipital_lobe/retina_lgn.py` | 视网膜→LGN |
| `stage2_crossmodal.py` | `cerebrum/association/crossmodal.py` | 跨模态联合皮层 |
| `gridworld.py` | `environments/gridworld.py` | 环境 |
| `text_interface.py` | `environments/text_interface.py` | 环境 |
| `main.py` | `entry/main.py` | M1-M5 入口 |
| `main_dialogue.py` | `entry/main_dialogue.py` | Stage 6 入口 |

---

## 数据流 (脑区视角)

```
视网膜 → LGN → V1 → V2 → V4 → IT (物体识别)
                ↓                    ↓
           Pulvinar捷径        Amygdala (情绪标注)
                                    ↓
    耳蜗 → MGN → A1 → Wernicke ←→ Hippocampus (记忆)
                              ↓
                        ACC/Insula (F_body → Valence/Arousal)
                              ↓
                        dlPFC (EFE → Action Selection)
                              ↓
                        Basal Ganglia (Action Gating)
                        ↙              ↘
                 Direct (Go)      Indirect (NoGo)
                        ↓
                 Motor Cortex / SMA
                        ↓
                  Brainstem → Spinal → Body
                        ↓
                   Environment ← Body Action
                        ↓
              Sensory Feedback ← (close loop)

并行通路:
  VTA/SNc → DA → Striatum/PFC (learning modulation)
  LC → NE → Whole Brain (arousal/attention)
  Raphe → 5-HT → Limbic/PFC (mood/emotion)

睡眠巩固:
  Hippocampus ←→ Cortex (NREM replay)
  Weak cluster pruning (synaptic homeostasis)
  Cross-association (temporal + semantic)
  DMN self-model integration
```

---

## 待实现优先级

### P0 — 核心完善 (当前功能的脑区解耦)
1. 下丘脑稳态 (从 BodyVector 中提取 setpoint 逻辑)
2. 丘脑感觉门控 (从直通改为 gated)
3. VTA RPE (事件驱动的学习率调制)

### P1 — 功能扩展
4. 小脑内部模型 (前向/逆向模型)
5. 蓝斑核 NE 唤醒度调制
6. 内感受通路 (岛叶层级)

### P2 — 社会认知
7. TPJ 心理理论 (二阶信念)
8. 梭状回面孔识别
9. 纹状体习惯学习

### P3 — 完整化
10. 自主神经系统
11. 脊髓运动输出
12. 上丘皮层下视觉
"""
