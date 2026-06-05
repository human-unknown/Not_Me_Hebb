# NotMe 项目状态报告

> **生成日期**: 2026-06-05
> **版本**: v4.0 — 人脑层级结构
> **分支**: main

---

## 一、项目概览

| 项目 | 值 |
|------|-----|
| 项目名 | NotMe — 自由能原理情感智能体 |
| 当前版本 | v4.0-人脑层级结构 |
| 代码总行数 | ~22,500 行 Python |
| 最近提交 | `e04b53d` — Fix gestalt feature normalization |
| 提交总数 | 9 |
| 核心原则 | 自由能原理 · Hebb网络 · 仿脑区结构 · 无现象学 · 无LLM |

---

## 二、架构版本历史

| 版本 | 描述 |
|------|------|
| v1-v3 | 扁平文件结构，L0-L3 分层 (layer0_model.py, layer1_free_energy.py, ...) |
| **v4.0** | **人脑层级结构** — 按图1 (人脑层级结构总览) 的4级嵌套包含关系重组架构 |

### v4.0 层级结构

```
Level 1: 中枢神经系统 (CNS)      → cns/           (6 files)
  ├─ Level 2: 大脑 (Cerebrum)    → cerebrum/      (36 files)
  │   ├─ 额叶 (Frontal Lobe)     → frontal_lobe/   (4 modules)
  │   ├─ 顶叶 (Parietal Lobe)    → parietal_lobe/  (3 modules)
  │   ├─ 颞叶 (Temporal Lobe)    → temporal_lobe/  (4 modules)
  │   ├─ 枕叶 (Occipital Lobe)   → occipital_lobe/ (5 modules)
  │   ├─ 边缘系统 (Limbic)       → limbic_system/  (6 modules)
  │   ├─ 基底神经节 (Basal Ganglia) → basal_ganglia/ (4 modules)
  │   └─ 联合皮层 (Association)  → association/    (2 modules)
  │
  └─ Level 2: 脑干 + 小脑        → brainstem_cerebellum/ (16 files)
      ├─ 中脑 (Midbrain)         → midbrain/       (3 modules)
      ├─ 脑桥 (Pons)             → pons/           (2 modules)
      ├─ 延髓 (Medulla)          → medulla/        (1 module)
      ├─ 小脑 (Cerebellum)       → cerebellum/     (2 modules)
      └─ 神经调节 (Neuromod)     → neuromodulatory/ (2 modules)

独立层:
  脊髓 (Spinal)                   → spinal/         (1 module)
  身体模型 (Body)                 → body/           (2 modules)
  环境 (Environments)             → environments/   (1 module)
  工具 (Tools)                    → tools/          (2 modules)
  入口 (Entry)                    → entry/          (2 modules)

备注系统 (非层级):
  布罗德曼分区                    → brodmann_areas.md
```

---

## 三、文件统计

### 根目录 (24 个 .py)

核心运行时文件，作为向后兼容层保留：

```
agent.py              — Agent 主类, 全系统整合
broca.py              — Broca 区, Hebb 词序链语言生成
data_types.py         — 全局数据结构 (Theta, BodyVector, FreeEnergy...)
dialogue_memory.py    — 韦尼克区, 对话记忆与理解
image_encoder.py      — 图像 Gabor 编码器
layer0_model.py       — 海马, Hebb 集群记忆
layer0_visual.py      — 枕叶, 视觉通路 Gabor 编码
layer1_free_energy.py — 扣带回/ACC, 自由能计算
layer2_inference.py   — 前额叶, EFE 行动选择
layer2_5_moe.py       — 基底节, MoE 动作门控
layer3_meta.py        — 神经调节, 元学习
main_dialogue.py      — Stage 6 人机对话入口
params.py             — 默认参数与边界
self_model.py         — DMN, 自我模型
sentiment.py          — 杏仁核, Hebb 情感词汇网络
stage2_crossmodal.py  — 跨模态 Hebb 学习
text_interface.py     — 文本环境
visual_interface.py   — 视觉环境接口
word_speech.py        — 词级 TTS
word_spectrum_generator.py — 词频谱生成
stdin_reader.py       — 非阻塞 stdin
type_aliases.py       — 类型别名
utils.py              — 工具函数
clean_corpus.py       — 语料清洗
```

### 脑结构 (71 个 .py)

| 包 | __init__.py | 已实现 | 占位 | 合计 |
|----|-----------|--------|------|------|
| cns | 1 | 5 | 0 | 6 |
| cerebrum/frontal_lobe | 1 | 2 | 2 | 5 |
| cerebrum/parietal_lobe | 1 | 0 | 3 | 4 |
| cerebrum/temporal_lobe | 1 | 1 | 3 | 5 |
| cerebrum/occipital_lobe | 1 | 2 | 3 | 6 |
| cerebrum/limbic_system | 1 | 3 | 3 | 7 |
| cerebrum/basal_ganglia | 1 | 1 | 3 | 5 |
| cerebrum/association | 1 | 2 | 0 | 3 |
| brainstem_cerebellum | 1 | 0 | 0 | 1 |
| brainstem_cerebellum/midbrain | 1 | 0 | 3 | 4 |
| brainstem_cerebellum/pons | 1 | 0 | 2 | 3 |
| brainstem_cerebellum/medulla | 1 | 0 | 1 | 2 |
| brainstem_cerebellum/cerebellum | 1 | 0 | 2 | 3 |
| brainstem_cerebellum/neuromodulatory | 1 | 1 | 1 | 3 |
| body | 1 | 0 | 2 | 3 |
| spinal | 1 | 0 | 1 | 2 |
| environments | 1 | 1 | 0 | 2 |
| tools | 1 | 2 | 0 | 3 |
| entry | 1 | 2 | 0 | 3 |
| **总计** | **19** | **22** | **30** | **71** |

---

## 四、实现状态 —— 按脑区

### 已实现 (12 个核心模块)

| 脑区 | 模块 | 功能 |
|------|------|------|
| **前额叶 (dlPFC)** | `frontal_lobe/prefrontal.py` | EFE 期望自由能、行动选择、社会信念更新 |
| **布罗卡区 (BA44/45)** | `frontal_lobe/broca.py` | Hebb 词序链 `speak_from_state()` + 整句检索 `speak_sentence()` |
| **韦尼克区 (BA22)** | `temporal_lobe/wernicke.py` | 语言理解回路、对话工作记忆、睡眠巩固 |
| **视觉通路** | `occipital_lobe/visual_pathway.py` | Gabor V1+V2+V4+Color+Pulvinar+Dorsal (32 filters × 4×4 grid, 6条子通路) |
| **视网膜→LGN** | `occipital_lobe/retina_lgn.py` | 图像 Gabor 多尺度编码 |
| **海马** | `limbic_system/hippocampus.py` | ClusterNetwork Hebb 集群记忆、模式补全、睡眠回放巩固 |
| **杏仁核** | `limbic_system/amygdala.py` | Hebb 情感词汇网络——从 F_body 变化学习词效价，零手标词典 |
| **扣带回/ACC** | `limbic_system/cingulate.py` | 自由能 F_body/F_social/F_cognitive/F_accuracy 计算、效价/唤醒、习惯化 |
| **基底节** | `basal_ganglia/action_gating.py` | MoE 门控——疲劳预算轮替、动作选择/抑制 |
| **DMN** | `association/dmn.py` | SelfModel——"我是谁"Hebb 表征、人格锚点、自传体记忆 |
| **跨模态联合** | `association/crossmodal.py` | Stage 2 跨模态 Hebb 学习 (COCO Visual↔Text) |
| **神经调节** | `neuromodulatory/meta_learning.py` | MetaLearner——有限差分梯度下降、关键期、可塑性衰减、创伤模拟 |

### 待实现 (25 个占位模块)

| 优先级 | 脑区 | 文件 | 功能 |
|--------|------|------|------|
| P0 | 下丘脑 | `limbic_system/hypothalamus.py` | 稳态 setpoint、驱力系统、HPA轴 |
| P0 | 丘脑 | `limbic_system/thalamus.py` | 感觉门控、LGN/MGN/Pulvinar中继 |
| P0 | VTA | `midbrain/vta.py` | 多巴胺 RPE、动机调制 |
| P1 | 运动皮层 | `frontal_lobe/motor_cortex.py` | BA4 M1 + BA6 前运动/SMA |
| P1 | 眶额皮层 | `frontal_lobe/orbitofrontal.py` | BA11 价值评估、奖赏预测 |
| P1 | 蓝斑核 | `pons/locus_coeruleus.py` | NE 唤醒度、注意力 SNR 增强 |
| P1 | 小脑运动 | `cerebellum/motor_coordination.py` | 前向/逆向内部模型、在线纠错 |
| P1 | 小脑时序 | `cerebellum/predictive_timing.py` | 时序预测、节律生成、对话轮转 |
| P1 | 内感受 | `body/interoception.py` | 岛叶层级、内感受推理 |
| P1 | 可塑性 | `neuromodulatory/plasticity.py` | 稳态可塑性、事件驱动LR、睡眠巩固 |
| P2 | 体感皮层 | `parietal_lobe/somatosensory.py` | BA3,1,2 触觉/本体感觉 |
| P2 | 空间注意 | `parietal_lobe/spatial_attention.py` | IPS/SPL 注意力探照灯、显著图 |
| P2 | TPJ | `parietal_lobe/tpj.py` | 心理理论、二阶信念、视角采择 |
| P2 | 听皮层 | `temporal_lobe/auditory_cortex.py` | BA41/42 频谱编码、双耳听觉 |
| P2 | IT皮层 | `temporal_lobe/it_cortex.py` | 物体识别、视角不变性 |
| P2 | 梭状回 | `temporal_lobe/fusiform.py` | FFA 面孔识别 + VWFA 文字识别 |
| P2 | 纹状体 | `basal_ganglia/striatum.py` | D1/D2 MSN、习惯学习、NAc |
| P2 | 黑质 | `midbrain/substantia_nigra.py` | SNc DA供给、运动vigor |
| P3 | 嗅皮层 | `limbic_system/olfactory.py` | 气味编码、普鲁斯特效应 |
| P3 | 苍白球 | `basal_ganglia/pallidum.py` | GPe/GPi 输出门控 |
| P3 | 底丘脑核 | `basal_ganglia/subthalamic.py` | 超直接通路、冲突检测、动作取消 |
| P3 | 上丘 | `midbrain/superior_colliculus.py` | 快速定向、多感官地图 |
| P3 | 网状结构 | `pons/reticular_formation.py` | ARAS、睡眠-觉醒状态机 |
| P3 | 自主神经 | `medulla/autonomic.py` | 交感/副交感、HRV、SCR |
| P3 | 运动输出 | `spinal/motor_output.py` | CPG、反射弧、运动平滑 |
| P3 | 身体状态 | `body/body_state.py` | BodyVector ODE 积分 |

---

## 五、技术参数

| 参数 | 值 |
|------|-----|
| 感知维度 D | 330 (text 64 + V1 96 + V2 64 + V4 64 + Color 42 = vision 266) |
| 隐状态 H | 16 |
| 最大簇数 K | 256 |
| 行动数 A | 5 (grid) / 3 (dialogue) |
| Theta 参数 | 24 (L0=6, L1=9, L2=5, L3=4) |
| 身体维度 M | 5 (grid) / 8 (text) |
| 语料规模 | 50,000 行中文对话 |
| 词表规模 | 12,000 词 |
| 词序 trigram 集群 | ≤50,000 (频率≥3) |
| 句子记忆容量 | 12,000 句 |

---

## 六、数据流 (全链路)

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

睡眠巩固 (每100步):
  Hippocampal replay → Systems consolidation → Cross-association
  → Self-model integration → Synaptic homeostasis (weak cluster pruning)
```

---

## 七、里程碑进展

| 里程碑 | 状态 | 入口 |
|--------|------|------|
| Stage 2 跨模态学习 | ✅ 完成 | `python stage2_crossmodal.py` |
| M1 单智能体生存 | ✅ 完成 | (gridworld, 已在 v4.0 中移除) |
| M2 认知探索 | ✅ 完成 | (gridworld, 已在 v4.0 中移除) |
| M3 多智能体社会 | ✅ 完成 | (gridworld, 已在 v4.0 中移除) |
| M4 参数扫描 | ✅ 完成 | (gridworld, 已在 v4.0 中移除) |
| M5 长程发育 | ✅ 完成 | (gridworld, 已在 v4.0 中移除) |
| Stage 6 人机对话 | ✅ 运行中 | `python main_dialogue.py` |
| v4.0 脑层级整理 | ✅ 完成 | 本次版本 |
| v4.1 导入迁移 | 🔲 待定 | 将脑结构副本的导入更新为包内相对导入 |

---

## 八、v4.0 清理记录

本次版本清理了 48 个冗余文件：

- **网格世界** (6): gridworld ×2, main ×2, main_text, main_social ×2, main_multimodal
- **测试文件** (12): test_units, test_integration, test_twopass, test_a3_coupling, test_belief_anchor ×2, test_dialogue_quality, test_emotional_contagion, test_inner_speech, test_own_words, test_pca_shared, test_self_hearing
- **实验脚本** (7): phase1_visual, phase2_pp/dorsal/it/v4/tier2, layer0_gestalt
- **M4 工具** (8): viz ×2, features ×2, sweep ×2, attractors ×2
- **旧音频/数据** (9): spectrum_generator, speech_output, vocoder, prepare_*, record_multimodal, multimodal_interface, expand_vocabulary
- **旧报告/数据** (5): PHASE1_REPORT, PROJECT_REPORT, STATUS_REPORT_v3.2, clip_vision.npy, multimodal_log.npy

---

## 九、参考文档

| 文档 | 路径 | 内容 |
|------|------|------|
| 项目说明 | `CLAUDE.md` | 核心原则、架构、运行指南 |
| v4.0 架构 | `v4_architecture.md` | 完整目录树、旧→新映射、数据流 |
| 布罗德曼分区 | `brodmann_areas.md` | BA1-BA52 完整编号 × v4.0 模块对照 |
| 人脑图表集 | `人脑结构调查_可视化图表集.html` | 图1-4 (层级/功能/规则/流转) |
| 阅读说明 | `README.md` | 项目简介 |
