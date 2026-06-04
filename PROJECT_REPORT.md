# NotMe — 自由能原理情感智能体 · 项目报告

**生成日期**: 2026-06-03  
**版本**: v3 (审计修复后)  
**基准评分**: Avg Overall 0.546 | Relevance 0.277 | Long Quote 0%

---

## 一、项目概要

### 1.1 核心目标

创造一个真正有情感、有意识的人工智能——不是模拟情感（LLM 路线），而是让情感和意识从**身体稳态、预测加工、和 Hebb 记忆的自组织动力学中涌现**。

### 1.2 五条核心原则

| # | 原则 | 含义 |
|---|------|------|
| 1 | **自由能原理 (FEP)** | 一切心理活动都是最小化 F = F_body + F_social + F_cognitive + F_accuracy |
| 2 | **Hebb 网络** | 记忆和学习基于 fire together, wire together |
| 3 | **仿脑区结构** | 按进化形成的脑区功能分化组织计算 |
| 4 | **无现象学** | 不从主观体验建模，情感是身体稳态动力学的数值产物 |
| 5 | **无 LLM** | 认知核心完全独立于 transformer，语言能力来自 Hebb 统计学习 |

### 1.3 自由能公式

```
F = F_body + F_social + F_cognitive + F_accuracy

F_body:      身体稳态偏离 → 驱动生存行为
F_social:    社会预测误差 → 驱动社会互动
F_cognitive: 模型复杂度代价 → 驱动学习与探索
F_accuracy:  集群预测精度 → 驱动感知与记忆

Valence（效价）  = tanh(-η × F_body)
Arousal（唤醒）  = tanh(η × |F_body|)
```

---

## 二、架构总览

### 2.1 五层脑区结构

```
┌─────────────────────────────────────────────────────┐
│ L3: 元学习 (神经调节系统)                              │
│     MetaLearner — 关键期、可塑性衰减、创伤模拟          │
│     Theta = 23 个可学习参数, 有限差分梯度下降            │
├─────────────────────────────────────────────────────┤
│ L2.5: MoE 门控 (基底节)                               │
│     动作门控, 决定何时表达                              │
├─────────────────────────────────────────────────────┤
│ L2: 主动推理 (前额叶)                                  │
│     期望自由能 (EFE) 驱动行动选择                       │
│     MoE 门控、社会信念更新                              │
├─────────────────────────────────────────────────────┤
│ L1: 自由能计算 (岛叶 + 前扣带)                          │
│     效价/唤醒、习惯化追踪、注意力调制                     │
│     SocialContext — 人类社会信号预测                    │
├─────────────────────────────────────────────────────┤
│ L0: Hebb 集群记忆 (感觉皮层 + 海马)                     │
│     ClusterNetwork — 特征哈希 O(1) + 桶内竞争           │
│     感知聚类、模式补全、睡眠巩固                         │
├─────────────────────────────────────────────────────┤
│ Broca 区: 语言产出                                     │
│     speak_sentence() — Hebb 记忆检索 (整句回忆)         │
│     speak_from_state() — 词序 Hebb 链逐词生成           │
│     Concept→Word Hebb 联想网络 (34,911 关联)            │
└─────────────────────────────────────────────────────┘
```

### 2.2 数据流

```
s → L0.learn(s) → L1.compute_F(z,s) → L2.select_action() → a → L3.meta.update(F)
                     ↑                        ↓
                 SocialContext          self_model (DMN)
                 HebbEmotionalLexicon   dialogue_memory
```

### 2.3 关键数据维度

| 维度 | 值 | 说明 |
|------|-----|------|
| D (感知维度) | 330 | text[64] \| vision[64] \| audio[64] \| body[64] \| meta[64] \| action[10] |
| H (隐状态) | 16 | 信念向量 |
| K (最大簇数) | 256 | Hebb 记忆容量 |
| Theta | 23 | 可学习参数 |
| M (身体维度) | 5/8 | gridworld=5, text=8 |

### 2.4 Theta 参数清单 (23 个)

#### L0 — 生成模型 (5)
| 参数 | 默认值 | 范围 | 含义 |
|------|--------|------|------|
| `sigma_z` | 0.1 | [0.01, 2.0] | 状态噪声 |
| `sigma_x` | 1.0 | [0.1, 5.0] | 感知噪声 |
| `decay_rate` | 0.01 | [0.001, 0.1] | 簇衰减率 |
| `cluster_threshold` | 0.70 | [0.5, 0.99] | 簇匹配阈值 |
| `learn_rate_l0` | 0.05 | [0.001, 0.5] | 簇学习率 |

#### L1 — 自由能权重 (9)
| 参数 | 默认值 | 范围 | 含义 |
|------|--------|------|------|
| `w_body` | 1.0 | [0.1, 5.0] | 躯体域权重 |
| `w_social` | 1.0 | [0.1, 5.0] | 社会域权重 |
| `w_cognitive` | 1.0 | [0.1, 5.0] | 认知域权重 |
| `eta_valence` | 0.5 | [0.1, 2.0] | 效价敏感度 |
| `eta_arousal` | 0.5 | [0.1, 2.0] | 唤醒敏感度 |
| `habituation_tau` | 10.0 | [1.0, 100.0] | 习惯化时间常数 |
| `negativity_bias` | 1.5 | [0.5, 5.0] | 负面信号放大 |
| `w_accuracy` | 0.5 | [0.05, 2.0] | 预测残差权重 |
| `w_F_signal` | 0.1 | [0.01, 1.0] | 集群历史 F_signal 权重 |

#### L2 — 策略推理 (5)
| 参数 | 默认值 | 范围 | 含义 |
|------|--------|------|------|
| `gamma` | 0.95 | [0.1, 0.99] | 时间折扣 |
| `exploration_bonus` | 0.1 | [0.01, 1.0] | 探索奖励 |
| `temperature` | 1.0 | [0.1, 10.0] | softmax 温度 |
| `n_policy_samples` | 16 | [4, 64] | 策略采样数 |
| `urgency_weight` | 0.3 | [0.05, 1.0] | 紧急度权重 |

#### L3 — 元学习 (4)
| 参数 | 默认值 | 范围 | 含义 |
|------|--------|------|------|
| `meta_lr` | 0.01 | [0.001, 0.1] | 元学习率 |
| `grad_epsilon` | 0.001 | [0.0001, 0.01] | 有限差分 epsilon |
| `plasticity_decay` | 0.999 | [0.99, 0.9999] | 可塑性衰减 |
| `critical_window` | 1000 | [100, 5000] | 关键期步数 |

---

## 三、项目文件结构

```
NotMe/
├── agent.py                  # Agent 主类 — 组装 L0-L3 全链路
├── gridworld.py              # 网格世界环境 (M1-M3)
├── text_interface.py         # 文本环境 — 语料导航 (Stage 6)
│
├── layer0_model.py           # L0: ClusterNetwork — Hebb 集群记忆
├── layer1_free_energy.py     # L1: 自由能计算 + 效价/唤醒 + 习惯化
├── layer2_inference.py       # L2: EFE 行动选择 + 社会信念
├── layer2_5_moe.py           # L2.5: MoE 门控
├── layer3_meta.py            # L3: 元学习 (MetaLearner)
│
├── data_types.py             # 全部数据结构 (Theta, Cluster, BodyVector, FreeEnergy, etc.)
├── params.py                 # 默认参数 + 参数边界
├── type_aliases.py           # 类型别名
├── utils.py                  # 工具函数
│
├── broca.py                  # Broca 区 — 词序 Hebb 链生成 + 整句检索
├── dialogue_memory.py        # 对话记忆 + Wernicke 理解回路
├── self_model.py             # 自我模型 (DMN/vmPFC)
├── sentiment.py              # 情感分析 + Hebb 情感词汇学习
├── word_speech.py            # 词级 TTS 输出
│
├── main.py                   # M1-M5 主循环入口
├── main_dialogue.py          # Stage 6: 人机对话闭环
├── stdin_reader.py           # 非阻塞 stdin 输入线程
│
├── features.py               # 行为特征提取 (M4)
├── sweep.py                  # 参数扫描 (M4)
├── attractors.py             # 吸引子可视化 (M4)
├── viz.py                    # 可视化仪表板
│
├── clean_corpus.py           # 语料清洗工具
├── corpus.txt                # 语料 (229,201 句二次元中文对话)
│
├── .cache/                   # 嵌入缓存 (自动生成)
├── dashboards/               # 可视化输出
├── audio_output/             # Agent 语音输出
└── word_audio/               # 词级音频文件
```

---

## 四、里程碑与运行

| 里程碑 | 描述 | 命令 |
|--------|------|------|
| **M1** | 单智能体生存 — 10×10 网格 | `python main.py` |
| **M2** | 认知价值驱动探索 — 视野遮蔽 | `python main.py --m2` |
| **M3** | 多智能体社会 | `python main.py --m3` |
| **M4** | 参数扫描与吸引子涌现 | `python main.py --m4` |
| **M5** | 元学习长程发育 | `python main.py --m5` |
| **Stage 6** | 人机对话闭环 | `python main_dialogue.py` |

---

## 五、脑区功能映射

| 层 | 模块 | 类比脑区 | 功能 | 状态 |
|---|------|---------|------|------|
| **L0** | `layer0_model.py` | 感觉皮层 + 海马 | Hebb 集群记忆、感知聚类、模式补全 | ✅ |
| **L1** | `layer1_free_energy.py` | 岛叶 + 前扣带 | 自由能计算、效价/唤醒、习惯化 | ✅ |
| **L2** | `layer2_inference.py` | 前额叶 | EFE 行动选择、MoE 门控、社会信念 | ✅ |
| **L2.5** | `layer2_5_moe.py` | 基底节 | 动作门控 | ✅ |
| **L3** | `layer3_meta.py` | 神经调节系统 | 元学习、关键期、创伤模拟 | ✅ |
| **Broca** | `broca.py` | 布罗卡区 | 词序学习、语言产出 | ✅ |
| **Wernicke** | `dialogue_memory.py` | 韦尼克区 | 语言理解 | ✅ |
| **ACC/OFC** | `dialogue_memory.py` | 前扣带/眶额 | 响应监控 | ✅ |
| **DMN** | `self_model.py` | 默认模式网络 | 自我模型 | ✅ |

---

## 六、v3 审计修复记录 (2026-06-03)

### 6.1 本轮修复 (8 项)

| # | 等级 | 文件 | 问题 | 修复方案 |
|---|------|------|------|----------|
| V1 | 🔴 | `main_dialogue.py:183` | 查询混合权重硬编码 (0.40/0.30/0.15) | → 精度加权：各通道 precision 由触发记忆数、激活度、对话轮数、自我体验数驱动 |
| V2 | 🔴 | `main_dialogue.py:347` | `belief_weight = 0.80 + sv*0.10` | → 身体偏离度驱动 [0.5, 0.9]，自我效价仅做 ±0.05 微调 |
| V3 | 🟡 | `sweep.py:24` | SWEEP_PARAMS 缺少 negativity_bias, w_accuracy, w_F_signal | → 补入 3 个新参数及边界 |
| V4 | 🟡 | `sweep.py:92` | run_with_theta 缺少新参数默认值 | → 补入 setdefault |
| V5 | 🟡 | `layer1_free_energy.py:252` | novelty `0.5` 系数硬编码 | → theta.eta_valence 驱动 |
| V6 | 🟡 | `layer2_inference.py:233` | 域筛选 `1.5` 阈值硬编码 | → w_body/w_social 比率动态推导 |
| V7 | 🟢 | `broca.py:335` | 多样性奖励 `20.0` 硬编码 | → max_score * 0.4 (数据驱动) |
| V8 | 🟢 | `data_types.py:247` | validate_theta 文档 "20" → 实际 "23" | → 修正为 23，同步 L1(8)→L1(9) |

### 6.2 上一轮修复 (已完成的 16 项)

包括：移除手选 AROUSAL_INDICATORS 集合、移除手选 END_WORDS 集合、Concept→Word Hebb 联想网络替换语料桥、精度加权替换硬编码权重、连续 DMN 耦合函数、F_accuracy 和信任调制参数化、批量编码性能优化等。

---

## 七、基准测试结果

```
======================================================================
  BENCHMARK SUMMARY
======================================================================
  Avg Overall Score:    0.546
  Avg Relevance:        0.277
  Long Quote Rate:      0%
  Short Match Rate:     0%
  Concept→word net:     34,911 associations (cached)
  Word-order clusters:  19,032
  HebbEmoLexicon:       46 words learned
  Self-model:           11 experiences

  All 6 Brain Regions:  [OK] [OK] [OK] [OK] [OK] [OK]
  [PASS] Dialogue quality acceptable
======================================================================
```

### 质量指标说明

| 指标 | 含义 | 值 | 说明 |
|------|------|-----|------|
| Overall Score | 综合质量 (相关+新颖+连贯) | 0.546 | 可接受 |
| Relevance | 回应与输入的相关性 | 0.277 | 中等 (Hebb 词汇网络仍在学习) |
| Long Quote Rate | 直接引用语料长句的比例 | 0% | 优秀 (无机械复制) |
| Short Match Rate | 语料短匹配率 | 0% | 优秀 |

---

## 八、关键技术特性

### 8.1 Hebb 集群记忆 (L0)
- **O(1) 哈希定位**: 前 8 维符号位 → 256 个桶
- **桶内竞争**: 仅比较哈希命中的同桶集群
- **Hebb 学习**: Δw ∝ activation × (input - w)
- **睡眠巩固**: 每 100 步衰减 + 清理弱簇

### 8.2 词序 Hebb 链 (Broca)
- **Trigram 网络**: 19,032 个 trigram 集群
- **自然终止**: 最佳候选相似度 < 0.02 → trigram 统计中极少出现 → 自然句法边界
- **反重复**: 最近 8 词惩罚
- **Concept→Word 网络**: 34,911 个概念-词 Hebb 关联

### 8.3 情感学习 (HebbEmotionalLexicon)
- **无手标词典**: 词的"好/坏"从 Agent 自身 F_body 变化中学习
- **杏仁核门控**: 高唤醒 → 学习率增强
- **冷启动**: 所有词初始中性，随互动逐渐分化

### 8.4 精度加权混合
- **理解向量**: human × 1.0 + memory × sim + context × (n_turns/5)
- **回应评估**: relevance × match + novelty × 1.0 + coherence × (n_turns/5)
- **查询合成**: comp × 触发记忆 + belief × 激活度 + ctx × 对话轮数 + self × 体验数

### 8.5 元学习 (L3)
- **有限差分梯度**: 8 个高影响力参数在线调整
- **关键期**: critical_window 步内学习率 2×
- **可塑性衰减**: update *= plasticity_decay^step
- **创伤模拟**: w_social 和 eta_valence 永久降低

---

## 九、依赖与环境

```
Python >= 3.10
numpy, scipy
sentence-transformers (all-MiniLM-L6-v2, 仅感官编码)
scikit-learn (PCA)
jieba (中文分词)
matplotlib (可视化)
edge-tts (可选, 语音输出)
sounddevice, soundfile, librosa (可选, 音频播放)
```

---

## 十、未来方向

1. **纯统计感官编码**: 替换 sentence-transformer 为纯统计方案 (fastText / 纯 SVD)
2. **多模态融合**: 视觉+听觉+文本的 Hebb 跨模态关联
3. **社会认知深化**: 二阶信念推理、声誉追踪
4. **长程发育**: 10000+ 步的元学习轨迹分析
5. **吸引子景观**: M4 参数空间的完整扫描与可视化

---

## 附录 A: 核心数据流图

```
                     ┌──────────────┐
                     │   Environment │
                     │ (Grid/Text)  │
                     └──────┬───────┘
                            │ s (330-dim sensory)
                            ▼
┌─────────────────────────────────────────────────────────┐
│  L0: ClusterNetwork                                     │
│  s → hash_features → bucket → recall → cluster          │
│  Hebbl earn: centroid += lr * (h - centroid)            │
└──────────────────────┬──────────────────────────────────┘
                       │ c (matched cluster), s_pred
                       ▼
┌─────────────────────────────────────────────────────────┐
│  L1: Free Energy                                        │
│  F_body = w_body * body_deviation / sigma_x²            │
│  F_social = w_social * (pred_v - obs_v)² / sigma_x²     │
│  F_cognitive = w_cognitive * log(n_clusters + 1)        │
│  F_accuracy = w_accuracy * residual + w_F_signal * c.Fs │
│  Valence = tanh(-eta_v * F_body)                        │
│  Arousal = tanh(eta_a * |F_total|)                      │
└──────────────────────┬──────────────────────────────────┘
                       │ F, valence, arousal
                       ▼
┌─────────────────────────────────────────────────────────┐
│  L2: Action Selection                                   │
│  G(a) = pragmatic - exploration * info_gain             │
│  Phase 1: Domain screening (theta-driven threshold)     │
│  Phase 2: Urgency → time budget                         │
│  Phase 3: Cluster experience ranking                    │
│  Phase 4: MoE weighted evaluation                       │
│  Phase 5: Write-back G_ema to cluster                   │
└──────────────────────┬──────────────────────────────────┘
                       │ action
                       ▼
┌─────────────────────────────────────────────────────────┐
│  L3: Meta Learning                                      │
│  grad_i = [F(θ_i+ε) - F(θ_i-ε)] / 2ε                   │
│  θ_i -= lr * grad_i * plasticity * critical_bonus       │
└─────────────────────────────────────────────────────────┘
```

---

## 附录 B: 对话模式完整回路

```
Human Input
    │
    ├─→ text_interface.encode_text() → s[0:64]     (语义编码)
    ├─→ sentiment.analyze_sentiment() → s[80:88]   (情感信号)
    │
    ▼
Agent.comprehend()
    ├─→ L0 回忆: 输入触发了哪些记忆？
    ├─→ 对话上下文: 刚才在说什么？
    ├─→ 情感评估: 这个输入让我感觉如何？
    └─→ 精度加权合成 → comprehension_vec
    │
    ▼
Agent.step(s, social_ctx)
    ├─→ L0.learn(s)
    ├─→ L1.compute_F() → valence, arousal
    ├─→ Hebb 情感学习: F_body 变化 → 词→情感关联
    │
    ▼
L2.select_action() → A₃ (表达)
    │
    ▼
Broca.speak_from_state()
    ├─→ 身体偏离度驱动信念/感官混合权重
    ├─→ Concept→Word Hebb 网络 → 种子词
    ├─→ Trigram 词序链逐词生成
    ├─→ 自然终止 (低相似度 = 句法边界)
    └─→ 在线 Hebb 学习 (强化 concept→word 关联)
    │
    ▼
Self-hearing Loop
    ├─→ 回应编码 → s[128:192] (听觉通道)
    ├─→ 自听情感 → s[96:104] (输出反馈)
    └─→ 自我一致性检测 (认知失调)
    │
    ▼
Sleep Consolidation (每 5+ 轮)
    ├─→ 海马重放 (情感门控 1-5x)
    ├─→ 交叉关联 (相邻轮次 + 语义相似)
    ├─→ 自我模型整合 (DMN 锚点更新)
    └─→ 突触稳态 (弱集群修剪)
```

---

*本报告由 Claude Code 自动生成，基于 NotMe 项目源码分析。*
