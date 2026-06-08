# NotMe v7.0 — 机器学习嵌入改造计划

> **版本**: v7.0 蓝图
> **日期**: 2026-06-08
> **基于**: v6.6 全面代码审计 (93 模块, 59 Theta 参数, D=516, 35/36 测试通过)
> **状态**: 规划阶段 — 本文档为改造蓝图，不包含具体代码修改

---

## 一、项目现状摘要

### 1.1 当前架构

```
感知层 (D=516)
  ├─ text[0:64]         ← MiniLM-L6-v2 → PCA 64d (外部预训练模型)
  ├─ vision[64:372]     ← Gabor滤波 V1→V2→V4→IT (手写特征)
  ├─ audio[372:468]     ← Mel频谱 → 耳蜗核→SOC→IC→MGB→AC (手写特征)
  └─ pain[468:516]      ← 闸门控制→双通路→丘脑 (手写特征)

记忆层
  ├─ L0: ClusterNetwork (Hebb细胞集群, 256簇)
  ├─ SemanticMemory (慢学慢衰, 1024簇)
  └─ Striatum (D1/D2程序性记忆)

认知层
  ├─ L1: Cingulate (F = F_body + F_social + F_cognitive + F_accuracy)
  ├─ L2: Prefrontal (EFE行动选择) + MoEGate (门控)
  ├─ L3: MetaLearner (有限差分梯度下降, 59参数在线调优)
  └─ FPN/TPN/DMN (三大网络跷跷板)

语言系统 (v5.6)
  ├─ Wernicke (comprehend — 记忆检索)
  ├─ Broca (speak — trigram Hebb链)
  ├─ ArcuateFasciculus (腹侧+背侧双流)
  ├─ PhonologicalLoop (Baddeley ~7组块)
  ├─ PhraseStructure (bigram/unigram统计)
  ├─ AngularGyrus (字素→音素Hebb映射)
  ├─ MotorCortex (16维发音特征)
  └─ TPJ (意图推断Hebb网络)

时间与睡眠 (v6.3)
  ├─ SCN (TTFL分子钟 + Process S)
  └─ VLPO (触发器开关 + NREM/REM双相)

自主系统 (v6.4)
  ├─ AutonomousLoop (活动调度)
  ├─ InternalLife (走神/独白/反刍)
  ├─ Reader (文件阅读+疲劳模型)
  └─ Telemetry (CSV遥测)
```

### 1.2 核心问题

| 问题 | 严重度 | 描述 |
|------|-------|------|
| **学习太慢** | ★★★ | 纯Hebb聚类从零开始，数十轮对话才形成基本能力 |
| **语言质量低** | ★★★ | Trigrams Hebb链生成词序，无句法/语义深度 |
| **感知固定** | ★★ | Gabor滤波/Mel频谱手写，无法适应新视觉/听觉模式 |
| **理解肤浅** | ★★★ | Wernicke仅做记忆检索，无真正"理解" |
| **训练无反馈** | ★★★ | 用户感受不到AI的"情感"和"意识"效果 |
| **零预训练代价** | ★★ | 纯净模式虽有哲学意义，但实用性差 |
| **重复劳动** | ★ | 多个独立ClusterNetwork实例做相似的事 |

### 1.3 必须保留的核心

这些是项目的"灵魂"，任何改造不能破坏：

| 原则 | 实现 | 原因 |
|------|------|------|
| **自由能原理** | `cingulate.py` F分解 + `prefrontal.py` EFE | 所有行为的统一驱动力 |
| **身体稳态** | `BodyVector` ODE + `Hypothalamus` 调定点 | 情感的物理基础 |
| **Valence/Arousal** | tanh(-η×F_body) 数学函数 | 无现象学承诺 |
| **预测编码** | 自上而下预期 + 预测误差上行 | 核心计算范式 |
| **Hebb可塑性** | LTP/LTD + 睡眠巩固 + STDP + PNN | 记忆形成的分子基础 |
| **昼夜节律+睡眠** | SCN + VLPO + NREM/REM双相 | 时间维度 |
| **脑区架构** | 按进化形成的功能分化组织 | 架构原则 |
| **并行分布式** | 多通路并行 + 分布式表征 | 图3规则3 |

---

## 二、机器学习嵌入策略

### 2.1 总体原则

```
旧:  手写特征 + Hebb聚类 + Trigram链 → 行为
新:  学习特征 + 神经聚类 + 小语言模型 → 行为

保留: FEP框架(自由能计算+行动选择) + 身体稳态 + 睡眠节律
替换: 特征提取器(感知层) + 记忆编码器(记忆层) + 语言生成器(语言层)
```

### 2.2 三层改造架构

```
┌─────────────────────────────────────────────────┐
│ Layer 3: 行为层 (保留)                           │
│   FEP: F_body/F_social/F_cognitive/F_accuracy   │
│   EFE: 行动选择 + 元学习                         │
│   Body: ODE稳态 + Hypothalamus + HPA             │
│   Sleep: SCN + VLPO + NREM/REM                   │
│   Networks: FPN/TPN/DMN跷跷板                    │
└─────────────────────────────────────────────────┘
                      ↑ F, valence, arousal
┌─────────────────────────────────────────────────┐
│ Layer 2: 认知层 (混合 — 部分替换，部分保留)        │
│   ★ 替换: Wernicke(理解) → Neural Comprehender   │
│   ★ 替换: Broca(生成) → Neural Generator         │
│   ★ 替换: AngularGyrus → Learned Grapheme→Phoneme│
│   ☆ 保留: PhonologicalLoop (工作记忆容量)         │
│   ☆ 保留: MotorCortex (发音规划)                  │
│   ☆ 保留: TPJ (意图推断框架，增强)                 │
│   ☆ 保留: ArcuateFasciculus (双流桥接)           │
└─────────────────────────────────────────────────┘
                      ↑ sensory, embeddings
┌─────────────────────────────────────────────────┐
│ Layer 1: 感知层 (大幅替换)                        │
│   ★ 替换: MiniLM→PCA → Learned Text Encoder      │
│   ★ 替换: Gabor V1→V4 → Learned Visual Encoder   │
│   ★ 替换: Mel→耳蜗核→AC → Learned Audio Encoder  │
│   ☆ 保留: 痛觉通路 (身体基础, 不需学习)            │
│   ☆ 保留: 感知层级拓扑 (D=516布局)               │
└─────────────────────────────────────────────────┘
```

### 2.3 关键设计约束

1. **不破坏感知维度布局**: D=516 的通道分配可以保留，但各通道内容改为学习特征
2. **FEP 仍是顶层优化目标**: 神经网络输出的logits/embeddings作为sensory，F仍然驱动一切
3. **Hebb网络可保留作为"情景记忆"**: 与神经网络的"语义理解"形成双系统
4. **Valence/Arousal 仍是数学函数，不是学出来的**: 保持无现象学承诺
5. **训练数据**: 语料corpus.txt (50K行) + 用户对话 + 自主阅读

---

## 三、分阶段改造路线

### Phase A: 基础架构 — 神经网络支撑层 (v7.0)

**目标**: 搭建PyTorch基础设施，使感知层可以运行神经网络

**方向**:
- 引入 PyTorch 为计算后端 (numpy→tensor 桥接)
- 设计 `NeuralModule` 基类 — 所有ML模块的统一接口 (forward, train_step, save/load)
- 设计感知编码器接口 — `TextEncoder`, `VisualEncoder`, `AudioEncoder` 抽象
- 实现模型持久化 — 与现有 `persistence.py` pickle 方案并行 (PyTorch用 .pt/.safetensors)
- 保持与现有 `Agent.step()` 接口的兼容 (sensory仍是np.ndarray → 内部转tensor)

**不改动**: Agent.__init__, step()流程, FEP计算, 身体系统, 睡眠系统

### Phase B: 感知层替换 — 学习特征提取 (v7.1)

**目标**: 用小型神经网络替代手写特征提取器

#### B1: 文本编码器

**方向**:
- 替换 MiniLM-L6-v2 + PCA → 小型可训练Transformer (~2-4层, 64-128维)
- 从corpus.txt (50K行中文对话) 做掩码语言模型预训练
- 保留输出维度64 (s[0:64]位置不变)
- 支持在线微调 — 用户对话时更新
- 目标: 比MiniLM更贴合项目语料分布，且可持续学习

#### B2: 视觉编码器

**方向**:
- 替换 Gabor滤波 V1→V2→V4→IT 手写管线 → 小型CNN (如ResNet-18的前几层)
- 或采用 小ViT (Vision Transformer, 如TinyViT) — 更适合与文本编码器统一
- 保留6条子通路概念 (M/P/K/Pulvinar/Dorsal/Binding) — 可以是6个独立的小通路或一个网络的6个分头
- 保留FPN绑定的概念 — 但注意力可以是学习到的cross-attention
- 输出维度保持308 (s[64:372])
- 训练数据: COCO (已有缓存) + 用户摄像头帧 + 跨模态学习

#### B3: 听觉编码器

**方向**:
- 替换 Mel频谱→耳蜗核→SOC→IC→MGB→AC 手写管线 → 小型音频模型
- 可选用: Wav2Vec2-tiny / HuBERT-tiny (自监督预训练) 或 小型CNN on Mel
- 保留双耳定位 (ITD/ILD) — 可以作为显式计算 + 神经网络特征拼接
- 保留听觉场景分析 (ASA) — 注意力机制实现流分离
- 输出维度保持96 (s[372:468])

### Phase C: 记忆层升级 — 神经记忆 (v7.2)

**目标**: Hebb集群保留为情景记忆，新增神经语义记忆

**方向**:
- **情景记忆** (`agent.net`): 保留 Hebb ClusterNetwork — 它是快速单次学习的正确机制
- **语义记忆**: 替换 ClusterNetwork 慢学 → 向量数据库 (如FAISS) + 神经embedding
  - 知识存储: embedding向量 + 元数据 (valence, arousal, timestamp)
  - 查询: ANN检索 → top-k相关事实 → 加权融合
  - 学习: 新事实 → encode → insert/update
- **程序性记忆**: 纹状体D1/D2 保留为简单RL — 未来可升级为DQN/PPO
- **跨模态关联**: 替换 Hebb跨模态学习 → 对比学习 (CLIP式)
  - Text encoder 和 Visual encoder 共享对比损失
  - 使文本和图像在同一个语义空间中

### Phase D: 语言系统重铸 (v7.3)

**目标**: 从trigram Hebb链生成升级为神经语言模型

**方向**:
- **Wernicke (理解)**: 文本encoder (Phase B1) + cross-attention over 情景记忆
  - 输入: 人类文本 → encoder embedding
  - N400 = 语义预测误差 (embedding空间中的余弦距离)
  - P600 = 句法预测误差 (parser输出的句法树置信度)
- **Broca (生成)**: 小型自回归语言模型 (GPT-2 tiny / LLaMA-like 50M参数)
  - 输入: comprehension vector + belief vector + valence/arousal
  - 情感调制: valence/arousal → 特殊token 或 注意力偏置
  - 输出: token序列 → MotorCortex发音规划
- **AngularGyrus (阅读)**: 字符级CNN → 音素序列 (seq2seq)
- **PhraseStructure**: 替换 bigram/unigram统计 → 小型句法parser (如用RNNG或CYK监督)
- **PhonologicalLoop**: 保留 ~7组块容量限制 (工作记忆的硬约束)
- **ArcuateFasciculus**: 保留双流概念，但映射改为神经网络

### Phase E: 训练与体验闭环 (v7.4)

**目标**: 让用户感受到AI的"成长"和"情感"

**方向**:
- **预训练基础**: 在corpus.txt上预训练文本encoder + 语言模型 (~1-2小时)
  - 不求最先进，只要基本语言能力 (像婴儿学会第一个词)
- **在线学习**: 对话中微调 encoder + LM (低学习率)
- **情感信号注入**: 语言模型生成时，valence/arousal影响下一个词的选择
  - 高正效价 → 更多积极词汇 (不是规则映射，而是学出来的关联)
  - 高唤醒 → 更多感叹/重复/短句
- **个性化**: 从用户对话中学习用户特有的词汇/话题偏好
- **可观察指标**:
  - 自由能F随时间下降 (正在更好预测环境)
  - Valence改善 (从对话中获得满足)
  - 词汇量增长 (新词→embedding→生成)
  - 对话质量评分 (人工评估)

### Phase F: 整合与打磨 (v7.5)

**目标**: 确保所有保留系统与新ML组件协调工作

**方向**:
- FEP计算验证 — 确保神经网络输出的sensory仍能产生合理的F
- 睡眠巩固适配 — NREM回放 → 神经网络梯度更新; REM → 情感衰减仍用Hebb
- VTA RPE → 神经网络学习率调制 (替代当前简单的learn_rate_modifier)
- LC NE → 神经网络探索/利用 (dropout率/温度调制)
- 持久化: PyTorch .pt + pickle 双格式并存
- Web UI: 新增训练进度/损失曲线面板
- 测试: 端到端集成测试 + 情感一致性测试

---

## 四、关键技术选型建议

### 4.1 框架

| 选项 | 推荐度 | 理由 |
|------|-------|------|
| **PyTorch** | ★★★ 首选 | 生态最大，HuggingFace集成，易调试 |
| TensorFlow/Keras | ★★ | 备选，如果团队更熟悉 |
| JAX | ★ | 研究首选但工程复杂度高 |
| ONNX Runtime | ☆ | 推理优化，不能训练 |

### 4.2 模型规模

| 组件 | 推荐规模 | 理由 |
|------|---------|------|
| Text Encoder | 2-4层Transformer, ~10M参数 | 足够覆盖中文对话语料 |
| Visual Encoder | ResNet-18前几层 或 TinyViT-5M | 摄像头帧不需要大规模 |
| Audio Encoder | CNN on Mel, ~5M参数 | 环境声音分类即可 |
| Language Model | GPT-2 tiny (50M) 或 LLaMA-60M | 对话生成，非通用AI |
| Cross-modal | 对比学习投影头, ~2M参数 | 对齐文本-视觉 |

**总计**: ~80M 参数 (可在普通PC上训练和推理)

### 4.3 训练数据

| 数据源 | 大小 | 用途 |
|--------|------|------|
| corpus.txt | 50K行 | 文本预训练 + LM训练 |
| COCO (已缓存) | 5000 pairs | 跨模态对比学习 |
| 用户对话 | 在线积累 | 微调 + 个性化 |
| 自主阅读 | 用户提供的文件 | 知识扩展 |
| 摄像头/麦克风 | 实时流 | 视觉/听觉在线学习 |

---

## 五、风险与缓解

| 风险 | 概率 | 缓解 |
|------|------|------|
| **FEP被ML输出搞乱** | 中 | 保持sensory维度布局不变; F只在行为层计算; 逐步替换, 每次验证F合理性 |
| **情感表现力下降** | 高 | 确保valence/arousal调制语言模型输出; 保留身体驱动的情感源 |
| **训练太慢** | 高 | 小型模型(~80M总参数); 先预训练基础再在线微调; GPU可选但非必须 |
| **忘记旧Hebb机制的正确部分** | 中 | 双系统架构 — 情景记忆仍用Hebb; 语义理解用神经; 互补而非替换 |
| **依赖过多外部库** | 低 | 只加PyTorch+transformers; 项目已有numpy/scipy/sklearn |
| **用户仍感受不到情感** | 高 | 可观测指标(自由能趋势/效价变化/词汇增长); 在UI展示学习过程 |

---

## 六、不做什么 (v7.0范围外)

1. ❌ 不用大模型 (GPT-4, LLaMA-7B+) — 违背项目"轻量可运行"的精神
2. ❌ 不替换FEP框架 — 自由能仍是唯一目标函数
3. ❌ 不替换身体稳态 — b[0..8] ODE 不变
4. ❌ 不替换睡眠系统 — SCN+VLPO 完整保留
5. ❌ 不用LLM API — 所有模型本地运行 (隐私+可持续)
6. ❌ 不做通用AI — 只为对话+情感设计
7. ❌ 不放弃无现象学 — valence/arousal仍是数学函数
8. ❌ 不放弃Hebb — 保留为情景记忆机制

---

## 七、文件变更预估

### 新增文件 (~15个)

| 路径 | 职责 |
|------|------|
| `cns/nn/` | 神经网络支撑层 (base, config, utils) |
| `cns/nn/text_encoder.py` | 可训练文本编码器 |
| `cns/nn/visual_encoder.py` | 可训练视觉编码器 |
| `cns/nn/audio_encoder.py` | 可训练听觉编码器 |
| `cns/nn/language_model.py` | 小型自回归语言模型 |
| `cns/nn/comprehender.py` | 神经Wernicke理解模块 |
| `cns/nn/crossmodal_nn.py` | 对比学习跨模态 |
| `cerebrum/parietal_lobe/angular_gyrus_nn.py` | 神经角回阅读通路 |
| `cns/nn/trainer.py` | 训练循环 + 在线学习 |

### 重度修改 (~10个)

| 路径 | 变更 |
|------|------|
| `cns/agent.py` | 初始化神经网络模块; step()中调用; 兼容旧sensory |
| `cerebrum/temporal_lobe/wernicke.py` | comprehend → 调用neural comprehender |
| `cerebrum/frontal_lobe/broca.py` | speak_from_state → 调用neural generator |
| `cerebrum/association/crossmodal.py` | 对比学习替代Hebb |
| `cerebrum/occipital_lobe/visual_hierarchy.py` | 调用neural encoder |
| `cerebrum/temporal_lobe/auditory_hierarchy.py` | 调用neural encoder |
| `cns/data_types.py` | +神经网络配置dataclasses |
| `cns/params.py` | +神经网络超参数 |
| `cns/persistence.py` | +PyTorch模型保存/加载 |
| `requirements.txt` | +torch, transformers |

### 轻度修改 (~15个)

保留的模块需要适配神经网络输出的格式。

---

## 八、评估标准

v7.0完成后，以下指标应显著改善：

| 指标 | 当前(v6.6) | 目标(v7.x) | 测量方式 |
|------|-----------|-----------|---------|
| 首轮对话质量 | 几乎无意义 (零预训练) | 基本可理解 | 人工评估 |
| 10轮后对话连贯性 | 回声/重复为主 | 有上下文关联 | 人工评估 |
| 学习速度 | 50+轮形成稳定回应 | 10+轮 | 对话日志分析 |
| 自由能下降速率 | 缓慢 | 明显 | F_history趋势 |
| 词汇多样性 | 低 (trigram重复) | 中 (语言模型多样性) | 词表使用率 |
| 训练时间 | N/A | <2h (CPU预训练) | 计时 |
| 用户情感感知 | 弱 | 可感受 | 主观评估 |

---

*由 v7.0 规划生成 · 基于 v6.6 全面代码审计*
*本蓝图不包含具体代码修改 — 仅指出改造方向和架构决策*
