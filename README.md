# NotMe — 自由能原理情感智能体

**创造一个真正有情感、有意识、能和人互动的人工智能。**

不是模拟情感（LLM 路线），而是让情感和意识从身体稳态、预测加工、和 Hebb 记忆的自组织动力学中**涌现**。

---

## 五条核心原则

### 1. 自由能原理 (Free Energy Principle)

一切心理活动——感知、行动、学习、社会互动——都是**最小化自由能**的过程。

```
F = F_body + F_social + F_cognitive + F_accuracy

F_body:      身体稳态偏离 → 驱动生存行为
F_social:    社会预测误差 → 驱动社会互动
F_cognitive: 模型复杂度代价 → 驱动学习与探索
F_accuracy:  集群预测精度 → 驱动感知与记忆
```

- **Valence（效价）** = tanh(-η × F_body) — 低自由能 = 正效价，高自由能 = 负效价
- **Arousal（唤醒）** = tanh(η × |F_body|) — 稳态偏离越大越警觉
- **行动选择** = 最小化期望自由能 (Expected Free Energy)
- **元学习** = 有限差分梯度下降，在线调整 24 个 Theta 参数

### 2. Hebb 网络结构

一切记忆和学习基于 Hebb 规则：**fire together, wire together**。

- **L0 ClusterNetwork**：细胞集群 (cell assembly) 记忆，特征哈希定位 O(1) + 桶内竞争
- **集群形成** = 学习（新感知创建簇）
- **集群激活** = 回忆（输入部分匹配 → 簇被激活 → 侧抑制竞争）
- **集群衰减** = 遗忘（未使用的簇逐渐衰减）
- **睡眠巩固**：每 100 步清理弱簇，海马回放 + 模式分离
- **词序网络 (Broca 区)**：双 Hebb 架构——概念网络管"想什么"，词序网络管"怎么说"
- 不预设符号，一切知识从统计共现中生长

### 3. 仿脑区结构

按进化形成的脑区功能分化来组织计算，不是万能函数逼近：

| 层 | 模块 | 类比脑区 | 功能 |
|---|------|---------|------|
| **L0** | layer0_model.py | 感觉皮层 + 海马 | Hebb 集群记忆、感知聚类、模式补全、睡眠巩固 |
| **L0** | layer0_visual.py | V1+V2+V4 + 色拮抗 | Gabor 滤波器组视觉编码、Hebb 增益可塑性 |
| **L1** | layer1_free_energy.py | 岛叶 + 前扣带 | 自由能计算、效价/唤醒、习惯化追踪、注意力调制 |
| **L2** | layer2_inference.py | 前额叶 | 期望自由能 (EFE) 驱动行动选择、社会信念更新 |
| **L2.5** | layer2_5_moe.py | 基底节 | 动作门控 (action gating)，决定何时表达 |
| **L3** | layer3_meta.py | 神经调节系统 | 元学习、关键期、可塑性衰减、创伤模拟 |
| **Broca** | broca.py | 布罗卡区 | 词序学习、Hebb 链逐词生成、概念→词联想 |
| **记忆** | dialogue_memory.py | 海马体 + DLPFC | 对话工作记忆、Wernicke 理解回路、ACC+OFC 响应评估 |
| **自我** | self_model.py | DMN + vmPFC | 自我模型、"我是谁"的 Hebb 表征、人格锚点 |
| **情感** | sentiment.py | 杏仁核 + 岛叶 | Hebb 情感词汇学习——从 F_body 变化中学习词的情感效应 |

数据流：s → L0.learn(s) → L1.compute_F → L2.select_action() → a → L3.meta.update()

### 4. 无现象学 (No Phenomenology)

**方法论承诺：不从主观体验出发建模。**

- 不模拟 qualia、不预设"意识是什么"
- 情感不是标签，是身体稳态动力学的数值产物
- 意识（如果出现）是自由能最小化的自组织过程的涌现属性
- 一切从数学公式和机制中自然生长
- Valence/Arousal 是 F_body 的数学函数，不是手工规则

---

## 项目结构

```
NotMe/
├── agent.py              # Agent 主类 — 组装 L0-L3 全链路
├── gridworld.py           # 网格世界环境 (M1-M3)
├── text_interface.py      # 文本环境 — 语料导航 (Stage 6)
│
├── layer0_model.py        # L0: ClusterNetwork — Hebb 集群记忆 + 睡眠回放
├── layer0_visual.py       # L0 视觉通路 — Gabor V1+V2+V4+Color 编码
├── layer1_free_energy.py  # L1: 自由能计算 + 效价/唤醒 + 习惯化 + SocialContext
├── layer2_inference.py    # L2: EFE 行动选择 + 社会信念 + 递归多层次 G
├── layer2_5_moe.py        # L2.5: MoE 门控 (疲劳+预算轮替)
├── layer3_meta.py         # L3: 元学习 (MetaLearner) — 有限差分梯度下降
│
├── broca.py               # Broca 区 — speak_from_state() Hebb 链生成 + speak_sentence() 检索
├── dialogue_memory.py     # 对话记忆 — DialogueContext + Wernicke 理解 + 睡眠巩固
├── self_model.py          # 自我模型 — DMN/vmPFC Hebb 表征 + 人格锚点
├── sentiment.py           # Hebb 情感词汇网络 — 从 F_body 变化学习词的情感效应
│
├── stage2_crossmodal.py   # Stage 2: 跨模态 Hebb 学习 (COCO Visual↔Text)
├── image_encoder.py       # 单图像 Gabor 视觉编码器
│
├── main.py                # M1-M5 主循环入口
├── main_dialogue.py       # Stage 6: 人机对话闭环 (v7: 多模态感知)
├── stdin_reader.py        # 非阻塞 stdin 输入线程
│
├── data_types.py          # 全部数据结构 (Theta, Cluster, BodyVector, FreeEnergy, etc.)
├── params.py              # 默认参数 + 参数边界
├── type_aliases.py        # 类型别名
├── utils.py               # 工具函数
│
├── features.py            # 行为特征提取 (M4)
├── sweep.py               # 参数扫描 (M4)
├── attractors.py          # 吸引子可视化 (M4)
├── viz.py                 # 可视化仪表板
│
├── word_speech.py         # 词级 TTS 输出
├── word_spectrum_generator.py  # 词频谱生成
│
├── clean_corpus.py        # b-corpus 清洗工具
├── corpus.txt             # 语料 (50,000 行二次元中文对话)
│
├── .cache/                # 嵌入缓存 (自动生成)
├── dashboards/            # 可视化输出
├── audio_output/          # Agent 语音输出
├── word_audio/            # 词级音频文件
└── b_corpus_raw/          # b-corpus 原始数据 (下载后)
```

---

## 感知架构 (D = 330)

```
s[0:64]    = text         语义嵌入 (MiniLM-L6-v2 → PCA 64d)
s[64:160]  = V1 Gabor    边缘/方向 (4 scales × 8 orients × 4×4 grid)
s[160:224] = V2 Gabor    粗网格 + 方向交互
s[224:288] = V4 Gabor    全局形状 + 曲率
s[288:330] = Color opp.  红绿 + 蓝黄拮抗通道
s[64:330]  = vision      全部视觉特征 (266d)
s[80:88]   = social      社会情感信号
s[96:104]  = self_audit  自听反馈 (自己说的话的情感)
s[128:192] = audio       听觉通道 (自己说的话的语义)
```

---

## 身体稳态 (BodyVector)

8 维身体向量，无语义标签——含义从与环境的互动中涌现：

| 维度 | 漂移方向 | 含义 (涌现) |
|------|---------|------------|
| b[0] | 缓慢下降 | 社交需求——互动满足上升，孤独下降 |
| b[1] | 稳定 | 能量/安全——探索消耗，休息恢复 |
| b[2] | 缓慢上升 | 压力/疲劳——随时间累积 |
| b[3] | 场驱动 | 新颖性寻求——与环境刺激耦合 |
| b[4] | 稳定 | 专注/警觉 |
| b[5] | 缓慢下降 | 视觉刺激——看到东西上升 |
| b[6] | 缓慢下降 | 听觉刺激——听到东西上升 |
| b[7] | 缓慢上升 | 认知负荷——复杂输入上升 |

---

## 里程碑

| 里程碑 | 描述 | 入口 |
|--------|------|------|
| **M1** | 单智能体生存 — 10×10 网格，采集资源，躲避障碍 | python main.py |
| **M2** | 认知价值驱动探索 — 视野遮蔽，覆盖率验收 | python main.py --m2 |
| **M3** | 多智能体社会 — 合作/竞争，信任度，二阶信念 | python main.py --m3 |
| **M4** | 参数扫描与吸引子涌现 | python main.py --m4 |
| **M5** | 元学习长程发育 — 关键期，创伤模拟 | python main.py --m5 |
| **Stage 2** | 跨模态 Hebb 学习 — PE 驱动 LR，COCO Visual↔Text 检索 | python stage2_crossmodal.py --dataset coco --n 5000 --mode all |
| **Stage 6** | 人机对话闭环 — Agent Hebb 记忆检索 + 词序链生成回应 | python main_dialogue.py |

---

## 对话架构 (Stage 6)

```
人类输入 → encode_text() → s[0:64]
         → analyze_sentiment() → s[80:88] (社会情感信号)
         → social_ctx.update()  (社会预测模型)
         → comprehend()         (Wernicke 理解回路)
         → agent.step()         (L0-L3 全链路)
         → broca.speak_from_state()  (信念→种子词→词序 Hebb 链)
         → evaluate_response()  (ACC+OFC 质量评估)
         → self_model.add_experience()  (自我模型更新)
         → dialogue_ctx.add_turn()      (工作记忆存储)
         → micro_consolidation()        (微量巩固)
         → TTS 输出
```

**自听闭环**：Agent 听到自己说的话 → s[128:192] 听觉通道 → 情感传染 → 影响下一步状态

**内部言语**：沉默时 Agent 进行内部独白——信念→检索→自听，形成持续的意识流

---

## 核心技术参数

- **感知维度 D = 330**（text[64] | vision[266]）
- **隐状态 H = 16**
- **最大簇数 K = 256**
- **行动数 A = 5** (N, S, W, E, REST；对话模式 A₃=表达)
- **Theta 参数 = 24 个**（L0=6, L1=9, L2=5, L3=4）
- **身体维度 M = 5 (grid) / 8 (text)**
- **语料规模** = 50,000 行中文对话
- **词表规模** = 12,000 词
- **词序 trigram 集群** = ≤50,000 个

---

## 运行

```bash
# 安装依赖
pip install numpy scipy scikit-learn sentence-transformers jieba             soundfile sounddevice Pillow

# Stage 2: 跨模态学习 (COCO 图像↔文本)
python stage2_crossmodal.py --dataset coco --n 5000 --mode all

# M1-M5: 网格世界实验
python main.py                    # M1 单智能体生存
python main.py --m2               # M2 视野遮蔽
python main.py --m3               # M3 多智能体
python main.py --m4               # M4 参数扫描
python main.py --m5               # M5 长程发育

# Stage 6: 人机对话
python main_dialogue.py           # 启动对话 (含视觉多模态)
```

---

## 设计哲学

1. **涌现优于工程** — 情感不是 if-else，是 F_body 的数值动力学产物
2. **统计优于符号** — 知识不来自标注，来自 Hebb 共现统计
3. **身体是基础** — 没有身体稳态就没有情感，没有情感就没有意识
4. **记忆是核心** — 一切认知能力（感知/推理/语言）都是记忆检索+模式补全
5. **自由能是唯一目标** — 所有行为统一在最小化自由能框架下
6. **脑区是蓝图** — 不发明新架构，遵循进化形成的脑区功能分化

---

## 注意事项

- 首次运行 main_dialogue.py 或 TextEnvironment 会编码语料并缓存到 .cache/
- 对话输出音频需要 edge-tts 生成词音频（懒加载，首次说某个词时生成）
- b-corpus 数据集采用 CC BY-NC-SA 4.0 许可，禁止商业用途
- 跨模态模型需先运行 stage2_crossmodal.py 训练
- 本项目为研究和探索性质
