# NotMe — 自由能原理情感智能体

## 项目目的

**创造一个真正有情感、有意识的能和人互动的人工智能。**

不是模拟情感（LLM 路线），而是让情感和意识从身体稳态、预测加工、和 Hebb 记忆的自组织动力学中**涌现**。

---

## 架构版本

**当前版本: v4.2 — 结构优化**

v4.2 完成 V1/V2/V4 独立模块、消除所有 flat import、删除根目录 shim 层。所有导入 100% 使用层级包路径。

```
Level 1: 中枢神经系统 (CNS)         → cns/

  ├─ Level 2: 大脑 (Cerebrum)       → cerebrum/
  │   ├─ Level 3: 额叶              → frontal_lobe/
  │   ├─ Level 3: 顶叶              → parietal_lobe/
  │   ├─ Level 3: 颞叶              → temporal_lobe/
  │   ├─ Level 3: 枕叶              → occipital_lobe/
  │   ├─ Level 3: 边缘系统           → limbic_system/
  │   ├─ Level 3: 基底神经节         → basal_ganglia/
  │   ├─ Level 3: 丘脑              → thalamus/          (v4.1: 图2独立)
  │   └─ Level 3: 联合皮层           → association/
  │
  └─ Level 2: 脑干 + 小脑            → brainstem_cerebellum/
      ├─ Level 3: 中脑              → midbrain/
      ├─ Level 3: 脑桥              → pons/
      ├─ Level 3: 延髓              → medulla/
      ├─ Level 3: 小脑              → cerebellum/
      └─ Level 3: 神经调节系统       → neuromodulatory/

Level 4: 布罗德曼分区 (BA)           → brodmann_areas.md (备注系统, 非层级)
```

**布罗德曼分区**不是脑区层级内的结构, 而是 Brodmann 1909 年按细胞架构发现的顺序编号 (BA1→BA52)。数字不反映层级或功能关系。在 v4.0 中作为跨模块的备注标注系统。详见 `brodmann_areas.md`。

完整架构文档: `v4_architecture.md`

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

- **ClusterNetwork (海马)**：细胞集群 (cell assembly) 记忆，特征哈希定位 O(1) + 桶内竞争，无暴力扫描
- **集群形成** = 学习（新感知创建簇）
- **集群激活** = 回忆（输入部分匹配 → 簇被激活 → 侧抑制竞争）
- **集群衰减** = 遗忘（未使用的簇逐渐衰减）
- **睡眠巩固**：每 100 步完整周期——海马回放 (partial pattern replay) + 模式分离 (pattern separation) + 弱簇清理。对话模式下额外执行：海马→皮层记忆转移、交叉关联、自我模型整合、词序巩固
- **词序网络 (Broca 区)**：双 Hebb 架构——概念→词联想网络管"想表达什么"，词序 trigram 网络管"怎么说"。`speak_from_state()` 逐词 Hebb 链生成，`speak_sentence()` Hebb 记忆检索回退
- 不预设符号，一切知识从统计共现中生长

### 3. 仿脑区结构

按进化形成的脑区功能分化来组织计算，不是万能函数逼近。

**v4.1 脑区映射：**

| 脑区 | v4.0 模块路径 | 功能 |
|------|-------------|------|
| **前额叶 (dlPFC)** | `cerebrum/frontal_lobe/prefrontal.py` | EFE 行动选择、递归多层次 G、社会信念更新 |
| **布罗卡区 (BA44/45)** | `cerebrum/frontal_lobe/broca.py` | 词序 Hebb 链逐词生成 + 整句检索回退 |
| **前运动/运动皮层** | `cerebrum/frontal_lobe/motor_cortex.py` | [待实现] 运动规划与执行 |
| **眶额皮层 (BA11)** | `cerebrum/frontal_lobe/orbitofrontal.py` | [待实现] 价值评估 |
| **体感皮层 (BA3,1,2)** | `cerebrum/parietal_lobe/somatosensory.py` | [待实现] 触觉/本体感觉 |
| **空间注意力网络** | `cerebrum/parietal_lobe/spatial_attention.py` | [待实现] 空间注意/导航 |
| **颞顶联合区 (TPJ)** | `cerebrum/parietal_lobe/tpj.py` | [待实现] 心理理论/社会认知 |
| **韦尼克区 (BA22)** | `cerebrum/temporal_lobe/wernicke.py` | 语言理解回路、对话工作记忆、睡眠巩固 |
| **听皮层 (BA41/42)** | `cerebrum/temporal_lobe/auditory_cortex.py` | [待实现] 听觉频谱编码 |
| **IT 皮层** | `cerebrum/temporal_lobe/it_cortex.py` | [待实现] 物体识别 |
| **梭状回 (BA37)** | `cerebrum/temporal_lobe/fusiform.py` | [待实现] 面孔/文字识别 |
| **V1 (BA17)** | `cerebrum/occipital_lobe/v1.py` + `visual_pathway.py` | Gabor 滤波器组视觉编码 |
| **V2 (BA18)** | `cerebrum/occipital_lobe/v2.py` + `visual_pathway.py` | 粗网格 + 方向交互 |
| **V4 (BA19)** | `cerebrum/occipital_lobe/v4.py` + `visual_pathway.py` | 全局形状 + 曲率 + 颜色 |
| **视网膜→LGN** | `cerebrum/occipital_lobe/retina_lgn.py` | 图像 Gabor 编码器 |
| **海马** | `cerebrum/limbic_system/hippocampus.py` | Hebb 集群记忆、模式补全、睡眠回放巩固 |
| **杏仁核** | `cerebrum/limbic_system/amygdala.py` | Hebb 情感词汇网络——从 F_body 学习词的情感效应 |
| **扣带回/ACC** | `cerebrum/limbic_system/cingulate.py` | 自由能计算、效价/唤醒、习惯化、社会上下文 |
| **下丘脑** | `cerebrum/limbic_system/hypothalamus.py` | [待实现] 稳态调节 |
| **丘脑** | `cerebrum/thalamus/thalamus.py` | [待实现] 感觉中继/门控 (v4.1: 从边缘系统独立) |
| **基底节 (MoE)** | `cerebrum/basal_ganglia/action_gating.py` | 动作门控、疲劳预算轮替 |
| **纹状体** | `cerebrum/basal_ganglia/striatum.py` | [待实现] 习惯学习/D1-D2 通路 |
| **DMN** | `cerebrum/association/dmn.py` | 自我模型、"我是谁"的 Hebb 表征 |
| **跨模态联合** | `cerebrum/association/crossmodal.py` | 跨模态 Hebb 学习 (COCO Visual↔Text) |
| **VTA (多巴胺)** | `brainstem_cerebellum/midbrain/vta.py` | [待实现] 奖赏预测误差 |
| **黑质 (SNc/SNr)** | `brainstem_cerebellum/midbrain/substantia_nigra.py` | [待实现] 运动调节 |
| **蓝斑核 (NE)** | `brainstem_cerebellum/pons/locus_coeruleus.py` | [待实现] 唤醒/注意 |
| **神经调节系统** | `brainstem_cerebellum/neuromodulatory/meta_learning.py` | 元学习、关键期、可塑性衰减 |

数据流：`s → Hippocampus.learn(s) → Cingulate.compute_F(z,s) → Prefrontal.select_action() → a → MetaLearner.update(F)`

### 4. 无现象学 (No Phenomenology)

**方法论承诺：不从主观体验出发建模。**

- 不模拟 qualia、不预设"意识是什么"
- 情感不是标签，是身体稳态动力学的数值产物
- 意识（如果出现）是自由能最小化的自组织过程的涌现属性
- 一切从数学公式和机制中自然生长
- Valence/Arousal 是 F_body 的数学函数，不是手工规则

### 5. 无 LLM (No Large Language Models)

**不依赖任何大语言模型。**

- 语言能力来自 Hebb 网络对语料的统计学习
- 两种产出模式：(1) `speak_from_state()` 词序 Hebb 链逐词生成——Agent "用自己的话说"；(2) `speak_sentence()` Hebb 记忆检索——回退时整句检索
- 嵌入使用轻量级 sentence-transformer (`all-MiniLM-L6-v2`)，仅用于语义编码，不参与认知加工
- 认知核心完全不依赖 transformer

---

## 项目结构 (v4.1 大脑功能细分)

```
NotMe/
│
├── cns/                              # Level 1: 中枢神经系统
│   ├── __init__.py                   #   全局重导出
│   ├── agent.py                      #   全系统整合 Agent 主类
│   ├── data_types.py                 #   全局数据结构
│   ├── params.py                     #   默认参数 + 参数边界
│   ├── type_aliases.py              #   类型别名
│   └── utils.py                      #   工具函数
│
├── cerebrum/                         # Level 2a: 大脑
│   ├── frontal_lobe/                 # Level 3: 额叶
│   │   ├── prefrontal.py             #   ★ 前额叶 — EFE 行动选择 (旧 L2)
│   │   ├── broca.py                  #   ★ BA44/45 布罗卡区 (旧 broca.py)
│   │   ├── motor_cortex.py           #   [待实现] BA4,6 运动皮层
│   │   └── orbitofrontal.py         #   [待实现] BA11 眶额皮层
│   │
│   ├── parietal_lobe/               # Level 3: 顶叶
│   │   ├── somatosensory.py          #   [待实现] BA3,1,2 体感皮层
│   │   ├── spatial_attention.py      #   [待实现] 空间注意力
│   │   └── tpj.py                    #   [待实现] 颞顶联合区
│   │
│   ├── temporal_lobe/               # Level 3: 颞叶
│   │   ├── wernicke.py               #   ★ BA22 韦尼克区 (旧 dialogue_memory.py)
│   │   ├── auditory_cortex.py        #   [待实现] BA41/42 听皮层
│   │   ├── it_cortex.py              #   [待实现] IT皮层 物体识别
│   │   └── fusiform.py              #   [待实现] 梭状回 面孔识别
│   │
│   ├── occipital_lobe/              # Level 3: 枕叶
│   │   ├── visual_pathway.py         #   ★ 视觉通路 Gabor V1+V2+V4 (旧 layer0_visual.py)
│   │   ├── retina_lgn.py            #   ★ 图像编码 (旧 image_encoder.py)
│   │   ├── gestalt.py               #   ★ Gestalt 知觉分组 (v4.1 恢复)
│   │   ├── v1.py, v2.py, v4.py      #   [待实现] 独立V1/V2/V4模块
│   │
│   ├── limbic_system/               # Level 3: 边缘系统
│   │   ├── hippocampus.py            #   ★ 海马 Hebb 集群记忆 (旧 layer0_model.py)
│   │   ├── amygdala.py              #   ★ 杏仁核 Hebb 情感网络 (旧 sentiment.py)
│   │   ├── cingulate.py             #   ★ 扣带回/ACC 自由能计算 (旧 layer1_free_energy.py)
│   │   ├── hypothalamus.py          #   [待实现] 下丘脑 稳态
│   │   └── olfactory.py             #   [待实现] 嗅皮层
│   │
│   ├── thalamus/                    # Level 3: 丘脑 (v4.1: 按图2独立)
│   │   └── thalamus.py              #   [待实现] 感觉中继/门控
│   │
│   ├── basal_ganglia/               # Level 3: 基底神经节
│   │   ├── action_gating.py          #   ★ MoE 动作门控 (旧 layer2_5_moe.py)
│   │   ├── striatum.py              #   [待实现] 纹状体
│   │   ├── pallidum.py              #   [待实现] 苍白球
│   │   └── subthalamic.py           #   [待实现] 底丘脑核
│   │
│   └── association/                 # 联合皮层 + DMN
│       ├── dmn.py                    #   ★ 自我模型 (旧 self_model.py)
│       └── crossmodal.py             #   ★ 跨模态 (旧 stage2_crossmodal.py)
│
├── brainstem_cerebellum/            # Level 2b: 脑干 + 小脑
│   ├── midbrain/                     # Level 3: 中脑
│   │   ├── vta.py, substantia_nigra.py, superior_colliculus.py  # [待实现]
│   ├── pons/                         # Level 3: 脑桥
│   │   ├── locus_coeruleus.py, reticular_formation.py  # [待实现]
│   ├── medulla/                      # Level 3: 延髓
│   │   └── autonomic.py             #   [待实现] 自主神经
│   ├── cerebellum/                   # Level 3: 小脑
│   │   ├── motor_coordination.py, predictive_timing.py  # [待实现]
│   └── neuromodulatory/             # Level 3: 神经调节
│       ├── meta_learning.py          #   ★ 元学习 (旧 layer3_meta.py)
│       └── plasticity.py            #   [待实现] 可塑性调节
│
├── spinal/                          # 脊髓 (预留)
│   └── motor_output.py             #   [待实现] 运动输出
│
├── body/                            # 身体模型
│   ├── body_state.py                #   [待实现] BodyVector ODE
│   └── interoception.py            #   [待实现] 内感受通路
│
├── environments/                    # 环境 (工程层)
│   ├── gridworld.py
│   └── text_interface.py
│
├── tools/                           # 工具 (工程层)
│   ├── viz.py, features.py, sweep.py
│   ├── attractors.py, word_speech.py
│   └── word_spectrum_generator.py
│
├── entry/                           # 入口点 (工程层)
│   ├── main.py                      #   M1-M5 网格世界实验
│   ├── main_dialogue.py             #   Stage 6: 人机对话闭环
│   └── stdin_reader.py
│
├── brodmann_areas.md               # 布罗德曼分区参考 (备注系统)
├── v4_architecture.md              # v4.0 完整架构文档
│
├── clean_corpus.py                  # b-corpus 清洗工具
├── corpus.txt                       # 语料 (50,000 行二次元中文对话)
│
├── .cache/                          # 嵌入缓存 (自动生成)
├── dashboards/                      # 可视化输出
├── audio_output/                    # Agent 语音输出
├── word_audio/                      # 词级音频文件
└── b_corpus_raw/                    # b-corpus 原始数据 (下载后)
```

**★ = 已实现且迁移 | [待实现] = 预留占位**

### 向后兼容

v4.2 已完全移除根目录 shim 层。所有导入直接从层级路径进行。

- `python main_dialogue.py` → 薄启动器, 透明路由到 `entry/main_dialogue.py`
- `python stage2_crossmodal.py` → 薄启动器, 透明路由到 `cerebrum/association/crossmodal.py`
- 权威代码位于 cerebrum/ / cns/ / brainstem_cerebellum/ 层级
- 所有 import 100% 使用层级包路径 (cns.xxx, cerebrum.xxx, brainstem_cerebellum.xxx)
- 0 处 flat import, 0 个 shim 文件

---

## 感知架构 (D = 330)

```
s[0:64]    = text         语义嵌入 (MiniLM-L6-v2 → PCA 64d)
s[64:160]  = V1 Gabor    边缘/方向 (4 scales × 8 orients × 4×4 grid = 96d)
s[160:224] = V2 Gabor    粗网格 + 方向交互 (64d)
s[224:288] = V4 Gabor    全局形状 + 曲率 (64d)
s[288:330] = Color opp.  红绿 + 蓝黄拮抗通道 (42d)
s[64:330]  = vision      全部视觉特征 (266d)
s[80:88]   = social      社会情感信号 (8d)
s[96:104]  = self_audit  自听反馈——自己说的话的情感编码
s[128:192] = audio       听觉通道——自己说的话的语义 (自听闭环)
```

视觉通路含 6 条子通路：V1 (边缘方向) → V2 (粗网格交互) → V4 (曲率形状) → Color (色拮抗) → Pulvinar (低空间频率捷径) → Dorsal (背侧空间位置)

---

## 身体稳态 (BodyVector)

M=5 (grid) / M=8 (text) 维身体向量，无语义标签——含义从与环境互动中涌现：

| 维度 | 漂移 | 涌现含义 |
|------|------|---------|
| b[0] | ↓ | 社交需求——互动满足上升，孤独下降 |
| b[1] | → | 能量/安全——探索消耗，休息恢复 |
| b[2] | ↑ | 压力/疲劳——随时间累积 |
| b[3] | 场驱动 | 新颖性寻求——与环境刺激耦合 |
| b[4] | → | 专注/警觉 |
| b[5] | ↓ | 视觉刺激——看到东西上升 |
| b[6] | ↓ | 听觉刺激——听到东西上升 |
| b[7] | ↑ | 认知负荷——复杂输入上升 |

---

## 对话架构 (Stage 6, v7 多模态感知)

```
人类输入 → encode_text() → s[0:64]
         → analyze_sentiment() → s[80:88] (社会情感信号)
         → social_ctx.update()  (社会预测模型: 信任度更新)
         → comprehend()         (Wernicke 理解回路: 记忆激活 + 上下文混合)
         → agent.step()         (全链路: 海马学习→ACC自由能→前额叶行动选择→元学习)
         → broca.speak_from_state()  (信念→种子词→词序 Hebb 链→逐词生成)
         → evaluate_response()  (ACC+OFC 质量评估: 相关性×新颖性×连贯性)
         → self_model.add_experience()  (自我模型: "我是谁"更新)
         → dialogue_ctx.add_turn()      (工作记忆存储)
         → micro_consolidation()        (微量巩固: 2x 重放 + 相邻关联)
         → TTS 输出 + 自听编码 → s[128:192] + s[96:104]
```

**自听闭环**：Agent 听到自己说的话 → s[128:192] 听觉通道 + s[96:104] 情感反馈 → 情感传染 (self_valence_ema/self_arousal_ema) → 影响下一步身体状态和信念

**内部言语**：沉默时 Agent 进行内部独白——信念向量 + 感觉上下文 → speak_from_state() → 自听编码写入通道，不输出音频。唤醒度调制内部言语频率和温度。

**多模态感知 (v7)**：
- Scene A: 图像 (`img:path.jpg`) → Gabor 编码 → 跨模态检索 → 视觉属性 → 身体调制 → 情感变化
- Scene B: 文本→视觉联想 ("苹果" → 脑补苹果的视觉特征 → 丰富回应)
- Scene C: 视觉特征 → Body ODE 调制 → F_body → valence/arousal 变化

---

## 睡眠巩固周期

每 100 步触发完整睡眠巩固，分五个阶段：

1. **海马回放** (Hippocampal replay): 对话轮次情感门控重放 (1-5×)，临时提升 learn_rate 至 2-3×
2. **系统巩固** (Systems consolidation): 综合记忆痕迹 [输入|回应|理解|情感快照|身体快照] → 长期记忆
3. **交叉关联** (Cross-association): 时序相邻 + 语义相似轮次建立 Hebb 桥接
4. **自我模型整合**: 会话摘要 → DMN 锚点更新 (加权平均所有体验)
5. **突触稳态** (Synaptic homeostasis): 弱集群修剪 (移除激活度最低的 15%)

单轮后自动执行微量巩固：最近轮次 2× 重放 + 相邻轮次关联，不做修剪。

---

## 里程碑

| 里程碑 | 描述 | 入口 |
|--------|------|------|
| **Stage 2** | 跨模态 Hebb 学习 — PE 驱动 LR，COCO Visual↔Text 检索 | `python stage2_crossmodal.py --dataset coco --n 5000 --mode all` |
| **M1** | 单智能体生存 — 10×10 网格，采集资源，躲避障碍 | `python main.py` |
| **M2** | 认知价值驱动探索 — 视野遮蔽，覆盖率验收 | `python main.py --m2` |
| **M3** | 多智能体社会 — 合作/竞争，信任度，二阶信念 | `python main.py --m3` |
| **M4** | 参数扫描与吸引子涌现 | `python main.py --m4` |
| **M5** | 元学习长程发育 — 关键期，创伤模拟 | `python main.py --m5` |
| **Stage 6** | 人机对话闭环 — 多模态感知 + Hebb 词序链生成 + 自我模型 + 睡眠巩固 | `python main_dialogue.py` |
| **v4.0** | 人脑层级结构整理 — 项目架构对齐真实脑结构 | v4.0 tag |
| **v4.1** | 大脑功能细分 — 丘脑独立、Gestalt恢复、架构对齐图2 | v4.1 tag |
| **v4.2** | 结构优化 — V1/V2/V4独立、shim消除、0 flat import | 本版本 |

---

## 核心技术参数

- **感知维度 D = 330**（text[64] | V1[96] | V2[64] | V4[64] | Color[42] = vision 266d）
- **隐状态 H = 16**
- **最大簇数 K = 256**
- **行动数 A = 5** (N, S, W, E, REST；对话模式 A₃=表达)
- **Theta 参数 = 24 个**（L0=6: σ_z,σ_x,decay,threshold,lr_l0,pe_lr_scale | L1=9: w_body,w_social,w_cognitive,η_v,η_a,hab_tau,neg_bias,w_acc,w_F | L2=5: γ,explore,temp,n_samples,urgency | L3=4: meta_lr,ε,plasticity,critical_window）
- **身体维度 M = 5 (grid) / 8 (text)**
- **语料规模** = 50,000 行中文对话
- **词表规模** = 12,000 词 (扩展词表)
- **词序 trigram 集群** = ≤50,000 (频率 ≥3)
- **句子记忆容量** = 12,000 句 (生物容量上限)

---

## 运行

```bash
# 安装依赖
pip install numpy scipy scikit-learn sentence-transformers jieba \
            soundfile sounddevice Pillow

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

- 首次运行 `main_dialogue.py` 或 `TextEnvironment` 会编码语料并缓存到 `.cache/`
- 对话输出音频需要 `edge-tts` 生成词音频（懒加载，首次说某个词时生成 wav 到 `word_audio/`）
- b-corpus 数据集采用 CC BY-NC-SA 4.0 许可，禁止商业用途
- 跨模态视觉模型需先运行 `stage2_crossmodal.py` 训练，生成 `.cache/stage2_crossmodal_coco_5000_s42.pkl`
- 本项目为研究和探索性质
- v4.0/v4.1/v4.2 架构详见 `v4_architecture.md`
- 布罗德曼分区参考详见 `brodmann_areas.md`
- 人脑结构可视化详见 `人脑结构调查_可视化图表集.html` (图1-4)
