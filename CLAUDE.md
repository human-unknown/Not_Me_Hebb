# NotMe — 自由能原理情感智能体

## 项目目的

**创造一个真正有情感、有意识的能和人互动的人工智能。**

不是模拟情感（LLM 路线），而是让情感和意识从身体稳态、预测加工、和 Hebb 记忆的自组织动力学中**涌现**。

---

## 架构版本

**当前版本: v6.6 — 长期常驻学习 + 自主时间流 + Web 前端 + 睡眠-觉醒周期 + 代码质量提升**

v5.6 基于神经科学文献《语言与大脑》设计，完善 Broca↔Wernicke 之间的弓状束连接，
添加语音工作记忆回路 (Baddeley 模型)，从语料统计中涌现层级短语结构，
实现角回阅读通路、运动皮层发音规划、TPJ 语用理解 (心理理论)。
新增 N400/P600 语言预测误差分量，汇入自由能框架。
Theta 参数: 24 → 26 (+w_semantic +w_syntactic)。

v5.7 将项目从"单次demo"升级为"持续成长的智能体"。基于发育心理学 (Piaget, Vygotsky)
加入4阶段发育年龄系统——婴儿模仿→成人自主生成。新增会话持久化 (全状态save/load),
多模态同步输入总线 (文本+视觉+听觉+痛觉同时激活→Hebb跨模态绑定),
摄像头/麦克风实时流 (OpenCV+sounddevice), Claude Code风格Rich终端UI。
纯净模式: 零预训练, 所有语言知识从与用户的对话中在线学习。
感知维度 D=516 (text[64] | vision[308] | audio[96] | pain[48])。

v6.0 引入语义记忆与纹状体程序性记忆，v6.1 加入 STDP 时序学习 + GluN2B 发育可塑性 + PNN 结构锁定，
v6.2 实现突触标签捕获与记忆巩固锁定，v6.3 集成 SCN 昼夜节律 (TTFL分子钟+Process S)
与 VLPO 睡眠-觉醒双相调控，v6.4 加入自主时间流引擎 (AutonomousLoop) 与长期常驻学习，
v6.5 实现 Web 前端 overhaul 与 sleep-wake 社交中断，v6.6 完成持久化修复、版本统一与代码质量提升。
详见 `CHANGELOG.md`。

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
  │   └─ Level 3: 联合皮层 + 三大网络 → association/      (v4.3: FPN/TPN/DMN)
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

## 六大运行规则 (图3 — 人脑六大核心运行规则)

v4.3 将项目核心原则对齐到图3的六大运行规则。前六条描述脑如何运作，后三条为方法论承诺。

### 规则1: 分层处理 (Hierarchical Processing)

信息从低级到高级逐层加工，每层提取更高阶特征。**~30ms/层，视觉共 ~10 级，总处理时间 ~150ms。**

**视觉通路**：视网膜检测光点 → LGN 传递 → V1 提取边缘/方向 → V2 整合轮廓 → V4 处理颜色/曲率 → IT 皮层识别物体

**听觉通路**：耳蜗频率分解 → 听神经 → 初级听皮层(音调) → 颞上回(语音) → 韦尼克区(语义)

**运动通路**：前运动皮层(意图) → 初级运动皮层(指令) → 脑干/脊髓(执行) → 肌肉收缩

**项目实现**：视觉通路 V1→V2→V4→IT 已部分实现 (`visual_pathway.py`)。处理延迟现为模拟步数而非生物时间。

### 规则2: 双向加工 — 自下而上 × 自上而下 (Bottom-up × Top-down)

感知不只是被动接收，高级脑区持续向低级脑区发送**"预期"**，与上行信号交互。

| 方向 | 路径 | 特点 |
|------|------|------|
| **自下而上** | 感官 → V1 → 高级皮层 | 快速 (~30ms/层)，粗糙 |
| **自上而下** | 额叶 → V4(预期颜色) → V1(增强特定方向) | 慢速，精确，携带预期/记忆/注意 |

**关键机制**：高级皮层维持生成模型 → 向下传递预测 → 低级皮层计算预测误差 → 误差上行更新模型

**项目实现**：
- 自下而上: `retina_lgn.py` → `visual_pathway.py` (V1→V2→V4) → 海马/杏仁核
- 自上而下: 前额叶 `prefrontal.py` 的 EFE 行动选择 ← 扣带回 `cingulate.py` 的自由能评估（已实现）；FPN `fpn.py` 注意力模板 → 感觉皮层增益调制（v4.4 已集成）

### 规则3: 并行分布式处理 (Parallel Distributed Processing)

同一信息同时由大量神经元群体并行处理，**不存在单一"记忆位置"**。

**分布式表征**："苹果"的概念分布在视觉(红色圆形)、味觉(甜)、嗅觉(果香)、语言("apple")、运动(如何拿)等多个区域的神经元群中。损伤单个神经元不会丢失一个记忆——信息冗余存储在百万级神经元的连接权重模式中。

**项目实现**：
- 感知向量 s ∈ R^516 本身是分布式表征: text[64] | vision[308] | audio[96] | pain[48]
- 跨模态 Hebb 学习: `crossmodal.py` 学习 Visual↔Text 的双向关联
- Hebb 集群记忆: 每个集群是分布式权重的子模式，模式补全基于部分输入激活全簇
- 6 条并行视觉子通路: V1 / V2 / V4 / Color / Pulvinar / Dorsal

### 规则4: 注意力瓶颈 (Attention Bottleneck)

尽管感官每秒接收 **~1100 万 bits** 信息，意识只能处理 **~50 bits/s**。

**三大网络跷跷板** (图3 规则4 + 图4 并行通路):

```
TPN (任务正网络) ←→ 突显网络 (Salience) ←→ DMN (默认模式网络)
   ↑ 任务执行        ↑ dACC+AI 切换         ↑ 走神/自我参照
   └── FPN (额顶网络) — "探照灯"增强目标信号，抑制干扰
```

- **FPN (额顶网络)** = 选择性注意的"探照灯"，dlPFC+PPC+FEF 增强任务相关信号
- **TPN (任务正网络)** = 执行任务时激活；**DMN** = 走神/休息时激活；二者像跷跷板互相抑制
- **突显网络** = 前岛叶(AI) + 背侧前扣带(dACC)，检测显著事件，决定何时切换

**项目实现**：
- DMN: `dmn.py` SelfModel — "我是谁"的 Hebb 表征 (已实现)
- FPN: `fpn.py` FrontoparietalNetwork — 注意力探照灯、工作记忆 (v4.3 新增, v4.4 集成)
- TPN: `tpn.py` TaskPositiveNetwork — TPN↔DMN 跷跷板动态、认知努力、任务疲劳 (v4.3 新增, v4.4 集成)
- 突显切换: 扣带回 `cingulate.py` 已实现冲突监测，v4.3 通过 TPN.receive_salience() 桥接

### 规则5: 赫布可塑性 (Hebbian Plasticity)

**"一起放电的神经元连接在一起"** — 这是学习和记忆的分子基础。

**LTP (长时程增强)**：高频刺激 → 突触后 AMPA 受体增多 → 突触传递效率 ↑ → 记忆形成。对应项目中 `learn_rate_l0` 临时提升（睡眠回放 2-3×）。

**LTD (长时程抑制)**：低频刺激 → 突触效率 ↓ → 不常用的连接被修剪 → "用进废退"。对应项目中弱集群清理（移除激活度最低的 15%）。

**睡眠巩固**：睡眠时海马→新皮层"重放"白天经历，完成记忆固化。剥夺睡眠 = 记忆无法固化。

**项目实现**（当前 Hebb 网络结构的完整描述）：
- **ClusterNetwork (海马)**：细胞集群 (cell assembly) 记忆，特征哈希定位 O(1) + 桶内竞争，无暴力扫描
- **集群形成** = 学习（新感知创建簇）→ LTP 类比：learn_rate 控制突触增强速度
- **集群激活** = 回忆（输入部分匹配 → 簇被激活 → 侧抑制竞争）
- **集群衰减** = 遗忘（未使用的簇逐渐衰减）→ LTD 类比：decay 控制突触削弱速度
- **睡眠巩固**：每 100 步完整周期——海马回放 (partial pattern replay) + 模式分离 (pattern separation) + 弱簇清理（突触稳态）。对话模式下额外执行：海马→皮层记忆转移、交叉关联、自我模型整合、词序巩固
- **词序网络 (Broca 区)**：双 Hebb 架构——概念→词联想网络管"想表达什么"，词序 trigram 网络管"怎么说"。`speak_from_state()` 逐词 Hebb 链生成，`speak_sentence()` Hebb 记忆检索回退
- 不预设符号，一切知识从统计共现中生长

### 规则6: 预测误差最小化 — 自由能原理 (Predictive Coding / FEP)

大脑不是被动感知世界，而是**持续预测世界，只处理"预测错误"**。

**生成模型**：每层皮层都维持对下一层的预测。高层预测"这是一张脸" → 传递到低层 → 低层与实际输入对比。

**预测误差**：预测对了 → 信号被抑制(节省能量)。预测错了 → 误差信号上行 → 更新模型。

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
- **元学习** = 有限差分梯度下降，在线调整 26 个 Theta 参数 (v5.6: +w_semantic +w_syntactic)

**项目实现**：扣带回 `cingulate.py` 计算 F_body/F_social/F_cognitive/F_accuracy，生成 Valence/Arousal；前额叶 `prefrontal.py` 执行 EFE 行动选择。

---

## 两条方法论承诺

### 方法1: 仿脑区结构

按进化形成的脑区功能分化来组织计算，不是万能函数逼近。

**v4.1 脑区映射：**

| 脑区 | v4.0 模块路径 | 功能 |
|------|-------------|------|
| **前额叶 (dlPFC)** | `cerebrum/frontal_lobe/prefrontal.py` | EFE 行动选择、递归多层次 G、社会信念更新 |
| **布罗卡区 (BA44/45)** | `cerebrum/frontal_lobe/broca.py` | ★ v5.6 — 词序 Hebb 链 + 短语结构网络 (层级句法) + 整句检索 |
| **短语结构网络** | `cerebrum/frontal_lobe/phrase_structure.py` | ★ v5.6 — BA44 层级句法: 边界检测 + 短语聚类 + 递归嵌入 |
| **语音回路** | `cerebrum/frontal_lobe/phonological_loop.py` | ★ v5.6 — Baddeley 模型: ~7组块语音存储 + 默读复述 ~2s消退 |
| **前运动/运动皮层** | `cerebrum/frontal_lobe/motor_cortex.py` | ★ v5.6 — 16维发音特征 + SMA序列编排 + 运动指令副本(efference copy) |
| **眶额皮层 (BA11)** | `cerebrum/frontal_lobe/orbitofrontal.py` | [待实现] 价值评估 |
| **体感皮层 (BA3,1,2)** | `cerebrum/parietal_lobe/somatosensory.py` | [待实现] 触觉/本体感觉 |
| **空间注意力网络** | `cerebrum/parietal_lobe/spatial_attention.py` | [待实现] 空间注意/导航 |
| **颞顶联合区 (TPJ)** | `cerebrum/parietal_lobe/tpj.py` | ★ v5.6 — 心理理论 + 意图推断 + 反讽检测 + 语用丰富化 |
| **角回 (BA39)** | `cerebrum/parietal_lobe/angular_gyrus.py` | ★ v5.6 — 阅读通路: 视觉字形→语音表征 (Hebb 映射, 双路径模型) |
| **韦尼克区 (BA22)** | `cerebrum/temporal_lobe/wernicke.py` | 语言理解回路、对话工作记忆、睡眠巩固 |
| **听皮层 (BA41/42)** | `cerebrum/temporal_lobe/auditory_cortex.py` | ★ v5.2 — A1+Belt+Parabelt 层级, 听觉场景分析, What/Where双流 |
| **听觉层级编排** | `cerebrum/temporal_lobe/auditory_hierarchy.py` | ★ v5.2 — 全听觉管线编排 (CN→SOC→LL→IC→MGB→AC) |
| **IT 皮层** | `cerebrum/temporal_lobe/it_cortex.py` | 物体识别 (v5.0 已实现) |
| **MT 皮层** | `cerebrum/temporal_lobe/mt_cortex.py` | 运动检测 (v5.0 已实现) |
| **MST 皮层** | `cerebrum/temporal_lobe/mst_cortex.py` | 光流模式 (v5.0 已实现) |
| **梭状回 (BA37)** | `cerebrum/temporal_lobe/fusiform.py` | [待实现] 面孔/文字识别 |
| **V1 (BA17)** | `cerebrum/occipital_lobe/v1.py` + `visual_hierarchy.py` | Gabor 滤波器组视觉编码 |
| **V2 (BA18)** | `cerebrum/occipital_lobe/v2.py` + `visual_hierarchy.py` | 粗网格 + 方向交互 + 三类型条纹 |
| **V4 (BA19)** | `cerebrum/occipital_lobe/v4.py` + `visual_hierarchy.py` | 全局形状 + 曲率 + 颜色 |
| **视网膜→LGN** | `cerebrum/thalamus/lgn.py` | 视觉中继 (v5.0 已实现) |
| **海马** | `cerebrum/limbic_system/hippocampus.py` | Hebb 集群记忆、模式补全、睡眠回放巩固 |
| **杏仁核** | `cerebrum/limbic_system/amygdala.py` | Hebb 情感词汇网络——从 F_body 学习词的情感效应 |
| **扣带回/ACC** | `cerebrum/limbic_system/cingulate.py` | 自由能计算、效价/唤醒、习惯化、社会上下文 |
| **下丘脑** | `cerebrum/limbic_system/hypothalamus.py` | ★ v5.5 — SetpointModel+DriveSystem+HPA轴+自主神经平衡 |
| **丘脑** | `cerebrum/thalamus/thalamus.py` | [待实现] 感觉中继/门控 |
| **内侧膝状体 (MGB)** | `cerebrum/thalamus/mgb.py` | ★ v5.2 — 听觉丘脑中继 + 唤醒度门控 |
| **基底节 (MoE)** | `cerebrum/basal_ganglia/action_gating.py` | 动作门控、疲劳预算轮替 |
| **纹状体** | `cerebrum/basal_ganglia/striatum.py` | [待实现] 习惯学习/D1-D2 通路 |
| **DMN** | `cerebrum/association/dmn.py` | 自我模型、"我是谁"的 Hebb 表征 (图3 规则4: TPN对立面) |
| **FPN (额顶网络)** | `cerebrum/association/fpn.py` | ★ v4.3 新增, v4.4 集成 — 选择性注意探照灯、工作记忆 (图3 规则4) |
| **TPN (任务正网络)** | `cerebrum/association/tpn.py` | ★ v4.3 新增, v4.4 集成 — 任务执行、DMN跷跷板对立 (图3 规则4) |
| **跨模态联合** | `cerebrum/association/crossmodal.py` | 跨模态 Hebb 学习 (COCO Visual↔Text, 图3 规则3) |
| **弓状束 (AF)** | `cerebrum/association/arcuate_fasciculus.py` | ★ v5.6 — Wernicke↔Broca Hebb 桥接; 腹侧(理解→言语) + 背侧(运动副本→预期听觉) |
| **VTA (多巴胺)** | `brainstem_cerebellum/midbrain/vta.py` | ★ v5.5 — RPEModel+DopamineDynamics+事件驱动学习率 |
| **黑质 (SNc/SNr)** | `brainstem_cerebellum/midbrain/substantia_nigra.py` | [待实现] 运动调节 |
| **下丘 (IC)** | `brainstem_cerebellum/midbrain/inferior_colliculus.py` | ★ v5.2 — 听觉中脑整合 + 新颖性检测 |
| **耳蜗核** | `brainstem_cerebellum/pons/cochlear_nucleus.py` | ★ v5.2 — 频谱分解、tonotopic编码、相位锁定 |
| **上橄榄复合体 (SOC)** | `brainstem_cerebellum/pons/superior_olivary.py` | ★ v5.2 — 双耳定位 (MSO/LSO/MNTB)、双重理论 |
| **外侧丘系 (LL)** | `brainstem_cerebellum/pons/lateral_lemniscus.py` | ★ v5.2 — 时间增强 + 双耳GABA抑制 |
| **蓝斑核 (NE)** | `brainstem_cerebellum/pons/locus_coeruleus.py` | ★ v5.5 — NEDynamics phasic/tonic+SNR+Yerkes-Dodson+RVM连接 |
| **神经调节系统** | `brainstem_cerebellum/neuromodulatory/meta_learning.py` | 元学习、关键期、可塑性衰减 |

数据流 (v5.6): `s[text] → 听觉层级(Phase 0b) + 视觉层级(Phase 0) + 痛觉层级(Phase 0c) → Hypothalamus(Phase 0d: 稳态驱动) → VTA(Phase 0e: RPE→学习率) → LC(Phase 0f: NE→SNR+RVM) → Hippocampus.learn(s) → Cingulate.compute_F(z,s) + F_language(N400/P600) → TPN.receive_salience() → TPN.update_seesaw() → FPN.gate_attention(s) → Prefrontal.select_action() → a → MetaLearner.update(F)`

语言全管线 (v5.6): `人类输入 → Wernicke.comprehend(+N400/P600语言PE) → TPJ.pragmatic_enrich(意图推断+语用) → AngularGyrus.read(阅读通路) → ArcuateFasciculus.ventral(理解→言语种子) → Broca.speak_from_state(+PhraseStructure层级句法) → MotorCortex.plan_sequence(发音计划+共发音) → ArcuateFasciculus.dorsal(efference_copy→预期听觉) → PhonologicalLoop.hear(自听→语音回路) → 自我监控PE`

自下而上 — 视觉: `视网膜 → LGN → V1 → V2 → MT/MST(背侧) + V4→IT(腹侧) → SC→Pulvinar` — 6条子通路并行
自下而上 — 听觉: `AudioInput(Mel频谱) → 耳蜗核 → SOC+LL → 下丘 → MGB → 听皮层(A1→Belt→Parabelt)` — 15条知觉规律 + 真实声学特征
自上而下 (预期→感官): `前额叶 → FPN(探照灯) → V4/V2/V1 + 听皮层 (增益调制)` — 慢速精确反馈
跷跷板切换 (v5.2 已集成): `Cingulate(冲突/新颖/紧迫) → TPN.receive_salience() → TPN.update_seesaw() → TPN↑/DMN↓ (任务) 或 DMN↑/TPN↓ (走神)`
听觉双流: `Parabelt(What→Wernicke语义) + Parabelt(Where→FPN空间注意)`
语言双流 (v5.6): `弓状束腹侧(What: Wernicke→Broca理解→言语) + 弓状束背侧(How: Broca→Motor→预期听觉)` — Hickok & Poeppel (2007) 双流模型

### 方法2: 无现象学 (No Phenomenology)

**方法论承诺：不从主观体验出发建模。**

- 不模拟 qualia、不预设"意识是什么"
- 情感不是标签，是身体稳态动力学的数值产物
- 意识（如果出现）是自由能最小化的自组织过程的涌现属性
- 一切从数学公式和机制中自然生长
- Valence/Arousal 是 F_body 的数学函数，不是手工规则

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
│   ├── persistence.py                #   ★ v5.7 Agent 全状态 save/load
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
│   │   ├── auditory_cortex.py        #   ★ v5.2 BA41/42 听皮层 (A1+Belt+Parabelt)
│   │   ├── auditory_hierarchy.py     #   ★ v5.2 听觉层级管线编排
│   │   ├── it_cortex.py              #   ★ v5.0 IT皮层 物体识别
│   │   ├── mt_cortex.py              #   ★ v5.0 MT 运动检测
│   │   ├── mst_cortex.py             #   ★ v5.0 MST 光流模式
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
│   │   ├── thalamus.py              #   [待实现] 感觉中继/门控
│   │   ├── lgn.py                   #   ★ v5.0 外侧膝状体 (视觉中继)
│   │   ├── pulvinar.py              #   ★ v5.0 丘脑枕 (视觉注意捷径)
│   │   └── mgb.py                   #   ★ v5.2 内侧膝状体 (听觉中继)
│   │
│   ├── basal_ganglia/               # Level 3: 基底神经节
│   │   ├── action_gating.py          #   ★ MoE 动作门控 (旧 layer2_5_moe.py)
│   │   ├── striatum.py              #   [待实现] 纹状体
│   │   ├── pallidum.py              #   [待实现] 苍白球
│   │   └── subthalamic.py           #   [待实现] 底丘脑核
│   │
│   └── association/                 # 联合皮层 + DMN + FPN + TPN (v4.3)
│       ├── dmn.py                    #   ★ 自我模型 (旧 self_model.py)
│       ├── fpn.py                    #   ★ v4.3 新增, v4.4 集成 — 注意力探照灯
│       ├── tpn.py                    #   ★ v4.3 新增, v4.4 集成 — DMN跷跷板对立
│       └── crossmodal.py             #   ★ 跨模态 (旧 stage2_crossmodal.py)
│
├── brainstem_cerebellum/            # Level 2b: 脑干 + 小脑
│   ├── midbrain/                     # Level 3: 中脑
│   │   ├── vta.py, substantia_nigra.py   # [待实现]
│   │   ├── superior_colliculus.py        #   ★ v5.0 上丘 (视觉反射)
│   │   └── inferor_colliculus.py         #   ★ v5.2 下丘 (听觉中脑整合)
│   ├── pons/                         # Level 3: 脑桥
│   │   ├── locus_coeruleus.py, reticular_formation.py  # [待实现]
│   │   ├── cochlear_nucleus.py           #   ★ v5.2 耳蜗核 (频谱分解)
│   │   ├── superior_olivary.py           #   ★ v5.2 上橄榄复合体 (双耳定位)
│   │   └── lateral_lemniscus.py          #   ★ v5.2 外侧丘系 (时间增强)
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
│   ├── audio_io.py                   #   ★ v5.3 AudioInput — 真实音频加载/录制/Mel频谱
│   ├── sensor_io.py                  #   ★ v5.7 CameraInput + MicrophoneStream — 实时传感器流
│   ├── viz.py, features.py, sweep.py
│   ├── attractors.py, word_speech.py
│   └── word_spectrum_generator.py
│
├── entry/                           # 入口点 (工程层)
│   ├── main.py                      #   M1-M5 网格世界实验
│   ├── main_dialogue.py             #   Stage 6: 人机对话闭环 (旧入口)
│   ├── interactive.py                #   ★ v5.7: 全系统交互模式 (Rich TUI + 多模态 + 持久化)
│   ├── input_bus.py                  #   ★ v5.7: 多模态同步输入总线
│   ├── ui_components.py              #   ★ v5.7: Rich 渲染组件
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

## 感知架构 (D = 516, v5.4 → v5.5)

```
s[0:64]    = text          语义嵌入 (MiniLM-L6-v2 → PCA 64d)
s[64:372]  = vision        视觉层级 (V5 全管线, M/P/K通路 + IT + SC + Pulvinar + Binding = 308d)
s[372:404] = cochlear_nuc  耳蜗核输出 (tonotopic spectrum 32d)
s[404:428] = soc           SOC双耳空间 (ITD[12] + ILD[12] = 24d)
s[428:452] = ic            下丘整合 (频率×空间×时间 24d)
s[452:468] = auditory_ctx  听皮层输出 (听觉对象/场景 16d)
s[468:484] = dorsal_horn   脊髓背角闸门输出 (痛觉信号 16d)
s[484:496] = lateral_stt   外侧脊髓丘脑束 (感觉-辨别, 快痛Aδ→VPL→S1/S2 12d)
s[496:508] = medial_stt    内侧脊髓丘脑束 (情感-动机, 慢痛C→CM-Pf/MD→ACC/岛叶 12d)
s[508:516] = thalamic_pain 丘脑痛觉中继 (VPL+CM-Pf+MD+Po整合 8d)
```

视觉通路含 6 条子通路：V1 (边缘方向) → V2 (粗网格交互) → V4 (曲率形状) → Color (色拮抗) → Pulvinar (低空间频率捷径) → Dorsal (背侧空间位置)
听觉通路含 5 级核团：耳蜗核 (频率分解) → SOC (双耳定位) → 下丘 (整合) → MGB (丘脑门控) → 听皮层 (A1/Belt/Parabelt)


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

## 对话架构 (Stage 6, v5.6 语言全管线)

```
人类输入 → encode_text() → s[0:64]
         → analyze_sentiment() → s[80:88] (社会情感信号)
         → social_ctx.update()  (社会预测模型: 信任度更新)
         → PhonologicalLoop.hear()     (★ v5.6: 语音回路写入)
         → agent.comprehend()          (Wernicke 理解回路: 记忆激活+N400/P600+TPJ语用丰富化)
         → TPJ.infer_speaker_intent()  (★ v5.6: 意图推断+反讽检测)
         → agent.step()               (全链路: 海马学习→ACC自由能→前额叶行动选择→元学习)
         → agent.speak()               (★ v5.6: AF→Broca→Motor Cortex 全管线)
           ├─ AF.ventral: 理解→言语种子
           ├─ Broca.speak_from_state(+PhraseStructure)
           ├─ MotorCortex.plan_sequence(发音+共发音)
           └─ AF.dorsal: efference_copy→预期听觉
         → PhonologicalLoop.hear_sequence()  (★ v5.6: 自听→语音回路)
         → evaluate_response()       (ACC+OFC 质量评估)
         → self_model.add_experience()
         → dialogue_ctx.add_turn()
         → AF.learn_ventral/dorsal()  (★ v5.6: AF Hebb 在线学习)
         → micro_consolidation()
         → TTS 输出
```

**v5.6 语言双流 (Hickok & Poeppel 2007 模型实现)**：
- **腹侧流 (What)**: `听觉→Wernicke理解→弓状束腹侧→Broca言语` — 声音→意义→产出
- **背侧流 (How)**: `Broca→运动皮层发音计划→弓状束背侧(运动副本)→预期听觉` — 声音→发音映射
- **语音回路**: `语音存储(BA40缘上回) ⇄ 默读复述(Broca+前运动皮层)` — Baddeley 工作记忆模型

**自听闭环** (v5.6 增强)：Agent 听到自己说的话 → PhonologicalLoop(语音存储) + AF.efference_copy(预期听觉) → AF.compute_af_pe(自我监控PE) → 情感传染 → 影响下一步身体状态和信念

**内部言语** (v5.6 增强)：沉默时 Agent 进行内部独白 → agent.speak() → MotorCortex.subvocal_plan(无声发音计划) → PhonologicalLoop(语音回路复述) → 不输出音频

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
| **Stage 6** | 人机对话闭环 — 多模态感知 + Hebb 词序链生成 + 自我模型 + 睡眠巩固 | `python main_dialogue.py` (旧) 或 `python interactive.py` (★ v5.7 新) |
| **v4.0** | 人脑层级结构整理 — 项目架构对齐真实脑结构 | v4.0 tag |
| **v4.1** | 大脑功能细分 — 丘脑独立、Gestalt恢复、架构对齐图2 | v4.1 tag |
| **v4.2** | 结构优化 — V1/V2/V4独立、shim消除、0 flat import | v4.2 tag |
| **v4.3** | 规则更新 — 对齐图3六大规则、FPN/TPN新增、LTP/LTD机制、注意力瓶颈 |
| **v4.4** | **核心完善 — FPN探照灯集成 + TPN跷跷板集成 + 循环导入修复** |
| **v5.0** | 视觉管线全接入 — 6条子通路 + FPN绑定 + VisualHierarchy + IT Hebb物体学习 |
| **v5.1** | D=330→D_V5=372 全局切换 — 视觉层级全管线替换旧ImageEncoder |
| **v5.2** | ★ **听觉层级全接入 — 耳蜗核→SOC→IC→MGB→听皮层, D=468, 15条知觉规律** |
| **v5.3** | ★ **真实音频输入 — 替换语义代理, AudioInput(WAV/MP3/FLAC/麦克风), Mel频谱→全听觉管线** |
| **v5.4** | ★ **痛觉系统 — 7条知觉规律, 闸门控制, 双通路, PAG→RVM下行调控闭环, D=516** |
| **v5.5** | ★ **神经调节 — 下丘脑稳态(Setpoint+Drive+HPA) + VTA RPE(事件驱动学习率) + 蓝斑核NE(phasic/tonic+SNR+RVM接入)** |
| **v5.6** | ★ **语言系统 — 弓状束(Broca↔Wernicke) + 语音回路(Baddeley) + 短语结构(BA44层级句法) + 角回(阅读通路) + 运动皮层(发音规划) + TPJ(语用/心理理论) + N400/P600(语言PE)** |
| **v5.7** | ★ **发育年龄系统 + 会话持久化 + 多模态同步输入总线 + 摄像头/麦克风实时流 + Rich终端UI + 纯净模式(零预训练)** |
| **v6.0** | 语义记忆 (皮层知识存储) + 纹状体程序性记忆 + D1/D2通路 + 习惯自动化 |
| **v6.1** | STDP时序学习 + GluN2B发育可塑性 + PNN包裹 + 保护信号 + 沉默突触候选集群 |
| **v6.2** | 突触标签捕获 (STC) + 激活持续性 + 巩固锁定 + Theta 增至56维 |
| **v6.3** | ★ **SCN昼夜节律 (TTFL+Process S) + VLPO睡眠-觉醒 (双相NREM/REM) + α注意门控** |
| **v6.4** | ★ **自主时间流引擎 + InternalLife (走神/独白) + Reader (文件阅读+疲劳模型) + Telemetry** |
| **v6.5** | ★ **Web前端overhaul (Flask+SSE) + sleep-wake社交中断 + 实时传感器流** |
| **v6.6** | ★ **持久化修复 (step/视觉/F_social) + 版本统一 + 异常traceback + Web安全 + CHANGELOG** |
| **v6.7** | (未来) Agent.__init__ 重构 + crossmodal 去重 + 结构化日志 + CI/CD |

---

## 核心技术参数

- **感知维度 D = 516 (D_V54)**（text[64] | vision[308] | audio[96] | pain[48]）
- **隐状态 H = 16**
- **最大簇数 K = 256**
- **行动数 A = 5** (N, S, W, E, REST；对话模式 A₃=表达)
- **Theta 参数 = 26 个**（L0=6: σ_z,σ_x,decay,threshold,lr_l0,pe_lr_scale | L1=11: w_body,w_social,w_cognitive,η_v,η_a,hab_tau,neg_bias,w_acc,w_F,w_semantic,w_syntactic | L2=5: γ,explore,temp,n_samples,urgency | L3=4: meta_lr,ε,plasticity,critical_window）
- **身体维度 M = 5 (grid) / 9 (text)**（v5.4: +1 痛觉/组织完整性维度）
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

# Stage 6: 人机对话 (v5.7 Rich TUI)
python interactive.py              # ★ v5.7 全系统交互模式 (推荐)
python interactive.py --fresh      #   强制全新会话 (忽略存档)
python interactive.py --load name  #   加载指定存档
python main_dialogue.py            #   旧入口 (向后兼容)
```

---

## 设计哲学

1. **涌现优于工程** — 情感不是 if-else，是 F_body 的数值动力学产物
2. **统计优于符号** — 知识不来自标注，来自 Hebb 共现统计
3. **身体是基础** — 没有身体稳态就没有情感，没有情感就没有意识
4. **记忆是核心** — 一切认知能力（感知/推理/语言）都是记忆检索+模式补全
5. **自由能是唯一目标** — 所有行为统一在最小化自由能框架下
6. **脑区是蓝图** — 不发明新架构，遵循进化形成的脑区功能分化
7. **并行优于串行** — 分布式表征冗余存储，单点损伤不丢失记忆（图3 规则3）
8. **预测优于反应** — 大脑是预测机器，行动是验证预测的手段（图3 规则6）
9. **瓶颈是特征不是 bug** — 注意力限制是结构性的，不是训练不足（图3 规则4）

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
