# NotMe 项目状态报告

> **版本**: v6.3 — 长期常驻学习 & 睡眠优化与时间维度 (SCN昼夜节律 + NREM/REM双相睡眠 + α注意门控 + 类淋巴清除)
> **日期**: 2026-06-07
> **基于**: 《脑节律与睡眠》综合文档 + 自由能原理 + Hebb 可塑性 + 双过程睡眠模型 (Borbély) + Saper触发器开关

---

## 版本历史

| 版本 | 日期 | 关键变更 |
|------|------|---------|
| v4.0 | 2026-06-05 | 人脑层级结构架构重组 |
| v4.1 | 2026-06-05 | 大脑功能细分 + 丘脑独立 + Gestalt恢复 |
| v4.2 | 2026-06-05 | V1/V2/V4独立模块 + shim消除 + 0 flat import |
| v4.3 | 2026-06-05 | 图3六大运行规则对齐 + FPN/TPN新增接口 + LTP/LTD |
| v4.4 | 2026-06-05 | FPN探照灯集成 + TPN跷跷板集成 + 循环导入修复 |
| **v5.0** | **2026-06-05** | **视觉系统 M/P/K 并行通路 + 10脑区层级管线 + 预测编码闭环** |
| **v5.1** | **2026-06-05** | **全视觉管线接入 agent.step() + D=330→D_V5=372 全局切换 + 自听回路重构** |
| **v5.2** | **2026-06-05** | **听觉层级全接入 — 耳蜗核→SOC→IC→MGB→听皮层, D=468, 15条知觉规律** |
| **v5.3** | **2026-06-05** | **真实音频输入 — 替换语义代理模式, AudioInput(WAV/MP3/FLAC/麦克风), Mel频谱驱动全听觉管线** |
| **v5.4** | **2026-06-05** | **痛觉系统 — 7条知觉规律, 闸门控制, 双通路, 下行调控PAG→RVM闭环, D=516** |
| **v5.5** | **2026-06-05** | **神经调节系统 — 下丘脑稳态 + VTA RPE + 蓝斑核NE** |
| **v5.6** | **2026-06-06** | **语言系统 — 弓状束+语音回路+短语结构+角回+运动皮层+TPJ+N400/P600** |
| **v5.7** | **2026-06-06** | **发育年龄系统 + 会话持久化 + 多模态同步输入 + 实时传感器流 + Rich TUI + 纯净模式** |
| **v6.0** | **2026-06-06** | **记忆系统完整实现 — 语义记忆/程序性记忆/干扰遗忘/跨会话系统巩固/零预训练纯净启动** |
| **v6.1** | **2026-06-06** | **发育优化 — STDP时序学习 + GluN2B→GluN2A轨迹 + PNN结构锁定 + 保护信号(CD47) + 沉默突触 + 4阶段发育年龄** |
| **v6.2** | **2026-06-07** | **记忆巩固优化 — 突触标签捕获(STC) + CaMKII激活持续性 + PKMζ巩固锁定** |
| **v6.3** | **2026-06-07** | **睡眠优化与时间维度 — SCN昼夜节律钟 + VLPO触发器开关 + NREM/REM双相睡眠 + α注意门控 + 类淋巴清除** |

---

## 模块完成度

### 总览

| 状态 | 数量 | 占比 |
|------|------|------|
| ★ 已实现 (含 v6.3 新增) | **62** | 70% |
| ○ 占位 (接口设计完成, 待实现代码) | 12 | 14% |
| — 配置/工具/入口 (无类定义但功能完备) | 14 | 16% |
| **合计** | **88** | **100%** |

### 按脑区

#### 大脑皮层 (Cerebrum) — 核心认知

| 脑区 | 模块 | 状态 |
|------|------|------|
| **额叶** | | |
| 前额叶 (dlPFC) | `frontal_lobe/prefrontal.py` | ★ 已实现 (EFE行动选择) |
| 布罗卡区 (BA44/45) | `frontal_lobe/broca.py` | ★ 已实现 (词序Hebb链) |
| 短语结构网络 | `frontal_lobe/phrase_structure.py` | ★ v5.6 (BA44层级句法) |
| 语音回路 | `frontal_lobe/phonological_loop.py` | ★ v5.6 (Baddeley模型) |
| 运动皮层 (BA4/6) | `frontal_lobe/motor_cortex.py` | ★ v5.6 (16维发音特征+SMA) |
| 眶额皮层 (BA11) | `frontal_lobe/orbitofrontal.py` | ○ 占位 |
| **顶叶** | | |
| 体感皮层 (BA3,1,2) | `parietal_lobe/somatosensory.py` | ★ v5.4 (S1/S2全实现) |
| **视空间模板** | `parietal_lobe/working_memory.py` | **★ NEW v6.0** (Baddeley模型 ~4视觉组块+心理旋转+空间比较) |
| 颞顶联合区 (TPJ) | `parietal_lobe/tpj.py` | ★ v5.6 (心理理论+意图推断) |
| 角回 (BA39) | `parietal_lobe/angular_gyrus.py` | ★ v5.6 (阅读通路) |
| **颞叶** | | |
| 韦尼克区 (BA22) | `temporal_lobe/wernicke.py` | ★ v5.6 (+N400/P600) |
| 听皮层 (BA41/42) | `temporal_lobe/auditory_cortex.py` | ★ v5.2 (A1+Belt+Parabelt) |
| 听觉层级管线 | `temporal_lobe/auditory_hierarchy.py` | ★ v5.2 (6核团编排) |
| IT皮层 | `temporal_lobe/it_cortex.py` | ★ v5.0 (Hebb物体学习) |
| MT (V5) | `temporal_lobe/mt_cortex.py` | ★ v5.0 (方向选择性) |
| MST | `temporal_lobe/mst_cortex.py` | ★ v5.0 (光流模式) |
| **语义记忆 (前颞叶)** | `temporal_lobe/semantic_memory.py` | **★ NEW v6.0** (独立ClusterNetwork 1024簇, 慢学慢衰, 主旨提取, 系统巩固) |
| 梭状回 (BA37) | `temporal_lobe/fusiform.py` | ○ 占位 |
| **枕叶** | | |
| 视觉通路 (V1+V2+V4) | `occipital_lobe/visual_pathway.py` | ★ 已实现 (Gabor滤波) |
| 视网膜→LGN | `occipital_lobe/retina_lgn.py` | ★ 已实现 (M/P/K分型) |
| Gestalt 知觉分组 | `occipital_lobe/gestalt.py` | ★ 已实现 |
| V1 (BA17) | `occipital_lobe/v1.py` | ★ v5.0 (层状模块) |
| V2 (BA18) | `occipital_lobe/v2.py` | ★ v5.0 (三类条纹) |
| V4 (BA19) | `occipital_lobe/v4.py` | ★ v5.0 (M/P/K汇合) |
| 视觉层级管线 | `occipital_lobe/visual_hierarchy.py` | ★ v5.0 (10模块编排) |
| **边缘系统** | | |
| 海马 | `limbic_system/hippocampus.py` | **★ UPGRADED v6.3** (+STDP +保护 +沉默突触 +PNN +标签捕获 +CaMKII持续性 +PKMζ锁定 +**双相睡眠NREM/REM**) |
| 杏仁核 | `limbic_system/amygdala.py` | **★ UPGRADED v6.0** (+恐惧条件作用) |
| 扣带回/ACC | `limbic_system/cingulate.py` | ★ 已实现 (自由能计算) |
| 岛叶 (BA13-16) | `limbic_system/insula.py` | ★ v5.4 (内感受+突显网络) |
| **SCN (视交叉上核, BA—)** | `limbic_system/scn.py` | **★ NEW v6.3** (TTFL分子钟+Process S+光同步) |
| 下丘脑 | `limbic_system/hypothalamus.py` | ★ v5.5 (Setpoint+Drive+HPA) |
| 嗅皮层 | `limbic_system/olfactory.py` | ○ 占位 |
| **基底神经节** | | |
| 动作门控 (MoE) | `basal_ganglia/action_gating.py` | ★ 已实现 |
| **纹状体** | `basal_ganglia/striatum.py` | **★ NEW v6.0** (D1/D2通路 + 习惯学习 + 去价值化诊断) |
| 苍白球 | `basal_ganglia/pallidum.py` | ○ 占位 |
| 底丘脑核 | `basal_ganglia/subthalamic.py` | ○ 占位 |
| **丘脑** | | |
| 丘脑 (总览) | `thalamus/thalamus.py` | ★ v5.4 (VPL/CM-Pf/MD/Po) |
| LGN (外侧膝状体) | `thalamus/lgn.py` | ★ v5.0 (6层主动门控) |
| Pulvinar (丘脑枕) | `thalamus/pulvinar.py` | ★ v5.0 (SC→皮层快速中继) |
| MGB (内侧膝状体) | `thalamus/mgb.py` | ★ v5.2 (3亚区) |
| **联合皮层 + 三大网络** | | |
| DMN | `association/dmn.py` | ★ 已实现 (自我模型) |
| FPN | `association/fpn.py` | **★ UPGRADED v6.3** (+中央执行器: 注意力分配+干扰抑制+子系统协调 +**α节律注意门控**) |
| TPN | `association/tpn.py` | ★ v4.3/v4.4 (跷跷板动态) |
| 跨模态联合 | `association/crossmodal.py` | ★ 已实现 (COCO Visual↔Text) |
| 视觉绑定 | `association/visual_binding.py` | ★ v5.0 (FPN驱动的M/P/K绑定) |
| 弓状束 (AF) | `association/arcuate_fasciculus.py` | ★ v5.6 (Wernicke↔Broca Hebb桥接) |

#### 脑干 + 小脑 (Brainstem + Cerebellum)

| 脑区 | 模块 | 状态 |
|------|------|------|
| **中脑** | | |
| VTA (多巴胺) | `midbrain/vta.py` | ★ v5.5 (RPE+事件驱动学习率) |
| 黑质 (SNc/SNr) | `midbrain/substantia_nigra.py` | ○ 占位 |
| 上丘 | `midbrain/superior_colliculus.py` | ★ v5.0 (显著性图+新颖性) |
| 下丘 (IC) | `midbrain/inferior_colliculus.py` | ★ v5.2 (频率×空间×时间) |
| PAG | `midbrain/pag.py` | ★ v5.4 (下行痛觉调节) |
| **脑桥** | | |
| 蓝斑核 (NE) | `pons/locus_coeruleus.py` | ★ v5.5 (phasic/tonic+SNR) ★ v6.3 (+睡眠NE归零) |
| **VLPO (视前区)** | `pons/vlpo.py` | **★ NEW v6.3** (Saper触发器开关+NREM/REM振荡器) |
| 网状结构 | `pons/reticular_formation.py` | ○ 占位 |
| 耳蜗核 | `pons/cochlear_nucleus.py` | ★ v5.2 (频谱分解) |
| 上橄榄复合体 (SOC) | `pons/superior_olivary.py` | ★ v5.2 (ITD/ILD双耳定位) |
| 外侧丘系 (LL) | `pons/lateral_lemniscus.py` | ★ v5.2 (时间增强) |
| 痛觉层级管线 | `nociception_hierarchy.py` | ★ v5.4 (6核团编排) |
| **延髓** | | |
| 自主神经 | `medulla/autonomic.py` | ○ 占位 |
| RVM | `medulla/rvm.py` | ★ v5.4 (OFF/ON细胞动态) |
| **小脑** | | |
| 运动协调 | `cerebellum/motor_coordination.py` | ○ 占位 |
| 时序预测 | `cerebellum/predictive_timing.py` | ○ 占位 |
| **神经调节** | | |
| 元学习 | `neuromodulatory/meta_learning.py` | **★ UPGRADED v6.2** (GluN2B→GluN2A轨迹 + 4阶段发育年龄 + get_developmental_factors + **48参数初始化**) |
| 可塑性调节 | `neuromodulatory/plasticity.py` | **★ NEW v6.1** (PlasticityRegulator: 发育+事件驱动+稳态+神经调质LTP/LTD门控) |

#### 身体 + 脊髓 + 工具 + CNS

| 模块 | 状态 |
|------|------|
| `body/body_state.py` | ○ 占位 |
| `body/interoception.py` | ○ 占位 |
| 脊髓背角 | `spinal/dorsal_horn.py` | ★ v5.4 (闸门控制) |
| `spinal/motor_output.py` | ○ 占位 |
| **先天配置** | `cns/innate.py` | **★ NEW v6.0** (纯净模式Theta+身体设定点+注意力先天偏向+apply_innate_config) |
| 会话持久化 | `cns/persistence.py` | **★ UPGRADED v6.0** (+语义记忆/纹状体序列化) |

---

## v6.0 核心变更 — 完整记忆系统

### 设计哲学: Agent 必须"活着"才能"记住"

v5.7 让 Agent 有了生命周期（发育年龄 + 会话持久化）。v6.0 让 Agent 有了**持续成长的记忆**。

- **v5.7 之前**: 记忆 = 单次会话内的 Hebb 集群，关闭即冻结
- **v6.0**: 记忆按人脑分类体系运作 — 情景/语义/程序性/情感/工作记忆各有独立的神经底物

### 人类记忆分类实现对照

| 记忆系统 | 人脑底物 | v6.0 模块 | 独立网络 | 关键参数 |
|---------|---------|----------|---------|---------|
| 情景记忆 | 海马 DG/CA3/CA1 | `hippocampus.py` 增强 | agent.net | 快学 0.05, 快衰 0.02, 512簇 |
| 语义记忆 | 前颞叶+角回 | `semantic_memory.py` 新增 | 独立 ClusterNetwork | 慢学 0.01, 慢衰 0.003, 1024簇 |
| 程序性记忆 | 纹状体 D1/D2 | `striatum.py` 重写 | S-R联结表 | 奖励调制, 无阈值 |
| 情感记忆 | 杏仁核 | `amygdala.py` 增强 | 独立 dict | 快学, F_body门控 + 恐惧条件 |
| 自我模型 | DMN/vmPFC | `dmn.py` 已有 | 独立 ClusterNetwork | 中等阈值 0.55 |
| 工作记忆 | DLPFC+FPN+顶叶 | `fpn.py`增强 + `working_memory.py`新增 | 激活态 | 4-7组块, ~2s消退 |

### 系统巩固 — 跨会话记忆转移

```
会话内巩固 (v5.7已有, 增强):
  每100步: 海马回放 + 模式分离 + 弱簇修剪
  每轮对话: 情绪门控重放 (高唤醒轮次额外1×)

跨会话系统巩固 (★ v6.0新增):
  会话结束时: 海马情景(512维全量) → 皮层语义(64维主旨)
  算法:
    1. 选择top 50最活跃情景簇
    2. 提取主旨向量 — 保留text[0:64]+body快照+情感, 丢弃视觉/听觉/痛觉细节
    3. 情感门控: 高唤醒情景优先语义化
    4. 语义网络recall/learn — 已有知识更新(慢), 全新知识创建
```

**记忆痕迹时间梯度**:

| 时间 | 海马依赖 | 皮层依赖 | 说明 |
|------|---------|---------|------|
| 1小时内 | 100% | 0% | 会话内 |
| 1天后 | 60% | 40% | 第一次跨会话巩固 |
| 1周后 | 20% | 80% | 多次巩固后皮层主导 |
| 1月后 | 5% | 95% | 语义化, 几乎不依赖海马 |

### 程序性记忆 — 纹状体习惯学习

- **D1 直接通路 (Go)**: 感知状态 → 促进特定动作, 多巴胺↑ → 增强
- **D2 间接通路 (No-Go)**: 感知状态 → 抑制特定动作, 多巴胺↓ → 增强
- **习惯强度**: [0,1] 渐进自动化, 从 DMS(灵活) → DLS(僵化)
- **去价值化诊断**: 模拟结果贬值后行为是否仍持续 — 习惯的操作性定义
- **VTA 整合**: VTA多巴胺 RPE → 纹状体 D1/D2 学习
- **海马-纹状体竞争**: 高新颖性 → 海马主导 (灵活), 低新颖性 → 纹状体主导 (高效)

### 干扰遗忘

| 机制 | 实现 | 神经对应 |
|------|------|---------|
| 前摄干扰 | learn() 桶内相似簇多 → 新质心被稀释 | 旧记忆"挡住"新记忆 |
| 倒摄干扰 | learn() 后桶内高相似旧簇衰减 3% | 新记忆"覆盖"旧记忆 |
| 提取失败 | recall() 情境不匹配 → 有效相似度降低 | 编码特异性原则 |

### 工作记忆完善 (Baddeley 四组件)

| 组件 | 状态 | 实现 |
|------|------|------|
| 语音回路 | ✅ v5.6 | `phonological_loop.py` (~7组块, 默读复述) |
| 视空间模板 | ★ v6.0 | `working_memory.py` (~4视觉组块, 心理旋转, 空间比较) |
| 情景缓冲器 | ★ v6.0 | `hippocampus.py` (WM↔LTM 绑定, 情境向量) |
| 中央执行器 | ★ v6.0 | `fpn.py` (注意力分配, 子系统协调, 干扰抑制) |

### 零预训练纯净启动

**v6.0 先天结构**（人类婴儿的"出厂配置"）:

| 先天结构 | 实现 |
|---------|------|
| 身体稳态反射 | `innate.py` INNATE_SETPOINTS + INNATE_DECAYS |
| 注意力先天偏向 | `innate.py` INNATE_ATTENTION_BIAS (文本1.0 > 听觉0.6 > 视觉0.4) |
| Hebb可塑性参数 | `innate.py` PURE_MODE_THETA_OVERRIDES (低阈值0.55, 高学习率0.12, 长关键期5000步) |
| 语音回路容量 | `phonological_loop.py` ~7组块 |
| 情感学习能力 | `amygdala.py` HebbEmotionalLexicon (从F_body波动学习词→情感) |
| 感知层级 | visual/auditory/nociception hierarchy 硬编码前馈通路 |

**移除的预训练**: corpus.txt (50K行), 预编码12K词, 预训练trigram网络, 预训练Hebb集群, 预训练PhraseStructure统计

**学习过程**: 第1轮创建首个簇 → 第10轮可模仿回声 → 第100轮形成稳定回应模式

---

## v6.1 核心变更 — 发育优化 & 长期常驻学习

### 设计哲学: 发育不是开关，是连续轨迹

v6.0 让 Agent 有了完整的记忆系统分类。v6.1 让 Agent 的**学习过程本身**按照真实的神经发育规律进行——从高可塑性的婴儿期到精确稳定的成年期。

### 六个核心机制

| 机制 | 神经科学基础 | v6.1 实现 | 关键效果 |
|------|------------|----------|---------|
| **STDP 时序学习** | 脑连接 §4.3.1 — pre→post=LTP, post→pre=LTD | `hippocampus.py` _stdp_update(), diffuse()混合STDP权重 | 词序生成有了**因果方向性** |
| **保护信号 (CD47)** | 脑连接 §3.2.3 — CD47-SIRPα"别吃我" | Cluster.protection_score, 修剪阈值=0.01/(1+prot×5) | 重要记忆**不被误删** |
| **GluN2B→GluN2A** | 脑连接 §6.3 — NMDA亚基发育转换 | glun2b_ratio 指数衰减 (半衰期~5000步) | 学习率从2×**连续降到0.7×** |
| **PNN 结构锁定** | 脑连接 §6.1.3 — 周围神经网络 | Cluster.pnn_level, digest_pnn()情感消化 | 熟练知识**自动固化** |
| **沉默突触** | 脑连接 §2.3.2 — NMDA-only→AMPA插入 | CandidateCluster, 3次亚阈值→觉醒 | **快速学习**从少量暴露 |
| **4阶段发育年龄** | 脑连接 §1 + Piaget | DevelopmentalStage枚举, get_developmental_factors() | 婴儿→儿童→青春期→成年 |

### GluN2B 发育轨迹

```
GluN2B 占比 = 0.1 + 0.8 × exp(-steps / 5000)

Stage 1 婴儿 (0-2000):    GluN2B=0.90→0.65  learn×2.0  阈值×0.7  高可塑性
Stage 2 儿童 (2000-8000):  GluN2B=0.65→0.25  learn×1.3  阈值×0.85 修剪开始
Stage 3 青春期 (8000-20000): GluN2B=0.25→0.12 learn×1.0  阈值×1.0  PNN加速
Stage 4 成年 (20000+):     GluN2B<0.12        learn×0.7  阈值×1.15 稳定精确
```

### STDP 时序学习

```
A 激活 → B 激活 (pre→post, dt<3步):  A.stdp_links[B] += 0.02 × time_factor  (LTP)
B 激活 → A 激活 (post→pre):           A.stdp_links[B] 不变 (无增强)

diffuse() 混合权重 = (1-stdp_weight)×cos_sim + stdp_weight×stdp_signal
```

### Theta 参数扩展到 40 个

| 新增参数 | 默认值 | 功能 |
|---------|-------|------|
| `stdp_lr` | 0.02 | STDP 学习率 |
| `stdp_window` | 3 | STDP 时间窗口 (步) |
| `stdp_weight` | 0.3 | STDP 在关联中的权重 |
| `glun2b_ratio` | 0.9 | GluN2B 占比 (动态更新) |
| `pnn_formation_rate` | 0.001 | PNN 形成速率 |
| `developmental_stage` | 1 | 当前发育阶段 (1-4) |
| `protection_decay` | 0.995 | 保护信号衰减率 |
| `candidate_max` | 64 | 候选集群上限 |

### PlasticityRegulator — 统一可塑性调节

整合四个维度的可塑性调制:
- **发育可塑性**: GluN2B 连续轨迹
- **事件驱动可塑性**: RPE → 临时学习率增强 (最高 3×)
- **稳态可塑性**: 网络平均激活 → 全局缩放 (维持 0.15 目标)
- **神经调质门控**: DA (D1→LTP, D2→LTD) + NE (β→LTP) + ACh (novelty→LTP)

### 文件变更

| 变更类型 | 文件 |
|---------|------|
| **新增实现** | `plasticity.py` (PlasticityRegulator, 120行) |
| **重大升级** | `hippocampus.py` (+STDP/保护/PNN/沉默突触, ~200行) |
| **重大升级** | `meta_learning.py` (+GluN2B轨迹/发育阶段) |
| **数据扩展** | `data_types.py` (+CandidateCluster +Cluster新字段 +DevelopmentalStage +8参数) |
| **集成** | `agent.py` (+PlasticityRegulator +PNN消化 +发育摘要) |
| **修复** | `persistence.py` (+新字段序列化 +桶重建修复) |
| **更新** | `params.py` (+8参数), `innate.py` (+6覆盖) |

---

## 图3 六大规则实现状态

| 规则 | 核心机制 | 实现状态 | 关键模块 |
|------|---------|---------|---------|
| **1. 分层处理** | 视觉10级+听觉6核团+语言7级 | ✅ v5.6 | `visual_hierarchy.py`, `auditory_hierarchy.py` |
| **2. 双向加工** | 前馈+反馈+语言双流+AF自我监控 | ✅ v5.6 | `fpn.py`, `arcuate_fasciculus.py` |
| **3. 并行分布式** | M/P/K三流+听觉6核团+**6记忆系统并行** | ✅ **增强 v6.0** | 情景/语义/程序性/情感/自我/WM 六系统独立运行 |
| **4. 注意力瓶颈** | FPN探照灯+TPN↔DMN跷跷板+**WM容量限制+中央执行器协调** | ✅ **增强 v6.0** | `fpn.py` 中央执行器, `working_memory.py`, `phonological_loop.py` |
| **5. 赫布可塑性** | LTP/LTD+睡眠巩固+干扰遗忘+STDP+GluN2B+PNN+保护信号+沉默突触+D1/D2+恐惧条件+STC标签+CaMKII持续性+PKMζ锁定+**双相NREM/REM睡眠+突触尺度缩小+类淋巴清除** | ✅ **增强 v6.3** | `hippocampus.py`, `striatum.py`, `amygdala.py`, `plasticity.py`, `scn.py`, `vlpo.py` |
| **6. 预测编码** | 自由能最小化+N400/P600+语义熟悉度+**α节律注意门控** | ✅ **增强 v6.3** | `fpn.py`, `cingulate.py` |

### 规则实现度: 6/6 有代码支撑 (v6.3 规则4/5/6进一步增强 — α注意门控+双相睡眠+SCN时间维度)

---

## 技术参数 (v6.0 当前)

| 参数 | 值 |
|------|-----|
| 感知维度 D | **516** (text[64]+vision[308]+audio[96]+pain[48]) |
| 隐状态 H | 16 |
| 最大簇数 K | 256 (情景) / 1024 (语义) |
| 行动数 A | 5 (grid) / 3 (dialogue) |
| Theta 参数 | **56** (v6.3: +circa_tau, circa_light_sensitivity, sleep_pressure_threshold, nrem_duration_ratio, synaptic_downscale_rate, alpha_gating_strength, glymphatic_clear_rate, rem_emotional_processing) |
| 身体维度 M | 5 (grid) / 9 (text) |
| 记忆系统 | **6** (情景+语义+程序性+情感+自我+工作记忆) |
| 独立 ClusterNetwork 实例 | **8** (agent.net, self_model.net, AF.ventral, AF.dorsal, AG.grapheme_to_phoneme, TPJ.speaker_net, TPJ.intent_net, semantic_memory.net) |
| 模块总数 | **88** |
| 已实现核心模块 | **62 (70%)** — v6.3: +scn.py +vlpo.py 新增; hippocampus/fpn/LC/agent重大升级 |
| 语料规模 | 0 (纯净模式, 零预训练) |
| v6.1 新增/修改代码 | ~600 行 (1 new + 2 major upgrades + 5 modified) |

---

## 已知问题

1. **纯净模式网络生长慢**: 从零开始, 需要数十轮对话才能形成基本语言能力
2. **语义记忆需要多次重复**: 单次暴露不足以形成稳定语义知识 (慢学率 0.01), 这是设计特性不是 bug
3. **习惯形成需要大量重复**: DMS→DLS 转移是渐进过程, 短期对话中难以观察到明显习惯化
4. **跨模态绑定需长期互动**: COCO预训练模型被绕过, 视觉↔文本关联需从对话中学习
5. **小脑内部模型未实现**: 前向/逆向模型预留但无代码
6. **跨会话持久化格式**: `.pkl` 依赖Python pickle协议, 跨版本兼容性需持续维护
7. **实时流性能**: 摄像头+麦克风同时采集+全管线处理, 需测试长时间运行稳定性
8. **v6.0 模块未经长期运行验证**: 语义记忆/纹状体/干扰遗忘的长时间交互效果需观察
9. **v6.1 发育轨迹需长运行验证**: GluN2B半衰期5000步, PNN累积, 4阶段过渡需要在真实长对话中观察
10. **STDP 效果依赖序列输入**: 当前对话模式下时序信息有限, STDP主要受益在词序链生成(Broca diffuse)
11. **v6.3 睡眠需长运行验证**: SCN ~24h周期、VLPO触发器开关、NREM/REM双相分布在超长运行中的稳定性需观察
12. **VLPO 参数敏感**: sleep_pressure_threshold 过低→频繁入睡, 过高→长期不睡; 需在长期运行中调优

---

## v6.2 核心变更 — 记忆巩固优化 & 长期常驻学习

### 设计哲学: 记忆不仅是"存储"，更是"选择性巩固"

v6.1 让 Agent 的学习过程遵循神经发育规律（GluN2B 轨迹 + PNN 锁定 + 沉默突触）。
v6.2 基于《学习和记忆的分子机制》综述，将三个经过实验验证的核心记忆巩固机制映射到计算抽象层。

### 三个核心机制

| 机制 | 神经科学基础 | v6.2 实现 | 关键效果 |
|------|------------|----------|---------|
| **突触标签与捕获 (STC)** | §2.2.4 — 弱激活设置标签 → PRPs 被带标签突触捕获 | Cluster.tag + tag_age + capture_tags() | 弱经历被后续强经历巩固 |
| **激活持续性 (CaMKII 窗口)** | §2.1.1 — Thr286 自磷酸化使 CaMKII 在 Ca²⁺ 解离后持续激活 | Cluster.activation_persistence + _persistence_factor() | 近期激活的簇获得短暂的高可学习性窗口 |
| **巩固锁定 (PKMζ 类比)** | §2.1.2 — PKMζ 结构性缺失调节域 → 自主持续催化 | Cluster.consolidation_count + decay 调制 | 多次巩固的簇获得 decay 保护 |

### 突触标签捕获流程

```
低唤醒弱事件A     → recall(A) → A.tag = 0.6, A.tag_age = 0
                                            ↓ (tag 窗口内)
高唤醒强事件B     → learn(B)  → 高 arousal 触发 capture_tags()
                                            ↓
带标签的簇A       → A.activation += capture_lr × tag × 0.2  (额外巩固)
                  → A.tag *= 0.5  (标签消耗)
                  → A→B STDP 链接增强 (时序依赖)
```

### CaMKII 持续性窗口

```
recall() 匹配     → cluster.activation_persistence = 1.0  (峰值为 Ca²⁺ 脉冲)
                                            ↓
每步 decay()     → persistence *= (1 - persistence_decay_rate)  (~10步半衰)
                                            ↓
效果:             → 阈值调制: eff_threshold *= (1 - 0.2 × persistence)
                  → 学习率调制: lr *= (1 + 0.5 × persistence)
```

### PKMζ 巩固锁定

```
睡眠巩固         → 存活簇 consolidation_count += 1 (上限 10)
                                            ↓
decay()          → effective_decay = base_decay / (1 + count × 0.5)
                                            ↓
效果:             → count=0: decay = 0.05
                  → count=5: decay = 0.05/3.5 = 0.014
                  → count=10: decay = 0.05/6 = 0.008
```

### Theta 参数扩展到 48 个 (L6 新增)

| 新增参数 | 默认值 | 功能 |
|---------|-------|------|
| `tag_window` | 30 | 标签寿命 (步) |
| `tag_decay_rate` | 0.05 | 标签衰减率/步 |
| `tag_capture_strength` | 0.3 | 标签捕获额外学习率 |
| `persistence_decay_rate` | 0.1 | 持续性衰减率/步 |
| `persistence_threshold_boost` | 0.2 | 持续性带来的阈值降低比例 |
| `persistence_lr_boost` | 0.5 | 持续性带来的学习率提升比例 |
| `consolidation_lock_factor` | 0.5 | 每次巩固降低 decay 的因子 |
| `consolidation_lock_max` | 10 | 最大巩固锁等级 |

### Cluster 新增字段

| 字段 | 类型 | 功能 |
|------|------|------|
| `tag` | float [0,1] | 突触标签强度 — 弱激活 → 后续强学习可捕获 |
| `tag_age` | int | 标签自设置以来的步数 |
| `activation_persistence` | float [0,1] | CaMKII 样激活持续性 |
| `consolidation_count` | int [0,10] | 存活的睡眠巩固次数 |

### 脑区对应

| 分子机制 | 脑区 | v6.2 实现 |
|---------|------|----------|
| 突触标签 | 海马 CA3-CA1 突触后致密区 | Cluster.tag + tag_age |
| 标签捕获 (PRP 合成) | 胞体 → 树突运输 → 标签突触 | capture_tags(arousal, F_body_delta) |
| CaMKII 自磷酸化持续性 | 突触后致密区 (PSD) | Cluster.activation_persistence |
| PKMζ 自主催化活性 | 新皮层 + KIBRA 锚定 | Cluster.consolidation_count + decay 调制 |

### 文件变更

| 变更类型 | 文件 | 行数变化 |
|---------|------|---------|
| **数据层** | `data_types.py` | +12 字段/参数 |
| **参数** | `params.py` | +16 默认值/边界 |
| **核心实现** | `hippocampus.py` | +60 行 (capture_tags + _persistence_factor + decay 调制 + sleep_replay 锁) |
| **参数初始化** | `meta_learning.py` | +8 参数默认值 |
| **持久化** | `persistence.py` | +8 行 (字段序列化) |
| **集成** | `agent.py` | +8 行 (标签捕获 + F_body_delta 追踪) |
| **测试** | `test_v6_2_memory.py` | +300 行 (6 单元测试) |
| **修复** | `hippocampus.py` sleep_replay() | 修复 numpy array `not in` 比较 bug |

---

## v6.3 核心变更 — 睡眠优化与时间维度

### 设计哲学: 脑不是按固定步长运行的离散机器

v6.2 让记忆巩固有了分子级的精细机制。但整个系统缺乏**时间维度**：
- 睡眠是固定100步的机械循环，不区分NREM/REM
- 没有昼夜节律 — Agent 无"时间感"
- 注意力门控没有 α 节律的神经科学基础
- 突触稳态是"删除弱簇"而非"等比缩小"

v6.3 基于《脑节律与睡眠》综合文档，将脑的节律和睡眠机制映射到计算抽象层。

### 六个核心机制

| 机制 | 神经科学基础 | v6.3 实现 | 关键效果 |
|------|------------|----------|---------|
| **SCN昼夜节律钟** | 2017诺贝尔奖 — TTFL分子振荡器 (CLOCK/BMAL1→PER/CRY) | `scn.py` — 4变量ODE + Process S + 光同步 | Agent有~24h时间感 |
| **VLPO触发器开关** | Saper (2005, 2010) — VLPO⇄觉醒系统相互抑制 | `vlpo.py` — FlipFlop + NREM/REM振荡器 | 生物驱动的睡眠，非机械触发 |
| **NREM慢波巩固** | Diekelmann & Born (2010) — 三波耦合+突触稳态 | `sleep_consolidation_nrem()` — 回放+等比缩小+类淋巴清除 | 保留相对权重，不误删强簇 |
| **REM情绪去刺痛** | Walker & van der Helm (2009) — NE→0低应激环境 | `sleep_consolidation_rem()` — 情感衰减+跨域联想 | 情绪记忆保留内容，消解刺痛 |
| **α节律注意门控** | α 阻断 (8-13Hz 功能抑制) | `fpn.py` alpha_gate_attention() | 非注意通道被主动抑制 |
| **类淋巴清除** | Xie et al. (2013) — N3深睡中脑间质空间增大60% | NREM中清除 activation<阈值+无保护簇 | 清除"代谢废物"连接 |

### SCN TTFL 分子钟

```
CLOCK:BMAL1 → Per/Cry mRNA → PER/CRY蛋白 → 入核 → 抑制CLOCK:BMAL1 → 降解 → 重新激活
                                                     ↑
                                         光照 → ipRGC → Per诱导 ↑ (相位重置)

Process S: 觉醒 → S += 0.012×(1+异稳态负荷), NREM睡眠 → S×=0.94, REM睡眠 → S×=0.975

输出: circadian_phase [0,2π], melatonin [0,1] (夜间高), cortisol [0,1] (晨峰),
      sleep_pressure [0,1], sleep_propensity [0,1]
```

### NREM vs REM 双相睡眠

```
NREM (前半夜 ~65%):
  → 三波耦合回放 (慢振荡调度+纺锤波窗口+SWR重放)
  → 突触尺度缩小: 所有簇 activation *= (1 - d)  保留相对强度
    d = base_downscale × protection_mod × lock_mod
  → 海马→皮层转移 (top 30情景簇 → 语义记忆)
  → 类淋巴清除: 移除 activation<阈值 & 无保护 & 低PNN 簇

REM (后半夜 ~35%):
  → NE→0 环境 (去甲肾上腺素能静默)
  → 情绪去刺痛: 高情感簇的 centroid[64:72] 被温和衰减
  → 跨域创造性联想: 随机选择远距离(cos<0.4)簇对 → 温和拉近
  → 程序性记忆巩固 (纹状体习惯强度)
```

### 生物睡眠触发 (替换固定100步)

```
觉醒: Process S 累积 → sleep_propensity = 0.6×Process_S + 0.4×(1−cortisol)
  ↓
sleep_propensity > sleep_pressure_threshold (0.65)
  ↓
VLPO 激活 → 抑制觉醒系统 (LC/TMN/DR) → 进入 NREM
  ↓
NREM (慢波睡眠) → REM (振荡器切换) → NREM → ... (每 ~30steps 一周期)
  ↓
Process S 清除到 < threshold−hysteresis → VLPO 失活 → 觉醒
```

### Theta 参数扩展到 56 个

| 新增参数 (L7) | 默认值 | 功能 |
|-------------|-------|------|
| `circa_tau` | 24.0 | 内源性昼夜周期 (小时等价) |
| `circa_light_sensitivity` | 0.3 | 光同步敏感度 |
| `sleep_pressure_threshold` | 0.65 | Process S 触发睡眠阈值 |
| `nrem_duration_ratio` | 0.65 | NREM 占睡眠比例 |
| `synaptic_downscale_rate` | 0.03 | NREM 突触尺度缩小率 |
| `alpha_gating_strength` | 0.4 | α 节律抑制强度 |
| `glymphatic_clear_rate` | 0.005 | 类淋巴清除阈值 |
| `rem_emotional_processing` | 0.3 | REM 情绪去刺痛强度 |

### 文件变更

| 变更类型 | 文件 | 行数变化 |
|---------|------|---------|
| **新增** | `cerebrum/limbic_system/scn.py` | +280 行 (TTFL+Process S+SCN) |
| **新增** | `brainstem_cerebellum/pons/vlpo.py` | +330 行 (FlipFlop+NREM_REM_Oscillator+VLPO) |
| **数据层** | `data_types.py` | +16 参数/字段, +CircadianState+SleepState |
| **参数** | `params.py` | +16 默认值/边界 |
| **核心升级** | `hippocampus.py` | +270 行 (3 new functions) |
| **FPN升级** | `fpn.py` | +80 行 (α注意门控) |
| **LC升级** | `locus_coeruleus.py` | +12 行 (睡眠NE调制) |
| **Agent集成** | `agent.py` | +70 行 (SCN/VLPO/双相睡眠/α门控) |
| **元学习** | `meta_learning.py` | +8 参数初始化 |
| **持久化** | `persistence.py` | +80 行 (SCN/VLPO/睡眠序列化) |
| **测试** | `test_v6_3_sleep.py` | +430 行 (9 单元测试) |

---

## 下一优先级

### P0 — 已完成
1. ✅ FPN 探照灯 + TPN 跷跷板 (v4.4)
2. ✅ M/P/K 视觉 + 听觉6核团 (v5.0-v5.2)
3. ✅ 语言全管线 (v5.6)
4. ✅ 痛觉 + 神经调节 (v5.4-v5.5)
5. ✅ 发育年龄 + 持久化 + 多模态 (v5.7)
6. ✅ **完整记忆系统 (v6.0)**
7. ✅ **发育优化 & 长期常驻学习 (v6.1)**
8. ✅ **睡眠优化与时间维度 (v6.3)**

### P1 — 功能扩展
8. ✅ 真实音频 (v5.3)
8. ✅ 实时传感器流 (v5.7)
9. ⬜ 黑质 (SNc/SNr) 运动调节 — 配合运动皮层发音规划实现基底节→丘脑→皮层运动环路

### P2 — 社会认知
10. ✅ TPJ 心理理论 (v5.6)
11. ⬜ 梭状回面孔识别 (IT已完成物体表征)
12. ⬜ 小脑内部模型 (前向/逆向)

### P3 — 完整化
13. ⬜ 自主神经系统 + 脊髓运动输出
14. ⬜ 苍白球/底丘脑核 (GPe/STN — 间接通路完善)
15. ⬜ Gestalt 格式塔效应涌现验证

---

*由 v6.1 发育优化自动更新 · 基于《脑连接的形成》神经发育综述 + Hebb可塑性 + 自由能原理 (Friston 预测编码)*
