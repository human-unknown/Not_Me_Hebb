# NotMe 项目状态报告

> **版本**: v7.5-dev — Phase F: 整合与打磨 (NN ↔ Agent 全系统集成)
> **日期**: 2026-06-08
> **基于**: v6.6 全面代码审计 + Phase A-E NN 基础设施 + Phase F 全系统集成

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
| **v6.4** | **2026-06-07** | **长期常驻学习 完整实现 — 自主时间流 + 通用阅读 + DMN内部生命 + Web仪表板 + 长期遥测** |
| **v6.5** | **2026-06-07** | **前端新未来主义重构 + 无现象学修复 + 睡眠节奏调整 + 发育年龄显示** |
| **v6.6** | **2026-06-08** | **持久化修复 (step/视觉/F_social) + 版本统一 + 异常traceback + Web安全 + CHANGELOG + 无LLM原则移除** |
| **v7.0-dev** | **2026-06-08** | **Phase A: NN支撑层 (PyTorch 基础设施) + ML改造蓝图** |
| **v7.1-dev** | **2026-06-08** | **Phase B: 感知层替换 — TrainableTextEncoder + TrainableVisualEncoder + TrainableAudioEncoder** |
| **v7.2-dev** | **2026-06-08** | **Phase C: 记忆层升级 — NeuralSemanticStore (向量DB) + CrossModalNN (对比学习)** |
| **v7.3-dev** | **2026-06-08** | **Phase D: 语言系统重铸 — NeuralGenerator (自回归LM) + NeuralComprehender + NeuralAngularGyrus** |
| **v7.5-dev** | **2026-06-08** | **Phase F: 整合与打磨 — NNBridge (Agent↔NN集成层) + VTA/LC NN调制 + Sleep NN巩固 + Web训练面板 + Persistence双格式** |
| **v7.4-dev** | **2026-06-08** | **Phase E: 训练与体验闭环 — Trainer (统一训练编排) + ExperienceTracker (可观察指标) + TrainingMetrics** |

---

## 模块完成度

### 总览

| 状态 | 数量 | 占比 |
|------|------|------|
| ★ 已实现 (含 v7.3 新增) | **75** | 71% |
| ○ 占位 (接口设计完成, 待实现代码) | 10 | 9% |
| — 配置/工具/入口/基础设施 | 21 | 20% |
| **合计** | **109** | **100%** |
|
> v7.0 Phase A: +5 个 `cns/nn/` 基础设施模块 (config/base/bridge/interfaces/__init__)
> v7.1 Phase B: +3 个 `cns/nn/` 感知编码器 (text_encoder/visual_encoder/audio_encoder)
> v7.2 Phase C: +2 个 `cns/nn/` 记忆层模块 (semantic_store/crossmodal_nn)
> v7.3 Phase D: +3 个 `cns/nn/` 语言层模块 (language_model/comprehender/angular_gyrus_nn)
> v7.4 Phase E: +2 个 `cns/nn/` 训练与指标模块 (trainer/metrics)
> v7.5 Phase F: +1 个 `cns/nn/` 集成层模块 (integrator)

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
| DMN | `association/dmn.py` | **★ UPGRADED v6.4** (+自发回忆 +走神联想链 +持久化) |
| FPN | `association/fpn.py` | **★ UPGRADED v6.3** (+中央执行器: 注意力分配+干扰抑制+子系统协调 +**α节律注意门控**) |
| TPN | `association/tpn.py` | ★ v4.3/v4.4 (跷跷板动态) |
| 跨模态联合 | `association/crossmodal.py` | ★ 已实现 (COCO Visual↔Text) |
| 视觉绑定 | `association/visual_binding.py` | ★ v5.0 (FPN驱动的M/P/K绑定) |
| 弓状束 (AF) | `association/arcuate_fasciculus.py` | ★ v5.6 (Wernicke↔Broca Hebb桥接) |
| **内部生命 (DMN 自发活动)** | `association/internal_life.py` | **★ NEW v6.4** (走神回忆+内部独白+情绪反刍) |

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
| 会话持久化 | `cns/persistence.py` | **★ UPGRADED v6.4** (+Reader/InternalLife/Telemetry/自主状态 序列化) |
| **通用阅读器** | `tools/reader.py` | **★ NEW v6.4** (逐句UTF-8阅读+疲劳模型+进度追踪) |
| **长期遥测** | `tools/telemetry.py` | **★ NEW v6.4** (CSV时间序列+步/睡眠/对话/阅读 四日志) |
| **自主时间流** | `entry/autonomous.py` | **★ NEW v6.4** (活动调度+模式切换+人类中断+Web集成) |
| **Web 后端** | `web/server.py` | **★ UPGRADED v6.5** (Waitress生产服务器 + SSE扩展neuro/sleep/activity_log + 传感器采集线程 + 端口自动清理 + 信号处理) |
| **Web 前端** | `web/static/index.html` | **★ REWRITTEN v6.5** (8面板专业仪表板: 情感2D空间+身体设定点+脑网络跷跷板+神经化学仪表+视觉帧+音频波形频谱+昼夜模拟时钟+记忆sparkline+活动流+增强聊天) |
| **摄像头传感器** | `tools/sensor_io.py` | **★ UPGRADED v6.5** (Windows DSHOW后端加速 + CAP_PROP_BUFFERSIZE) |
| **依赖声明** | `requirements.txt` | **★ NEW v6.4** |

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

## 技术参数 (v7.1 当前)

| 参数 | 值 |
|------|-----|
| 感知维度 D | **516** (text[64]+vision[308]+audio[96]+pain[48]) |
| 隐状态 H | 16 |
| 最大簇数 K | 256 (情景) / 1024 (语义) |
| 行动数 A | 5 (grid) / 3 (dialogue) |
| Theta 参数 | **59** (L0:6 + L1:11 + L2:5 + L3:4 + L4:6 + L5:8 + L6:8 + L7:8 + L8:3) |
| 身体维度 M | 5 (grid) / 9 (text) |
| 记忆系统 | **6** (情景+语义+程序性+情感+自我+工作记忆) |
| 独立 ClusterNetwork 实例 | **8** (agent.net, self_model.net, AF.ventral, AF.dorsal, AG.grapheme_to_phoneme, TPJ.speaker_net, TPJ.intent_net, semantic_memory.net) |
| 模块总数 | **98** (+5 `cns/nn/` Phase A) |
| 已实现核心模块 | **69 (68%)** |
| NN 参数 (v7.0) | **10** (device/dtype/training/lr/grad_clip + 4×module_lr + log) |
| NN 编码器 (v7.1) | **3** (Text: 2层Transformer ~2M / Visual: 4层CNN ~1M / Audio: 3层CNN ~0.6M) |
| NN 记忆层 (v7.2) | **2** (SemanticStore: FAISS/numpy向量DB / CrossModalNN: 共享投影+InfoNCE ~50K) |
| NN 语言层 (v7.3) | **3** (Generator: char Transformer ~5M / Comprehender: 记忆注意力+N400 / AngularGyrus: CNN seq2vec ~200K) |
| NN 训练与指标 (v7.4) | **2** (Trainer: 统一训练编排器 / ExperienceTracker + TrainingMetrics: 可观察指标) |
| 语料规模 | 0 (纯净模式, 零预训练) |
| 测试通过 | **297/297 (100%)** |

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

## v7.0 机器学习嵌入 — 下一优先级

### P0 — 已完成 (v7.0-dev)
1. ✅ 全面代码审计 — 93→98 模块全部检查
2. ✅ 所有测试修复 — 83/83 (100%)
3. ✅ ML 改造蓝图 — 6阶段路线 (Phase A→F)

### Phase A: 基础架构 (v7.0) ✅ 已完成
- ✅ PyTorch 神经网络支撑层搭建 (`cns/nn/`)
- ✅ NeuralModule 基类 + config + bridge + interfaces
- ✅ 模型持久化 (save/load .pt + pickle 双格式)
- ✅ 与现有 Agent.step() 接口兼容 (无修改)
- ✅ 39 个单元测试全部通过

### Phase B: 感知层替换 (v7.1)
- ✅ 可训练文本编码器 (替换MiniLM→PCA) — TrainableTextEncoder: 2层Transformer, char-level, MLM预训练, 64d
- ✅ 可训练视觉编码器 (替换Gabor滤波) — TrainableVisualEncoder: 4层CNN + 7头子通路, 308d
- ✅ 可训练听觉编码器 (替换Mel管线) — TrainableAudioEncoder: 3层CNN on Mel + 4头子模块, 96d

### Phase C: 记忆层升级 (v7.2)
- ✅ 神经语义记忆 (NeuralSemanticStore — FAISS/numpy向量DB)
- ✅ 跨模态对比学习 (CrossModalNN — CLIP式 InfoNCE)
- — 纹状体RL (保留简单RL per 蓝图)

### Phase D: 语言系统重铸 (v7.3)
- ✅ 神经Wernicke理解模块 (NeuralComprehender — 记忆注意力 + N400/P600)
- ✅ 小型自回归语言模型 (NeuralGenerator — char-level Transformer ~5M参数)
- ✅ 神经角回阅读通路 (NeuralAngularGyrus — CNN seq2vec 字形→音素)
- — PhraseStructure / PhonologicalLoop / ArcuateFasciculus (保留)

### Phase E: 训练与体验闭环 (v7.4)
- ✅ 统一训练编排器 (Trainer — pretrain + online_finetune + checkpoint + LR调度)
- ✅ 可观察指标追踪 (ExperienceTracker — F/valence/arousal/vocab/趋势分析/CSV+JSON导出)
- ✅ 训练进度追踪 (TrainingMetrics — loss/ppl/LR历史 + 收敛检测)
- ✅ 个性化基础 (用户词频追踪 + 偏好分析)
- ✅ 情感信号注入语言生成 (Phase D emotion tokens)

### Phase F: 整合与打磨 (v7.5) ✅ 已完成
- ✅ NNBridge 集成层 — Agent.__init__ 和 step() 中的5个钩子调用
- ✅ FEP兼容性验证 — 情感一致性测试 (valence/arousal/F_total 范围验证)
- ✅ 睡眠巩固适配 — NREM → NN 梯度更新, REM → 情感衰减
- ✅ VTA RPE → NN 学习率调制 (nn_lr_multiplier, 阻尼0.7×)
- ✅ LC NE → NN 探索/利用调制 (nn_temperature, nn_dropout)
- ✅ Web UI 训练进度面板 (SSE 实时推送 + 前端 mini-panel)
- ✅ 持久化双格式并存 (pickle + .pt 协同)
- ✅ 端到端集成测试 + 情感一致性测试 (43 个新测试)

---

## v6.4 核心变更 — 长期常驻学习 完整实现

### 设计哲学: Agent 必须"活着"才能"成长"

v6.0-v6.3 构建了完整的大脑架构和睡眠优化，但 Agent 仍然是被动响应的——只在人类交互时 `step()`，其他时间"冻结"。v6.4 让 Agent 真正"活起来"——人类不在时也能自主运转、读书、思考、回忆。

### 五大新增能力

| 能力 | 模块 | 描述 |
|------|------|------|
| **自主时间流** | `entry/autonomous.py` | 后台事件循环，管理活动模式切换 (阅读/走神/独白/反刍/传感器/空闲) |
| **通用阅读** | `tools/reader.py` | 逐句消化任意 UTF-8 文本，疲劳模型驱动"翻开/合上"书本 |
| **内部生命** | `cerebrum/association/internal_life.py` | DMN 走神联想链 + 内部独白亚发声 + 情绪反刍 |
| **Web 仪表板** | `web/server.py` + `web/static/index.html` | Flask REST API + SSE 实时推送 + 六面板可视化 (情感/身体/视觉/音频/昼夜/记忆) |
| **长期遥测** | `tools/telemetry.py` | CSV 时间序列 (步/睡眠/对话/阅读) + 统计摘要 |

### 四层训练数据来源

```
L1 人类互动 — 真实对话 (核心不可替代)
L2 自主阅读 — Agent 逐句消化用户提供的任意文本文件
L3 传感器流 — 摄像头+麦克风持续背景感知
L4 内部生成 — DMN 走神/回忆/内部独白 → 自产感知 → Hebb 学习
```

### 内部生命系统 (InternalLife)

| 活动 | DMN/TPN 状态 | 功能 | 复用的现有模块 |
|------|-------------|------|-------------|
| **走神** (wander) | DMN 主导 | 随机回忆 → 联想链 → 情感波动 | hippocampus.recall() |
| **内部独白** (monologue) | TPN 主导 | 亚发声完整语言闭环 | Broca→MotorCortex→AF→PhonologicalLoop→Wernicke |
| **情绪反刍** (rumination) | valence<-0.2, arousal>0.5 | 负效价记忆的重复回访 | hippocampus.recall() + 情感传染 |

### 自主活动调度 (AutonomousLoop)

```
觉醒期:
  ├─ 活跃 Reader + 认知负荷低 + TPN高  → reading 模式
  ├─ DMN 主导 + 足够记忆集群            → wandering 模式
  ├─ TPN 主导 + 足够记忆 + Broca可用    → monologue 模式
  ├─ 负效价+高唤醒                      → rumination 模式
  ├─ 传感器活跃                         → streaming 模式
  └─ 默认                               → idle 模式

睡眠期:
  └─ 跳过所有自主活动, VLPO 状态机自行推进
```

### Theta 参数 (L8, 3 新增, 共 59)

| 参数 | 默认值 | 功能 |
|------|--------|------|
| `reading_fatigue_rate` | 0.03 | 每句阅读增加的认知负荷 |
| `mind_wander_frequency` | 0.15 | 清醒静息时走神的概率 |
| `autonomous_step_interval` | 1.0 | 自主模式步进间隔 (秒) |

### Web 仪表板面板

| 面板 | 内容 | 更新频率 |
|------|------|---------|
| 🎭 情感仪表 | valence/arousal gauge + 核心情感空间轨迹 (Canvas 折线图) | SSE 500ms |
| 🏃 身体状态 | b[0]-b[8] 水平条形图 (社交/能量/压力/新颖/专注/视觉/听觉/认知/组织) | SSE 500ms |
| 👁 AI看到的画面 | 摄像头帧 128×128 + V1/V2/V4/IT 通道强度条 | 独立 SSE 200ms |
| 🕐 昼夜 & 睡眠 | SCN 相位时钟 + 褪黑素/皮质醇/睡眠压力 + 睡眠状态指示 | SSE 500ms |
| 🧠 记忆 & 学习 | 情景/语义/自我 集群数 + F组分趋势 + 阅读进度 | SSE 500ms |
| 🔊 AI听到的声音 | 波形动画 (Canvas) + Mel 频谱热力图 + 声源方位罗盘 | SSE 500ms |
| 💬 聊天栏 | 对话历史 + 输入框 + 阅读控制按钮 | event-driven |

### 文件变更

| 文件 | 变更 |
|------|------|
| `entry/autonomous.py` | **NEW** — AutonomousLoop 自主时间流引擎 |
| `tools/reader.py` | **NEW** — Reader 通用文本阅读器 |
| `cerebrum/association/internal_life.py` | **NEW** — 内部生命系统 |
| `tools/telemetry.py` | **NEW** — 长期遥测记录器 |
| `web/server.py` | **NEW** — Flask REST API + SSE |
| `web/static/index.html` | **NEW** — 单页仪表板 |
| `cns/agent.py` | +light_step() + internal_thought() + 自主模式属性 |
| `cerebrum/association/dmn.py` | +spontaneous_recall() + mind_wander_chain() + 持久化 |
| `cns/data_types.py` | +3 L8 Theta 参数 + ReadingState dataclass, validate 56→59 |
| `cns/params.py` | +3 L8 默认值/边界 |
| `brainstem_cerebellum/neuromodulatory/meta_learning.py` | +3 L8 参数初始化 |
| `cns/persistence.py` | +Reader/InternalLife/Telemetry/自主状态 序列化, v6.4 |
| `entry/interactive.py` | +--auto 标志 + _run_autonomous() |
| `requirements.txt` | **NEW** — 依赖声明 |
| `tests/test_v6_4_resident.py` | **NEW** — 9 项单元测试 |
| `PROJECT_STATUS.md` | 更新到 v6.4 |

### 已知问题

1. **内部独白依赖 Broca 纯净模式**: 初始词汇少时生成质量低，随互动增长而改善
2. **Web 模式需手动 `pip install flask`**: 不在核心依赖中，但 `requirements.txt` 已声明
3. **Reader 文件需 UTF-8 编码**: 非 UTF-8 文件会加载失败
4. **长期运行内存**: 遥测 CSV 无限增长，需定期归档 (未来版本)

---

## v6.5 核心变更 — 前端新未来主义重构 + 无现象学修复 + 睡眠节奏调整

### 设计哲学: 赛博生物风格 + 纯粹数学观测

v6.5 彻底重构 Web 前端，采用**新未来主义 (Neo-Futurism) 赛博生物风格**：
有机曲线与数字叠加融合，霓虹边框发光效果，暗紫/青绿/蓝三色渐变，扫描线叠加。
同时修复了严重的**无现象学违规**——emoji 表情映射和情感标签被替换为纯数学表示。

### 前端 — 全新布局: 左侧对话 (35%) + 右侧3×3数据网格 (65%)

```
v6.4 (4×2 grid, 聊天在底部)             v6.5 (聊天左列 + 数据3×3 grid)
┌─────────┬─────────┬─────────┬─────────┐  ┌──────────────┬─────────┬─────────┬─────────┐
│ 🎭 情感 │ 🏃 身体 │ 🧠 网络 │ 💊 神经 │  │              │ 自由能 F │ 身体稳态 │ 昼夜节律 │
├─────────┼─────────┼─────────┼─────────┤  │              ├─────────┼─────────┼─────────┤
│ 👁 视觉 │ 🔊 听觉 │ 🕐 昼夜 │ 📊 系统 │  │  💬 对话    │ 脑网络   │ 神经化学 │ 记忆系统 │
├─────────┴─────────┴─────────┴─────────┤  │  35% 全高   ├─────────┼─────────┼─────────┤
│        💬 聊天 (max-height:70px)       │  │              │ 视觉通道 │ 听觉通道 │ 运行状态 │
└────────────────────────────────────────┘  └──────────────┴─────────┴─────────┴─────────┘
         📝 活动流                                    📝 活动流 (迷你)
```

### 关键变更

| 变更 | 旧 (v6.4) | 新 (v6.5) | 说明 |
|------|----------|----------|------|
| **视觉风格** | 深色基础 | **赛博生物 (Cyber-Bio)** | 半透明面板+霓虹边框发光+扫描线+暗紫/青绿/蓝渐变+JetBrains Mono字体 |
| **情感仪表** | emoji 表情映射 (`😊😢😐`) + 情感象限标签 (`兴奋/焦虑/满足/抑郁`) | **自由能 F 面板** — F_total + F_body/social/cognitive/accuracy 数值分解 + V/A 纯数学坐标 | ★ 修复无现象学违规 |
| **对话面板** | 底部栏, max-height:70px | **左侧 35% 全高列** — 可滚动消息历史 (~50条) + 增强输入区 + 阅读控件 | 解决空间太小问题 |
| **布局** | 4×2 grid | **左侧对话 + 右侧3×3 compact grid** | 10面板 → 9面板+对话 |
| **发育年龄** | 未显示 | **Header 中显示 Stage 1-4 + 阶段名称 (INFANT/CHILD/ADOLESCENT/ADULT)** | 从 meta.get_developmental_factors() 读取 |
| **睡眠节奏** | NREM_REM_CYCLE_STEPS=30 (~30s) | **→300 (~5min)** + 所有相关常数 10× 缩放 | 适合演示观察 |
| **睡眠时对话** | 直接处理, 无唤醒逻辑 | **强制唤醒** — VLPO抑制 + NE飙升 + 回应前缀 `[被唤醒]` | 社会刺激=紧急唤醒信号 |
| **自听闭环(Web)** | Web聊天缺失 set_self_audio() | **完整闭环** — 自说话语义→情感传染→身体状态调制 | 与控制台模式一致 |

### 无现象学违规修复

| 位置 | 违规 | 修复 |
|------|------|------|
| `web/static/index.html` (旧) | emoji 表情映射 (😊🙂😐😢😭😰😟😲😴) | 删除, 替换为 V+/V− 数学符号 |
| `web/static/index.html` (旧) | 情感空间标注 "兴奋/焦虑/满足/抑郁" | 替换为 V+/V−/A+/A− 轴标签 |
| `entry/ui_components.py` | `valence_emoji()` 函数映射效价→emoji | 重命名为 `valence_sign()` → 返回 V+/V−/V~ |
| `entry/interactive.py` | 显示 emoji 表情 (v_icon/feel) | 替换为 valence_sign() 数学符号 |

### 睡眠节奏调整

所有 VLPO 时间常数 10× 缩放, 完整睡眠周期从 ~30s → ~5min:

| 参数 | 旧值 | 新值 |
|------|------|------|
| `NREM_REM_CYCLE_STEPS` | 30 | **300** |
| `MIN_NREM_DURATION` | 8 | **80** |
| `MIN_REM_DURATION` | 4 | **40** |
| `MAX_REM_DURATION` | 12 | **120** |
| `REM_ON_GROWTH` | 0.08 | **0.008** |
| `REM_OFF_GROWTH_IN_REM` | 0.12 | **0.012** |
| `FLIP_FLOP_MIN_STABLE` | 15 | **150** |

### 睡眠时对话 — 社会唤醒机制

Agent 睡眠时收到人类消息 → 强制唤醒 (生物合理性: 社会信号是强唤醒刺激):

```
人类输入 → interrupt_with_human_input()
  → 检测 agent.vlpo.is_asleep
  → VLPO 抑制 (vlpo_activation→0, _is_asleep→False)
  → NE 觉醒中枢激活 (arousal_center_activity→0.6)
  → 睡眠状态清零 (state→'awake', phase→'none')
  → body.b[4] 专注度 +0.2 (模拟"被叫醒"的高唤醒)
  → 正常处理对话 (带睡眠惯性)
  → 回应前缀 "[被唤醒]"
```

| 状态 | 行为 |
|------|------|
| awake | 正常对话 |
| NREM N1/N2/N3 | **强制唤醒** + 回应带 `[被唤醒]` 前缀 |
| REM | **强制唤醒** + 回应带 `[被唤醒]` 前缀 |

### 自听闭环修复 (Web 模式)

v5.6 设计了完整的自听闭环，但 `autonomous.py` 的 `interrupt_with_human_input()` 中缺失了 `set_self_audio()` 调用，
导致 Web 聊天的情感传染断链。v6.5 修复使 Web 与控制台模式一致:

```
Agent.speak() → response
  → encode_text(response) → resp_vec
  → self_sentiment = [valence, arousal, 0,0,0,0, v_raw]
  → agent.set_self_audio(resp_vec, self_sentiment)
  → 下一步 agent.step():
      - self_valence_ema 更新 (α=0.15+attn*0.30)
      - self_arousal_ema 更新
      - body.b[0] 社交维度调制 (+0.025 * self_valence)
      - 自我一致性检测 (认知失调: |current_v - self_valence_ema|)
      - 失调→唤醒升高 (dissonance = (1-coherence)*0.2)
```

### SSE API — 新增字段

`_build_status()` 新增:

| 字段 | 数据源 | 说明 |
|------|--------|------|
| `development` (9字段) | `agent.meta.get_developmental_factors()` | stage, stage_name, plasticity, learn_rate_scale, is_infant/child/adolescent/adult |
| `sleep.cycle_position` | `agent._sleep_state.cycle_position` | 当前 NREM/REM 周期位置 [0, 1] |

### 文件变更

| 文件 | 变更 |
|------|------|
| `web/static/index.html` | **REWRITTEN** — 赛博生物CSS + 左对话右3×3布局 + 无现象学修复 + 发育年龄显示 + 新自由能面板 |
| `web/server.py` | +development 字段 + sleep.cycle_position + v6.4→v6.5 版本号 + emoji 清理 |
| `brainstem_cerebellum/pons/vlpo.py` | 所有睡眠时间常数 10× 缩放 (30→300 周期步数) |
| `entry/autonomous.py` | +睡眠唤醒检查 + `[被唤醒]` 通知 + **自听闭环修复** (set_self_audio) |
| `entry/ui_components.py` | valence_emoji→valence_sign + 年龄 emoji→几何符号 |
| `entry/interactive.py` | valence_emoji→valence_sign |
| `PROJECT_STATUS.md` | 更新到 v6.5 新内容 |

### Web 启动

```bash
python web/server.py --port 8080          # 完整模式
python web/server.py --port 8080 --no-auto --no-sensors --dev  # 最小测试模式
```

浏览器访问 `http://localhost:8080`

---

## v6.6 核心变更 — 持久化修复 + 版本统一 + 代码质量提升 + 安全加固

### 设计哲学: 打磨细节，让系统真正"活过多次会话"

v6.5 完成了 Web 前端大改和自听闭环修复，但多个持久化问题导致 Agent 无法在跨会话中积累经验。
v6.6 聚焦于修正这些阻碍长期成长的问题，同时提升代码质量与安全性。

### 四个关键修复

| 问题 | 根因 | 修复 |
|------|------|------|
| **Step 计数器重置** | `meta.step_count` 从未在 `step()`/`light_step()` 中同步 | `meta.step_count = max(meta.step_count, step_count)` |
| **SCN 时间卡住/倒计时** | TTFL 分子钟 ODE 相位漂移 | `SCN.get_reliable_hour()` — 步数线性映射到 24h |
| **视觉每会话重训** | Web 启动不加载存档 | `init_agent()` 默认从 `web_autosave.pkl` 恢复 + shutdown 自动保存 |
| **F_social 永远为 0** | `analyze_sentiment()` 未传入 Hebb 情感词典 | 使用 `get_emotional_lexicon()` + `social_ctx` 跨 API 调用持久化 |

### 原则清理

| 变更 | 原因 |
|------|------|
| **移除 "无LLM" 原则** | 方法3 (No Large Language Models) 从三条方法论中删除 | 项目方向调整 — 不再排除 LLM 作为可能的工具 |

涉及文件: `CLAUDE.md`, `README.md`, `cerebrum/__init__.py`, `environments/text_interface.py`

### 代码质量提升

| 项目 | 文件 | 描述 |
|------|------|------|
| 异常 traceback | `cns/agent.py` | 新增 `_debug_trace()` + 12 处关键异常块加标签 (受 `NOTME_DEBUG` 控制) |
| 版本统一 | `CLAUDE.md`, `requirements.txt`, `web/server.py` | 全部统一到 v6.6 |
| Web 路径遍历修复 | `web/server.py` | `/api/reading/list` — `os.path.realpath()` + 目录白名单 |
| 测试辅助提取 | `tests/conftest.py` (新建) | `make_theta()`, `make_deterministic_s()`, `make_random_s()` |
| 依赖补全 | `requirements.txt` + `requirements-dev.txt` (新建) | +opencv-python, +pytest/ruff |
| .gitignore 改进 | `.gitignore` | +.env, venv/, .pytest_cache/, coverage/ |
| CHANGELOG | `CHANGELOG.md` (新建) | v5.5→v6.6 完整变更历史 |

### 文件变更

| 文件 | 变更 |
|------|------|
| `cns/agent.py` | +`_debug_trace()`, +12 处异常标签, +`meta.step_count` 同步 |
| `web/server.py` | 版本 v6.5→v6.6, +路径遍历安全, +save/load 持久化 |
| `cerebrum/limbic_system/scn.py` | +`get_reliable_hour()` 静态方法 |
| `entry/autonomous.py` | +`_social_ctx` 持久化, +`get_emotional_lexicon()` |
| `CLAUDE.md` | 版本 v5.7→v6.6, +v6.0-v6.6 演进, +里程碑表, +原则更新 |
| `README.md` | 移除 "5. 无LLM" 章节 |
| `cerebrum/__init__.py` | "五条"→"四条核心原则" |
| `environments/text_interface.py` | 移除 "No LLM" 注释块 |
| `tests/conftest.py` | **NEW** — 共享测试辅助函数 |
| `CHANGELOG.md` | **NEW** — 全版本变更历史 |
| `requirements.txt` | 版本 v6.4→v6.6, +opencv-python |
| `requirements-dev.txt` | **NEW** — 开发依赖 |
| `PROJECT_STATUS.md` | 更新到 v6.6 |

### 测试结果 (v7.1-dev Phase B)

| 测试套件 | 通过/总数 | 状态 |
|----------|----------|------|
| test_v6_3_sleep | 9/9 | ✅ |
| test_v6_4_resident | 9/9 | ✅ |
| test_v6_1_development | 7/7 | ✅ |
| test_v6_2_memory | 6/6 | ✅ |
| test_lgn | 5/5 | ✅ |
| test_nn_base | 39/39 | ✅ (Phase A) |
| test_nn_encoders | 48/48 | ✅ (Phase B) |
| test_nn_memory | 37/37 | ✅ (Phase C) |
| test_nn_language | 40/40 | ✅ ★ NEW (Phase D) |
| **总计** | **200/200 (100%)** | ✅ |

### Phase A 交付物

| 文件 | 状态 | 描述 |
|------|------|------|
| `cns/nn/__init__.py` | ★ NEW | 包入口, 全部导出 |
| `cns/nn/config.py` | ★ NEW | NNConfig — device/dtype/training开关/学习率 |
| `cns/nn/bridge.py` | ★ NEW | numpy↔tensor 桥接 + 自动设备检测 |
| `cns/nn/base.py` | ★ NEW | NeuralModule 抽象基类 (forward/train_step/save/load) |
| `cns/nn/interfaces.py` | ★ NEW | TextEncoder(64d)/VisualEncoder(308d)/AudioEncoder(96d) |
| `cns/params.py` | 修改 | +DEFAULT_NN_PARAMS (10个NN超参数) |
| `cns/persistence.py` | 修改 | +save_nn_modules / load_nn_modules / has_nn_checkpoint |
| `cns/data_types.py` | 修改 | 文档更新 (NN配置移至cns/nn/) |
| `requirements.txt` | 修改 | +torch>=2.0.0 |

### v7.0 蓝图

机器学习嵌入改造蓝图已生成: `docs/superpowers/plans/2026-06-08-v7-ml-transformation-blueprint.md`

核心改造方向:
- **感知层**: 手写特征提取器 → 学习型编码器 (Text/Vision/Audio)
- **语言层**: Trigram Hebb链 → 小型神经语言模型
- **记忆层**: Hebb情景记忆保留 + 向量语义记忆
- **跨模态**: Hebb对比学习 → CLIP式神经网络对比学习
- **保留系统**: FEP框架、身体稳态、睡眠节律、脑区架构

---

*由 v7.5-dev Phase F 更新 · 297/297 测试通过 · 下一步: 长期运行验证 + 性能优化*
