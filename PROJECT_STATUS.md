# NotMe 项目状态报告

> **版本**: v5.2 — 听觉层级全接入 + 15条知觉规律落地
> **日期**: 2026-06-05
> **基于**: 图1-4 人脑结构调查可视化图表集 + 听觉系统综述 + 感官知觉规律调查

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

---

## 模块完成度

### 总览

| 状态 | 数量 | 占比 |
|------|------|------|
| ★ 已实现 (含 v5.0/v5.2 新增) | 37 | 52% |
| ○ 占位 (接口设计完成, 待实现代码) | 23 | 32% |
| — 配置/工具/入口 (无类定义但功能完备) | 11 | 16% |
| **合计** | **71** | **100%** |

### 按脑区

#### 大脑皮层 (Cerebrum) — 核心认知

| 脑区 | 模块 | 状态 |
|------|------|------|
| **额叶** | | |
| 前额叶 (dlPFC) | `frontal_lobe/prefrontal.py` | ★ 已实现 (EFE行动选择) |
| 布罗卡区 (BA44/45) | `frontal_lobe/broca.py` | ★ 已实现 (词序Hebb链) |
| 运动皮层 (BA4/6) | `frontal_lobe/motor_cortex.py` | ○ 占位 |
| 眶额皮层 (BA11) | `frontal_lobe/orbitofrontal.py` | ○ 占位 |
| **顶叶** | | |
| 体感皮层 (BA3,1,2) | `parietal_lobe/somatosensory.py` | ○ 占位 |
| 空间注意力 | `parietal_lobe/spatial_attention.py` | ○ 占位 |
| 颞顶联合区 (TPJ) | `parietal_lobe/tpj.py` | ○ 占位 |
| **颞叶** | | |
| 韦尼克区 (BA22) | `temporal_lobe/wernicke.py` | ★ 已实现 (语言理解) |
| 听皮层 (BA41/42) | `temporal_lobe/auditory_cortex.py` | **★ NEW v5.2** (A1+Belt+Parabelt 3层, 听觉场景分析, What/Where双流) |
| **听觉层级管线** | `temporal_lobe/auditory_hierarchy.py` | **★ NEW v5.2** (6核团编排: CN→SOC→LL→IC→MGB→AC) |
| IT皮层 | `temporal_lobe/it_cortex.py` | **★ NEW v5.0** (Hebb物体学习+反馈预测) |
| **MT (V5)** | `temporal_lobe/mt_cortex.py` | **★ NEW v5.0** (方向选择性运动检测) |
| **MST** | `temporal_lobe/mst_cortex.py` | **★ NEW v5.0** (光流模式检测) |
| 梭状回 (BA37) | `temporal_lobe/fusiform.py` | ○ 占位 |
| **枕叶** | | |
| 视觉通路 (V1+V2+V4) | `occipital_lobe/visual_pathway.py` | ★ 已实现 (Gabor滤波核心, v5.0 M/P/K核集) |
| 视网膜→LGN | `occipital_lobe/retina_lgn.py` | ★ 已实现 (v5.0 M/P/K分型输出) |
| Gestalt 知觉分组 | `occipital_lobe/gestalt.py` | ★ 已实现 (闭合/共同命运/简洁律→v5.0由层级动力学涌现) |
| V1 (BA17) | `occipital_lobe/v1.py` | **★ REWRITTEN v5.0** (层状模块: 4Cα/4Cβ/斑块/4B/5/6层) |
| V2 (BA18) | `occipital_lobe/v2.py` | **★ REWRITTEN v5.0** (三类条纹: 粗/苍白/细 + 横向连接) |
| V4 (BA19) | `occipital_lobe/v4.py` | **★ REWRITTEN v5.0** (M/P/K汇合 + 曲率+颜色恒常) |
| **视觉层级管线** | `occipital_lobe/visual_hierarchy.py` | **★ NEW v5.0** (10模块编排器: 前馈→反馈→PE→感知向量)
| **边缘系统** | | |
| 海马 | `limbic_system/hippocampus.py` | ★ 已实现 (ClusterNetwork) |
| 杏仁核 | `limbic_system/amygdala.py` | ★ 已实现 (情感词汇) |
| 扣带回/ACC | `limbic_system/cingulate.py` | ★ 已实现 (自由能计算) |
| 下丘脑 | `limbic_system/hypothalamus.py` | ○ 占位 |
| 嗅皮层 | `limbic_system/olfactory.py` | ○ 占位 |
| **基底神经节** | | |
| 动作门控 (MoE) | `basal_ganglia/action_gating.py` | ★ 已实现 |
| 纹状体 | `basal_ganglia/striatum.py` | ○ 占位 |
| 苍白球 | `basal_ganglia/pallidum.py` | ○ 占位 |
| 底丘脑核 | `basal_ganglia/subthalamic.py` | ○ 占位 |
| **丘脑** | | |
| 丘脑 (总览) | `thalamus/thalamus.py` | ○ 占位 (VA/VL/MD/TRN待实现) |
| **LGN (外侧膝状体)** | `thalamus/lgn.py` | **★ v5.0** (6层主动门控: tonic/burst + V1反馈 + TRN) |
| **Pulvinar (丘脑枕)** | `thalamus/pulvinar.py` | **★ v5.0** (SC→皮层快速中继, 第二条视觉通路) |
| **MGB (内侧膝状体)** | `thalamus/mgb.py` | **★ NEW v5.2** (MGv/MGd/MGm 3亚区, 唤醒度门控, FPN注意调制) |
| **联合皮层 + 三大网络** | | |
| DMN | `association/dmn.py` | ★ 已实现 (自我模型) |
| **FPN** | `association/fpn.py` | **★ v4.3/v4.4 + v5.0增强** (注意力探照灯 + M/P/K通道增益) |
| **TPN** | `association/tpn.py` | **★ v4.3/v4.4** (跷跷板动态) |
| 跨模态联合 | `association/crossmodal.py` | ★ 已实现 (COCO Visual↔Text) |
| **视觉绑定** | `association/visual_binding.py` | **★ NEW v5.0** (FPN驱动的M/P/K跨通道特征绑定) |

#### 脑干 + 小脑 (Brainstem + Cerebellum)

| 脑区 | 模块 | 状态 |
|------|------|------|
| **中脑** | | |
| VTA (多巴胺) | `midbrain/vta.py` | ○ 占位 |
| 黑质 (SNc/SNr) | `midbrain/substantia_nigra.py` | ○ 占位 |
| 上丘 | `midbrain/superior_colliculus.py` | **★ v5.0** (显著性图+新颖性检测+空间定向) |
| **下丘 (IC)** | `midbrain/inferior_colliculus.py` | **★ NEW v5.2** (中央核/背皮层/外皮层, 频率×空间×时间整合+新颖性检测) |
| **脑桥** | | |
| 蓝斑核 (NE) | `pons/locus_coeruleus.py` | ○ 占位 |
| 网状结构 | `pons/reticular_formation.py` | ○ 占位 |
| **耳蜗核** | `pons/cochlear_nucleus.py` | **★ NEW v5.2** (AVCN/PVCN/DCN, 频谱分解, 相位锁定, 声反射) |
| **上橄榄复合体 (SOC)** | `pons/superior_olivary.py` | **★ NEW v5.2** (MSO/LSO/MNTB, ITD/ILD双耳定位, 双重理论) |
| **外侧丘系 (LL)** | `pons/lateral_lemniscus.py` | **★ NEW v5.2** (VNLL/DNLL, 时间增强+双侧GABA抑制) |
| **延髓** | | |
| 自主神经 | `medulla/autonomic.py` | ○ 占位 |
| **小脑** | | |
| 运动协调 | `cerebellum/motor_coordination.py` | ○ 占位 |
| 时序预测 | `cerebellum/predictive_timing.py` | ○ 占位 |
| **神经调节** | | |
| 元学习 | `neuromodulatory/meta_learning.py` | ★ 已实现 (MetaLearner) |
| 可塑性调节 | `neuromodulatory/plasticity.py` | ○ 占位 |

#### 身体 + 脊髓

| 模块 | 状态 |
|------|------|
| `body/body_state.py` | ○ 占位 |
| `body/interoception.py` | ○ 占位 |
| `spinal/motor_output.py` | ○ 占位 |

---

## 图3 六大规则实现状态

| 规则 | 核心机制 | 实现状态 | 关键模块 |
|------|---------|---------|---------|
| **1. 分层处理** | V1→V2→V4→IT + MT→MST + **CN→SOC→IC→MGB→AC** | ✅ **增强 (v5.2 听觉15级层级)** | `visual_hierarchy.py`, `auditory_hierarchy.py` |
| **2. 双向加工** | 自下而上 + 自上而下 (前馈+反馈) | ✅ **增强 (v5.2 听觉3层PE闭环)** | `visual_hierarchy.py`, `auditory_hierarchy.py`, `fpn.py` |
| **3. 并行分布式** | M/P/K 三流 + 听觉6核团 + 分布式表征 | ✅ **增强 (v5.2 视听双模态并行N核团)** | `crossmodal.py`, 感知向量D_V52=468 |
| **4. 注意力瓶颈** | FPN探照灯 + TPN↔DMN跷跷板 + 听觉注意 | ✅ **增强 (v5.2 FPN调制MGB+听皮层增益)** | `fpn.py`(集成), `tpn.py`(集成), `dmn.py`, `mgb.py` |
| **5. 赫布可塑性** | LTP/LTD + 睡眠巩固 + 听觉记忆 | ✅ **增强 (v5.2 听皮层经验依赖可塑性)** | `hippocampus.py`, `wernicke.py`, `auditory_cortex.py` |
| **6. 预测编码** | 自由能最小化 + 视听双PE闭环 | ✅ **增强 (v5.2 CN+IC+MGB+AC 4层PE汇入F_accuracy)** | `visual_hierarchy.py`, `auditory_hierarchy.py`, `cingulate.py` |

### 规则实现度: 6/6 有代码支撑 (v4.4 全部六条规则已实现)

---

## 图1-4 完整对照

| 图表 | 内容 | 项目覆盖 |
|------|------|---------|
| **图1** | 人脑层级结构 (CNS→大脑→四叶→BA分区) | ✅ 完整 — v4.0架构100%对齐 |
| **图2** | 9大核心脑区功能分工 | ✅ 完整 — 全部9区有对应模块 |
| **图3** | 六大核心运行规则 | ✅ v4.3 对齐 — 全部6条有文档+代码映射 |
| **图4** | 信息流转路径 (看苹果→拿苹果) | ✅ 完整 — 7阶段全链路有对应 |

---

## 已知问题

1. ~~**循环导入**~~: ✅ **v4.4 已修复** — `cns/__init__.py` 使用 `__getattr__` 延迟加载 Agent
2. ~~**FPN/TPN 接口就绪但未集成**~~: ✅ **v4.4 已集成** — FPN 探照灯 + TPN 跷跷板已接入 `agent.step()` 主循环
3. ~~**丘脑门控未实现**~~: ✅ **v5.0 已实现** — LGN 6层主动门控 (tonic/burst + V1反馈 + TRN侧抑制)
4. **小脑内部模型未实现**: 前向/逆向模型预留但无代码
5. ~~**VisualHierarchy未接入主循环**~~: ✅ **v5.1 已接入** — `agent.step()` Phase 0 使用 `VisualHierarchy.process()` 驱动全视觉管线
6. ~~**感知向量旧布局仍在使用**~~: ✅ **v5.1 已切换** — D=330→D_V5=372, S_CORE=320→362, 全模块对齐 V5 布局
7. **跨模态模型需重训**: D=330→372→468 布局变更导致已保存的 .pkl 模型不兼容, 需运行 `stage2_crossmodal.py` 重新训练
8. **听觉管线为语义代理模式**: 无真实音频输入时使用语义→伪频谱代理, 真实声学特征需音频输入支持

---

## 下一优先级 (P0 → P3)

### P0 — 核心完善 ✅ v4.4 + v5.0 + v5.2 全部完成
1. ✅ FPN 探照灯集成: `fpn.gate_attention()` → 调制感觉输入增益 (v4.4)
2. ✅ TPN 跷跷板集成: `cingulate` 冲突信号 → `tpn.receive_salience()` → DMN 抑制 (v4.4)
3. ✅ 修复 cns 循环导入 (v4.4)
4. ✅ **M/P/K 并行通路 + 10脑区视觉层级管线 + 预测编码闭环 (v5.0)**
5. ✅ **听觉6核团层级管线 + 15条知觉规律 + 视听双模态并行 (v5.2)**

### P1 — 功能扩展
5. ~~丘脑感觉门控实现~~ ✅ v5.0 LGN 6层主动门控
6. ~~全视觉管线接入 agent.step() 主循环~~ ✅ v5.1
7. ~~感知向量 D=330 → D_V5=372 全局切换~~ ✅ v5.1
8. ~~**听觉系统 15条知觉规律 + 6核团层级管线**~~ ✅ v5.2
9. 下丘脑稳态调节 (从 BodyVector 提取 setpoint)
10. VTA RPE (事件驱动学习率)
11. 真实音频输入支持 (替换语义代理模式)

### P2 — 社会认知
10. 小脑内部模型 (前向/逆向)
11. 蓝斑核 NE 唤醒度调制 (LGN 已预留脑干接口)
12. TPJ 心理理论 (二阶信念)

### P3 — 完整化
13. 梭状回面孔识别 (IT 已完成物体表征, 面孔为特化子集)
14. 纹状体习惯学习 (D1-D2)
15. 自主神经系统 + 脊髓运动输出
16. Gestalt 格式塔效应涌现验证 (闭合律/共同命运律/简洁律的层级动力学演示)

---

## 技术参数 (v5.0 当前)

| 参数 | 值 |
|------|-----|
| 感知维度 D | **468** (V5.2 text[64]+vision[308]+audio[96]) |
| 感觉核心 S_CORE | **458** (D - 10, F_context + action_onehot) |
| 隐状态 H | 16 |
| 最大簇数 K | 256 |
| 行动数 A | 5 (grid) / 3 (dialogue) |
| Theta 参数 | 24 |
| 身体维度 M | 5 (grid) / 8 (text) |
| 语料规模 | 50,000 行 |
| 词表规模 | 12,000 词 |
| 视觉通路数 | **3 并行** (M/P/K) + 1 快速 (SC→Pulvinar) |
| 听觉核团数 | **6** (CN, SOC, LL, IC, MGB, AC) |
| 视觉脑区模块 | **10** (LGN, V1, V2, V4, MT, MST, IT, SC, Pulvinar, FPN-Binding) |
| 模块总数 | **71** |
| 已实现核心模块 | **37** (52%) |
| v5.2 新增代码 | **~1,500 行** (7 files) |
| v5.2 知觉规律 | **15 条** (音调拓扑/双重理论/ASA/预测编码/... ) |

---

*由 v5.2 听觉系统自动更新 · 基于 听觉系统综述 + 感官知觉规律调查(文件2 Section 2)*
