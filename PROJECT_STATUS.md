# NotMe 项目状态报告

> **版本**: v4.3 — 规则更新
> **日期**: 2026-06-05
> **基于**: 图1-4 人脑结构调查可视化图表集

---

## 版本历史

| 版本 | 日期 | 关键变更 |
|------|------|---------|
| v4.0 | 2026-06-05 | 人脑层级结构架构重组 |
| v4.1 | 2026-06-05 | 大脑功能细分 + 丘脑独立 + Gestalt恢复 |
| v4.2 | 2026-06-05 | V1/V2/V4独立模块 + shim消除 + 0 flat import |
| **v4.3** | **2026-06-05** | **图3六大运行规则对齐 + FPN/TPN新增 + LTP/LTD** |

---

## 模块完成度

### 总览

| 状态 | 数量 | 占比 |
|------|------|------|
| ★ 已实现 (含 v4.3 新增) | 21 | 36% |
| ○ 占位 (接口设计完成, 待实现代码) | 26 | 45% |
| — 配置/工具/入口 (无类定义但功能完备) | 11 | 19% |
| **合计** | **58** | **100%** |

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
| 听皮层 (BA41/42) | `temporal_lobe/auditory_cortex.py` | ○ 占位 |
| IT皮层 | `temporal_lobe/it_cortex.py` | ○ 占位 |
| 梭状回 (BA37) | `temporal_lobe/fusiform.py` | ○ 占位 |
| **枕叶** | | |
| 视觉通路 (V1+V2+V4) | `occipital_lobe/visual_pathway.py` | ★ 已实现 (Gabor滤波) |
| 视网膜→LGN | `occipital_lobe/retina_lgn.py` | ★ 已实现 |
| Gestalt 知觉分组 | `occipital_lobe/gestalt.py` | ★ 已实现 |
| V1 (BA17) | `occipital_lobe/v1.py` | ★ 已实现 |
| V2 (BA18) | `occipital_lobe/v2.py` | ★ 已实现 |
| V4 (BA19) | `occipital_lobe/v4.py` | ★ 已实现 |
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
| 丘脑 | `thalamus/thalamus.py` | ○ 占位 (接口完整) |
| **联合皮层 + 三大网络** | | |
| DMN | `association/dmn.py` | ★ 已实现 (自我模型) |
| **FPN** | `association/fpn.py` | **★ NEW v4.3** (注意力探照灯) |
| **TPN** | `association/tpn.py` | **★ NEW v4.3** (跷跷板动态) |
| 跨模态联合 | `association/crossmodal.py` | ★ 已实现 (COCO Visual↔Text) |

#### 脑干 + 小脑 (Brainstem + Cerebellum)

| 脑区 | 模块 | 状态 |
|------|------|------|
| **中脑** | | |
| VTA (多巴胺) | `midbrain/vta.py` | ○ 占位 |
| 黑质 (SNc/SNr) | `midbrain/substantia_nigra.py` | ○ 占位 |
| 上丘 | `midbrain/superior_colliculus.py` | ○ 占位 |
| **脑桥** | | |
| 蓝斑核 (NE) | `pons/locus_coeruleus.py` | ○ 占位 |
| 网状结构 | `pons/reticular_formation.py` | ○ 占位 |
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
| **1. 分层处理** | V1→V2→V4→IT 层级 | ✅ 部分 (视觉已实现, 听觉/运动待) | `visual_pathway.py` |
| **2. 双向加工** | 自下而上 + 自上而下 | ✅ 部分 (上行已实现, FPN下行v4.3新增) | `visual_pathway.py`, `fpn.py` |
| **3. 并行分布式** | 分布式表征 + 6条子通路 | ✅ 已实现 | `crossmodal.py`, 感知向量R^330 |
| **4. 注意力瓶颈** | FPN探照灯 + TPN↔DMN跷跷板 | ✅ 接口已定义 (v4.3) | `fpn.py`, `tpn.py`, `dmn.py` |
| **5. 赫布可塑性** | LTP/LTD + 睡眠巩固 | ✅ 已实现 | `hippocampus.py`, `wernicke.py` |
| **6. 预测编码** | 自由能最小化 | ✅ 已实现 | `cingulate.py`, `prefrontal.py` |

### 规则实现度: 5/6 有代码支撑, 1/6 接口就绪 (规则4 v4.3新增)

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

1. **循环导入**: `cns/__init__.py` → `cns/agent.py` → `cerebrum/association/dmn.py` → `cns/data_types.py` 存在循环依赖。不影响直接脚本运行但阻止 `from cerebrum.association import *` 式的导入
2. **FPN/TPN 接口就绪但未集成**: v4.3 定义了完整接口，但未接入 `agent.step()` 主循环
3. **丘脑门控未实现**: 感觉信息目前直通 L0，无门控层
4. **小脑内部模型未实现**: 前向/逆向模型预留但无代码

---

## 下一优先级 (P0 → P3)

### P0 — 核心完善
1. FPN 探照灯集成: `fpn.gate_attention()` → 调制 V1/V2/V4 增益
2. TPN 跷跷板集成: `cingulate` 冲突信号 → `tpn.receive_salience()` → DMN 抑制
3. 修复 cns 循环导入

### P1 — 功能扩展
4. 丘脑感觉门控实现
5. 下丘脑稳态调节 (从 BodyVector 提取 setpoint)
6. VTA RPE (事件驱动学习率)

### P2 — 社会认知
7. 小脑内部模型 (前向/逆向)
8. 蓝斑核 NE 唤醒度调制
9. TPJ 心理理论 (二阶信念)

### P3 — 完整化
10. 梭状回面孔识别
11. 纹状体习惯学习 (D1-D2)
12. 自主神经系统 + 脊髓运动输出

---

## 技术参数 (v4.3 当前)

| 参数 | 值 |
|------|-----|
| 感知维度 D | 330 |
| 隐状态 H | 16 |
| 最大簇数 K | 256 |
| 行动数 A | 5 (grid) / 3 (dialogue) |
| Theta 参数 | 24 |
| 身体维度 M | 5 (grid) / 8 (text) |
| 语料规模 | 50,000 行 |
| 词表规模 | 12,000 词 |
| 模块总数 | 58 |
| 已实现核心模块 | 21 |
| 总代码行数 (估算) | ~15,000 |

---

*由 v4.3 规则更新自动生成 · 基于 人脑结构调查_可视化图表集.html 图1-4*
