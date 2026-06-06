# NotMe 项目状态报告

> **版本**: v5.7 — 发育年龄系统 + 会话持久化 + 多模态同步输入 + 实时传感器流 + Rich 终端 UI
> **日期**: 2026-06-06
> **基于**: 发育心理学 (Piaget, Vygotsky 最近发展区) + 语言习得理论 (婴儿模仿学习) + 多模态感觉整合 (Stein & Meredith)

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
| **v5.4** | **2026-06-05** | **痛觉系统 — 7条知觉规律, 闸门控制, 双通路, 下行调控PAG→RVM闭环, D=516, 8个新增/升级脑区** |
| **v5.5** | **2026-06-05** | **神经调节系统 — 下丘脑稳态调节(SetpointModel+DriveSystem+HPA轴) + VTA RPE(事件驱动学习率) + 蓝斑核NE(phasic/tonic+SNR+RVM连接)** |
| **v5.6** | **2026-06-06** | **语言系统 — 弓状束(Broca↔Wernicke Hebb桥接) + 语音回路(Baddeley工作记忆) + 短语结构(BA44层级句法) + 角回(阅读通路) + 运动皮层(发音规划) + TPJ(心理理论/语用) + N400/P600(语言预测误差)** |
| **v5.7** | **2026-06-06** | **发育年龄系统 + 会话持久化 + 多模态同步输入总线 + 摄像头/麦克风实时流 + Rich终端UI + 纯净模式(零预训练)** |

---

## 模块完成度

### 总览

| 状态 | 数量 | 占比 |
|------|------|------|
| ★ 已实现 (含 v5.0/v5.2/v5.3/v5.4/v5.5/v5.6/v5.7 新增) | **54** | 65% |
| ○ 占位 (接口设计完成, 待实现代码) | 16 | 19% |
| — 配置/工具/入口 (无类定义但功能完备) | 13 | 16% |
| **合计** | **83** | **100%** |

### 按脑区

#### 大脑皮层 (Cerebrum) — 核心认知

| 脑区 | 模块 | 状态 |
|------|------|------|
| **额叶** | | |
| 前额叶 (dlPFC) | `frontal_lobe/prefrontal.py` | ★ 已实现 (EFE行动选择) |
| 布罗卡区 (BA44/45) | `frontal_lobe/broca.py` | ★ 已实现 (词序Hebb链) |
| **短语结构网络** | `frontal_lobe/phrase_structure.py` | **★ NEW v5.6** (BA44层级句法: 转移概率边界检测+短语聚类+递归嵌入) |
| **语音回路** | `frontal_lobe/phonological_loop.py` | **★ NEW v5.6** (Baddeley模型: ~7组块语音存储+默读复述+~2s消退) |
| **运动皮层 (BA4/6)** | `frontal_lobe/motor_cortex.py` | **★ UPGRADED v5.6** (16维发音特征+SMA序列编排+共发音+运动指令副本) |
| 眶额皮层 (BA11) | `frontal_lobe/orbitofrontal.py` | ○ 占位 |
| **顶叶** | | |
| 体感皮层 (BA3,1,2) | `parietal_lobe/somatosensory.py` | ★ UPGRADED v5.4 (S1/S2全实现: 痛觉定位+触觉编码+本体感觉+威胁评估) |
| 空间注意力 | `parietal_lobe/spatial_attention.py` | ○ 占位 |
| **颞顶联合区 (TPJ)** | `parietal_lobe/tpj.py` | **★ UPGRADED v5.6** (心理理论+意图推断+反讽检测+视角采择+语用丰富化) |
| **角回 (BA39)** | `parietal_lobe/angular_gyrus.py` | **★ NEW v5.6** (阅读通路: 视觉字形→语音表征 Hebb映射, 双路径阅读模型) |
| **颞叶** | | |
| 韦尼克区 (BA22) | `temporal_lobe/wernicke.py` | **★ UPGRADED v5.6** (+compute_language_PE: N400语义PE+P600句法PE+MMN语音PE, F_language汇入自由能) |
| 听皮层 (BA41/42) | `temporal_lobe/auditory_cortex.py` | ★ NEW v5.2 (A1+Belt+Parabelt 3层, 听觉场景分析, What/Where双流) |
| 听觉层级管线 | `temporal_lobe/auditory_hierarchy.py` | ★ NEW v5.2 (6核团编排: CN→SOC→LL→IC→MGB→AC) |
| IT皮层 | `temporal_lobe/it_cortex.py` | ★ NEW v5.0 (Hebb物体学习+反馈预测) |
| MT (V5) | `temporal_lobe/mt_cortex.py` | ★ NEW v5.0 (方向选择性运动检测) |
| MST | `temporal_lobe/mst_cortex.py` | ★ NEW v5.0 (光流模式检测) |
| 梭状回 (BA37) | `temporal_lobe/fusiform.py` | ○ 占位 |
| **枕叶** | | |
| 视觉通路 (V1+V2+V4) | `occipital_lobe/visual_pathway.py` | ★ 已实现 (Gabor滤波核心, v5.0 M/P/K核集) |
| 视网膜→LGN | `occipital_lobe/retina_lgn.py` | ★ 已实现 (v5.0 M/P/K分型输出) |
| Gestalt 知觉分组 | `occipital_lobe/gestalt.py` | ★ 已实现 (闭合/共同命运/简洁律→v5.0由层级动力学涌现) |
| V1 (BA17) | `occipital_lobe/v1.py` | ★ REWRITTEN v5.0 (层状模块: 4Cα/4Cβ/斑块/4B/5/6层) |
| V2 (BA18) | `occipital_lobe/v2.py` | ★ REWRITTEN v5.0 (三类条纹: 粗/苍白/细 + 横向连接) |
| V4 (BA19) | `occipital_lobe/v4.py` | ★ REWRITTEN v5.0 (M/P/K汇合 + 曲率+颜色恒常) |
| 视觉层级管线 | `occipital_lobe/visual_hierarchy.py` | ★ NEW v5.0 (10模块编排器: 前馈→反馈→PE→感知向量)
| **边缘系统** | | |
| 海马 | `limbic_system/hippocampus.py` | ★ 已实现 (ClusterNetwork) |
| 杏仁核 | `limbic_system/amygdala.py` | ★ 已实现 (情感词汇) |
| 扣带回/ACC | `limbic_system/cingulate.py` | ★ 已实现 (自由能计算) |
| 岛叶 (BA13-16) | `limbic_system/insula.py` | ★ NEW v5.4 (后岛叶内感受+前岛叶情感评估+突显网络) |
| 下丘脑 | `limbic_system/hypothalamus.py` | ★ NEW v5.5 (SetpointModel+DriveSystem+HPA轴+自主神经平衡) |
| 嗅皮层 | `limbic_system/olfactory.py` | ○ 占位 |
| **基底神经节** | | |
| 动作门控 (MoE) | `basal_ganglia/action_gating.py` | ★ 已实现 |
| 纹状体 | `basal_ganglia/striatum.py` | ○ 占位 |
| 苍白球 | `basal_ganglia/pallidum.py` | ○ 占位 |
| 底丘脑核 | `basal_ganglia/subthalamic.py` | ○ 占位 |
| **丘脑** | | |
| 丘脑 (总览) | `thalamus/thalamus.py` | ★ UPGRADED v5.4 (VPL/CM-Pf/MD/Po痛觉中继核团全实现) |
| LGN (外侧膝状体) | `thalamus/lgn.py` | ★ v5.0 (6层主动门控: tonic/burst + V1反馈 + TRN) |
| Pulvinar (丘脑枕) | `thalamus/pulvinar.py` | ★ v5.0 (SC→皮层快速中继, 第二条视觉通路) |
| MGB (内侧膝状体) | `thalamus/mgb.py` | ★ NEW v5.2 (MGv/MGd/MGm 3亚区, 唤醒度门控, FPN注意调制) |
| **联合皮层 + 三大网络** | | |
| DMN | `association/dmn.py` | ★ 已实现 (自我模型) |
| FPN | `association/fpn.py` | ★ v4.3/v4.4 + v5.0增强 (注意力探照灯 + M/P/K通道增益) |
| TPN | `association/tpn.py` | ★ v4.3/v4.4 (跷跷板动态) |
| 跨模态联合 | `association/crossmodal.py` | ★ 已实现 (COCO Visual↔Text) |
| 视觉绑定 | `association/visual_binding.py` | ★ NEW v5.0 (FPN驱动的M/P/K跨通道特征绑定) |
| **弓状束 (AF)** | `association/arcuate_fasciculus.py` | **★ NEW v5.6** (Wernicke↔Broca Hebb桥接; 腹侧(理解→言语)+背侧(运动副本→预期听觉); 复述通路) |

#### 脑干 + 小脑 (Brainstem + Cerebellum)

| 脑区 | 模块 | 状态 |
|------|------|------|
| **中脑** | | |
| VTA (多巴胺) | `midbrain/vta.py` | ★ NEW v5.5 (RPEModel+DopamineDynamics+事件驱动学习率调制) |
| 黑质 (SNc/SNr) | `midbrain/substantia_nigra.py` | ○ 占位 |
| 上丘 | `midbrain/superior_colliculus.py` | ★ v5.0 (显著性图+新颖性检测+空间定向) |
| 下丘 (IC) | `midbrain/inferior_colliculus.py` | ★ NEW v5.2 (中央核/背皮层/外皮层, 频率×空间×时间整合+新颖性检测) |
| PAG (导水管周围灰质) | `midbrain/pag.py` | ★ NEW v5.4 (下行痛觉调节总枢纽, 4功能柱, 内啡肽释放+SIΑ+安慰剂镇痛) |
| **脑桥** | | |
| 蓝斑核 (NE) | `pons/locus_coeruleus.py` | ★ NEW v5.5 (NEDynamics phasic/tonic+SNREnhancer+RVM NE连接) |
| 网状结构 | `pons/reticular_formation.py` | ○ 占位 |
| 耳蜗核 | `pons/cochlear_nucleus.py` | ★ NEW v5.2 (AVCN/PVCN/DCN, 频谱分解, 相位锁定, 声反射) |
| 上橄榄复合体 (SOC) | `pons/superior_olivary.py` | ★ NEW v5.2 (MSO/LSO/MNTB, ITD/ILD双耳定位, 双重理论) |
| 外侧丘系 (LL) | `pons/lateral_lemniscus.py` | ★ NEW v5.2 (VNLL/DNLL, 时间增强+双侧GABA抑制) |
| 痛觉层级管线 | `nociception_hierarchy.py` | ★ NEW v5.4 (6核团编排: 背角→脊髓丘脑→丘脑→S1/岛叶←PAG→RVM下行闭环) |
| **延髓** | | |
| 自主神经 | `medulla/autonomic.py` | ○ 占位 |
| RVM (延髓头端腹内侧区) | `medulla/rvm.py` | ★ NEW v5.4 (PAG→脊髓关键中继, OFF/ON细胞动态, 5-HT/NE调制) |
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
| 脊髓背角 | `spinal/dorsal_horn.py` | ★ NEW v5.4 (闸门控制, SG细胞+WDR神经元, 中枢敏化, 快/慢痛分离) |
| `spinal/motor_output.py` | ○ 占位 |

---

## 图3 六大规则实现状态

| 规则 | 核心机制 | 实现状态 | 关键模块 |
|------|---------|---------|---------|
| **1. 分层处理** | V1→V2→V4→IT + MT→MST + CN→SOC→IC→MGB→AC + **Wernicke→AF→Broca→Motor(语言层级)** | ✅ **增强 (v5.6 语言7级层级)** | `visual_hierarchy.py`, `auditory_hierarchy.py`, `arcuate_fasciculus.py` |
| **2. 双向加工** | 自下而上 + 自上而下 (前馈+反馈) + **语言双流(腹侧What+背侧How)** | ✅ **增强 (v5.6 AF双通路+语言PE闭环)** | `visual_hierarchy.py`, `auditory_hierarchy.py`, `fpn.py`, `arcuate_fasciculus.py` |
| **3. 并行分布式** | M/P/K 三流 + 听觉6核团 + **语言6模块并行** + 分布式表征 | ✅ **增强 (v5.6 视听语言全并行)** | `crossmodal.py`, 感知向量D=516 |
| **4. 注意力瓶颈** | FPN探照灯 + TPN↔DMN跷跷板 + **语音回路~7组块限制** | ✅ **增强 (v5.6 语音WM容量瓶颈)** | `fpn.py`, `tpn.py`, `dmn.py`, `phonological_loop.py` |
| **5. 赫布可塑性** | LTP/LTD + 睡眠巩固 + **AF Hebb在线学习+短语结构统计学习** | ✅ **增强 (v5.6 语言Hebb网络6个)** | `hippocampus.py`, `wernicke.py`, `arcuate_fasciculus.py`, `phrase_structure.py` |
| **6. 预测编码** | 自由能最小化 + **N400语义PE+P600句法PE+AF自我监控PE** | ✅ **增强 (v5.6 语言域F_language汇入总F)** | `visual_hierarchy.py`, `auditory_hierarchy.py`, `nociception_hierarchy.py`, `cingulate.py`, `wernicke.py` |

### 规则实现度: 6/6 有代码支撑 (v5.6 全部六条规则进一步增强 — 语言域全覆盖)

---

## v5.6 语言双流模型 (Hickok & Poeppel 2007)

```
                    听觉输入 (speech/text)
                           │
              ┌────────────┴────────────┐
              │   听觉层级 (CN→SOC→     │
              │   LL→IC→MGB→AC)        │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │   语音回路              │  ★ v5.6: Baddeley WM
              │   存储 ⇄ 默读复述       │
              └────────────┬────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
  ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐
  │ 角回 (阅读) │  │  Wernicke   │  │    TPJ      │
  │ BA39        │  │  BA22       │  │  语用理解    │
  │ 字形→语音   │  │ +N400/P600  │  │  意图推断    │
  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
         │                 │                 │
         └─────────────────┼─────────────────┘
                           │ 理解向量
              ┌────────────▼────────────┐
              │   弓状束 (AF)           │  ★ v5.6: Hebb桥接
              │   腹侧: 理解→言语种子   │
              │   背侧: 运动副本→预期   │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │   Broca区 (BA44/45)     │
              │   + 短语结构网络        │  ★ v5.6: 层级句法
              │   词序Hebb链逐词生成    │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │   运动皮层 (BA4/6)      │  ★ v5.6: 发音规划
              │   SMA序列+共发音+       │
              │   运动指令副本          │
              └────────────┬────────────┘
                           │
                    TTS 输出 + 自听 → 语音回路 (自我监控)
```

---

## 图1-4 完整对照

| 图表 | 内容 | 项目覆盖 |
|------|------|---------|
| **图1** | 人脑层级结构 (CNS→大脑→四叶→BA分区) | ✅ 完整 — v4.0架构100%对齐 |
| **图2** | 9大核心脑区功能分工 | ✅ 完整 — 全部9区有对应模块 |
| **图3** | 六大核心运行规则 | ✅ v5.6 增强 — 全部6条有语言域实现 |
| **图4** | 信息流转路径 (看苹果→拿苹果) | ✅ 完整 — 7阶段全链路有对应 |

---

## 已知问题

1. ~~**循环导入**~~: ✅ **v4.4 已修复**
2. ~~**FPN/TPN 接口就绪但未集成**~~: ✅ **v4.4 已集成**
3. ~~**丘脑门控未实现**~~: ✅ **v5.0 已实现**
4. **小脑内部模型未实现**: 前向/逆向模型预留但无代码
5. ~~**VisualHierarchy未接入主循环**~~: ✅ **v5.1 已接入**
6. ~~**感知向量旧布局仍在使用**~~: ✅ **v5.1 已切换**
7. **跨模态模型需重训**: D=330→372→468→516 布局变更导致已保存的 .pkl 模型不兼容, 需运行 `stage2_crossmodal.py` 重新训练
8. ~~**听觉管线为语义代理模式**~~: ✅ **v5.3 已解决**
9. **小脑内部模型未实现**: 前向/逆向模型预留但无代码
10. ~~**VTA RPE未实现**~~: ✅ **v5.5 已实现**
11. ~~**TPJ 心理理论未实现**~~: ✅ **v5.6 已实现** — 意图推断+反讽检测+视角采择+语用丰富化
12. ~~**运动皮层发音规划未实现**~~: ✅ **v5.6 已实现** — 16维发音特征+SMA序列编排+共发音+efference copy
13. ~~**Broca↔Wernicke 之间无直接连接**~~: ✅ **v5.6 已实现** — 弓状束 Hebb 桥接
14. ~~**无语音工作记忆**~~: ✅ **v5.6 已实现** — 语音回路 (Baddeley 模型)
15. ~~**无层级句法结构**~~: ✅ **v5.6 已实现** — 短语结构网络 (BA44 层级句法)
16. ~~**无语言预测误差信号**~~: ✅ **v5.6 已实现** — N400/P600/MMN (compute_language_PE)
17. **v5.6 模块预训练耗时**: 弓状束+短语结构+角回+TPJ 四模块从语料预训练 (~3000-5000句) 在对话启动时需要 ~30-60秒, 后续可缓存加速

---

## 下一优先级 (P0 → P3)

### P0 — 核心完善 ✅ v4.4 + v5.0 + v5.2 + v5.6 全部完成
1. ✅ FPN 探照灯集成 (v4.4)
2. ✅ TPN 跷跷板集成 (v4.4)
3. ✅ 修复 cns 循环导入 (v4.4)
4. ✅ M/P/K 并行通路 + 10脑区视觉层级管线 (v5.0)
5. ✅ 听觉6核团层级管线 + 15条知觉规律 (v5.2)
6. ✅ **语言系统: 弓状束+语音回路+短语结构+角回+运动皮层+TPJ+N400/P600 (v5.6)**

### P1 — 功能扩展 ✅ v5.5 全部完成 (v5.6 无新增P1)
7. ✅ 丘脑感觉门控 (v5.0)
8. ✅ 全视觉管线接入 (v5.1)
9. ✅ 真实音频输入 (v5.3)
10. ✅ 痛觉系统 (v5.4)
11. ✅ 神经调节系统 (v5.5)

### P2 — 社会认知 (部分完成)
12. ~~TPJ 心理理论~~ ✅ v5.6 已实现 (意图推断+语用)
13. 小脑内部模型 (前向/逆向) — 仍未实现

### P3 — 完整化
14. 梭状回面孔识别 (IT 已完成物体表征, 面孔为特化子集)
15. 纹状体习惯学习 (D1-D2)
16. 自主神经系统 + 脊髓运动输出
17. Gestalt 格式塔效应涌现验证 (闭合律/共同命运律/简洁律的层级动力学演示)
18. 黑质 (SNc/SNr) 运动调节 — 可配合运动皮层发音规划实现基底节→丘脑→皮层运动环路

---

## 技术参数 (v5.7 当前)

| 参数 | 值 |
|------|-----|
| 感知维度 D | **516** (V5.4 text[64]+vision[308]+audio[96]+pain[48]) |
| 感觉核心 S_CORE | **506** (D - 10, F_context + action_onehot) |
| 隐状态 H | 16 |
| 最大簇数 K | 256 |
| 行动数 A | 5 (grid) / 3 (dialogue) |
| Theta 参数 | **26** (v5.6: +w_semantic +w_syntactic) |
| 身体维度 M | 5 (grid) / **9 (text, v5.4 +1 痛觉)** |
| 语料规模 | 50,000 行 (纯净模式不使用语料预训练) |
| 词表规模 | 12,000 词 (TTS输出用, 认知网络从零生长) |
| 视觉通路数 | **3 并行** (M/P/K) + 1 快速 (SC→Pulvinar) |
| 听觉核团数 | **6** (CN, SOC, LL, IC, MGB, AC) |
| 痛觉核团数 | **6** (DorsalHorn, Spinothalamic, Thalamus-VPL/CM-Pf/MD/Po, Insula, PAG, RVM) |
| 神经调节模块 | **3** (Hypothalamus, VTA, LocusCoeruleus) |
| 语言模块 (v5.6) | **7** (AF, PhonologicalLoop, PhraseStructure, AngularGyrus, MotorCortex, TPJ, +LanguagePE via Wernicke) |
| 视觉脑区模块 | **10** (LGN, V1, V2, V4, MT, MST, IT, SC, Pulvinar, FPN-Binding) |
| **v5.7 基础设施** | **5** (Persistence, InputBus, SensorIO, Interactive, UIComponents) |
| 模块总数 | **83** |
| 已实现核心模块 | **54 (65%)** — v5.7: +persistence +input_bus +sensor_io +interactive +ui_components |
| v5.6 新增代码 | ~3,200 行 (11 files: 6 new + 4 modified + 1 doc) |
| **v5.7 新增代码** | **~5,700 行 (6 new + 10 modified)** |
| **v5.7 核心变更** | 发育年龄+婴儿模仿+会话持久化+多模态同步输入+摄像头/麦克风实时流+Rich终端UI+纯净模式 |

---

## v5.6 核心机制详解

### 弓状束 (Arcuate Fasciculus)
- **腹侧通路**: Wernicke理解向量 → Hebb召回 → 言语种子词 (复述通路)
- **背侧通路**: 言语计划 → 运动副本 → 预期听觉 (自我监控通路)
- **传导效率**: `conduction` 参数 [0,1] 模拟传导性失语症
- **Hub学习**: 每次对话回合在线强化 AF 连接
- **预训练**: 从语料 3000 句模拟"听→学说话"的统计学习

### 语音回路 (Phonological Loop)
- **语音存储**: 缘上回(BA40)等价, ~7±2 chunk 容量
- **消退率**: 0.08/step → 约2秒 @10fps 无复述消退
- **默读复述**: subvocal rehearsal → 激活度回升
- **语音干扰**: 存储中相似向量加速消退 (phonological similarity effect)
- **词长效应**: 长词占用更多复述时间 → 有效容量↓

### 短语结构网络 (Phrase Structure)
- **边界检测**: bigram转移概率骤降 → 短语边界 (零手写规则)
- **阈值**: 自适应 — 转移概率下四分位数
- **短语连贯性**: 候选词在短语内的连贯度 → 调制 trigram 得分
- **短语闭合**: 长度因素 + 尾词特征 → 结束概率
- **Broca失语模拟**: `phrase_strength` 降低 → 电报式言语

### 语言预测误差 (N400/P600)
- **N400 (语义PE)**: 输入 vs 上下文语义不匹配 → 0.6×(1-congruence) + 0.4×(1-mem_match)
- **P600 (句法PE)**: 连贯性缺口 (上下文coherence vs 实际匹配差异)
- **MMN (语音PE)**: 输入与记忆预期的向量差异
- **F_language**: w_semantic×N400 + w_syntactic×P600 + w_phonological×MMN → 汇入总自由能

### 角回阅读通路
- **字形编码**: Unicode码点+笔画代理+结构特征 → 64维视觉特征
- **字形→语音**: Hebb学习, 12K词表预训练
- **双路径**: 快速路径(MiniLM→语义, 效率) + 脑路径(角回→语音, 生物合理)

### 运动皮层发音规划
- **16维发音特征**: 唇/舌/下颌/软腭/声门 + 发音部位/方式/清浊/鼻音/边音/r音
- **SMA序列**: 词序列 → 共发音平滑过渡 (coarticulation weight=0.3)
- **运动副本**: 发音计划 → 预期听觉 → AF背侧 → 自我监控

### TPJ 语用理解
- **意图推断**: 话语+说话人模型+情境 → Hebb检索意图
- **反讽检测**: 字面正面 + 低信任 + 情境负面 + 高熟悉度 → sarcasm_score
- **语用丰富化**: 字面理解(w_literal) + 意图推断(w_intent) → 真正理解
- **说话人模型**: 从语料角色对话中无监督学习角色"语言指纹"

---

## v5.7 核心变更

### 开发哲学: 从"预训练demo"到"持续成长的智能体"

v5.7 不是新增脑区模块，而是解决一个根本矛盾: **Agent 有完整的脑，但没有生命。**

- **v5.6 之前**: `main_dialogue.py` 是单次 demo — 启动→预热→演示→丢弃
- **v5.7**: 会话持久化 + 发育年龄 + 多模态同步 + 实时传感器 — Agent 开始"活"了

### 发育年龄系统 (Developmental Age)

| 阶段 | 年龄 | 回应模式 | 学习策略 | Trigram阈值 |
|------|------|---------|---------|-------------|
| 婴儿 | 0 | 纯模仿 (回响人类内容词) | 仅学习人类输入 | <50 |
| 儿童 | 1 | 模仿+trigram混合 | 人类+长回应(≥6字) | 50-200 |
| 青少年 | 2 | trigram链生成 | 全部学习 | 200-1000 |
| 成人 | 3 | 全自主生成 | 全部学习 | >1000 |

- **模仿学习**: `_extract_content_words()` 提取内容词 (≥2字+重要单字), 过滤功能词
- **防止自我污染**: age=0 时 `learn_from_interaction()` 阻止Agent学习自己的胡话
- **自动升级**: trigram网络生长到阈值自动晋级, 控制台公告
- **手动控制**: `/age [0-3]` 命令

### 会话持久化 (Session Persistence)

- **全状态 save/load**: 所有Hebb网络 (L0/AF/AG/TPJ/SelfModel) + 身体稳态 + 神经调节 (Hypothalamus/VTA/LC) + 语言系统 (PhonLoop/PhraseStructure/MotorCortex) + 元学习参数 + 对话上下文 + 追踪历史
- **自动保存**: 每10轮对话、Ctrl+C/exit时、异常退出时
- **自动恢复**: 启动时自动加载最新存档, 跳过热身, 继续成长
- **存档路径**: `.notme/sessions/agent_YYYYMMDD_HHMMSS.pkl`
- **版本兼容**: 存档包含version字段, 支持跨版本迁移
- **命令**: `/save [name]`, `/load [name]`

### 多模态同步输入 (Input Bus)

- **核心问题**: 旧版 `if-elif-else` 单选输入 — 每次只能有一个通道活跃, 无法形成 Hebb 跨模态绑定
- **v5.7 方案**: `InputBus` 每帧同时构建完整感知向量 s∈R^516
  - 文本+视觉+听觉+痛觉+触觉+说话人身份 同时填入同一个s
  - Hebb网络一次性学习跨模态共现模式
- **输入模式**:
  - 纯文本 / `img:file.jpg` 图像+文本 / `audio:file.wav` 音频+文本
  - `pain:0.7` 痛觉刺激 / `speaker:name` 说话人身份
  - `stream start/stop` 实时摄像头+麦克风流

### 实时传感器流 (Camera + Microphone)

- **CameraInput**: OpenCV采集, 可配置FPS+分辨率, 非阻塞, 自动重连
- **MicrophoneStream**: sounddevice采集, chunk-based (200ms), mel频谱(32ch)兼容耳蜗核
- **StreamSession**: 组合管理, 异步采集线程与Agent处理线程分离
- **稀疏响应**: Agent不每帧说话, 只在文本输入/显著事件/内部言语触发时才回应
- **命令**: `/stream start` / `/stream stop` / `/stream status`

### 纯净模式 (Clean Mode)

- **零预训练**: 不喂corpus.txt到海马, 不训练trigram网络, 不预训练AF/PhraseStructure/AngularGyrus/TPJ
- **Broca**: `load_corpus=False` — 仅保留12K词表+词向量用于TTS输出
- **知识生长**: 所有语言知识从与用户的对话中在线学习
- **婴儿起点**: 初始年龄=0, trigram=0, L0 cluster=0, 句记忆=0
- **杏仁核种子**: 45个中文情感词弱先验 (低count + 低weight, 易被后续学习覆盖)

### Claude Code 风格终端 UI

- **Rich 渲染**: Panel/Table/Layout — 彩色终端, 自适应窗口
- **prompt_toolkit 输入**: Tab自动补全、输入历史(跨会话持久化)、多行输入
- **斜杠命令**: `/status` `/diag` `/memory` `/body` `/pain` `/touch` `/speaker` `/read` `/save` `/load` `/reset` `/stream` `/age` `/help`
- **Header**: 常驻顶栏 — Version/Session/Turns/Clusters/F/V/A/Age/Trigrams
- **信息密度分级**: Header (默认) → `/status` 展开五面板 (语言/感知/神经调节/身体/自由能)

### DEAD 模块激活

| 模块 | v5.6 状态 | v5.7 状态 |
|------|----------|----------|
| TPJ | 已定义, 需要手动传speaker_name | ✅ `comprehend()`默认speaker="human", 始终活跃 |
| AngularGyrus | 已训练, `read()`从未被调用 | ✅ `comprehend()`自动调用AG阅读通路 |
| PhraseStructure | 训练5000样本但仅打印诊断 | ✅ `broca.speak_from_state()`接受phrase_network并实际调制词选择 |
| 双重管线 | main_dialogue手动填s + agent.step重新处理 | ✅ InputBus统一构建, agent.step()是唯一处理入口 |

### 已知问题 (v5.7)

1. **v5.7 纯净模式网络生长慢**: 从零开始, 需要数十轮对话才能形成基本语言能力
2. **婴儿模仿可能不自然**: 内容词回响有时产生电报式输出 ("你好 天气 真好 呢")
3. **跨模态绑定需长期互动**: COCO预训练模型被绕过, 视觉↔文本关联需从对话中学习
4. **小脑内部模型未实现**: 前向/逆向模型预留但无代码
5. **跨会话持久化格式**: `.pkl` 依赖Python pickle协议, 跨版本兼容性需持续维护
6. **实时流性能**: 摄像头+麦克风同时采集+视觉/听觉全管线处理, 需测试长时间运行稳定性

---

*由 v5.7 交互系统自动更新 · 基于 发育心理学 (Piaget 婴儿期, Vygotsky 最近发展区, 婴儿指向语言/模仿学习) + 多模态感觉整合 (Stein & Meredith 多感觉神经元) + 自由能原理 (Friston 预测编码)*
