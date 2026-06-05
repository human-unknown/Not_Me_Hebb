# 布罗德曼分区 (Brodmann Areas) 参考

> **定位**: 布罗德曼分区不是脑区层级中的任何一级，而是 Korbinian Brodmann 在 1909 年对大脑皮层进行的**细胞架构编号系统**。他将大脑皮层按切片观察到的不同细胞排列模式依次编号 (BA1→BA52)，数字纯粹是发现顺序，**不反映任何层级关系或功能亲疏**。
>
> 因此，本文档作为**备注/标注系统**附在项目架构中，供跨脑区参考。每个分区的 "所在模块" 指向 v4.0 脑区层级中的具体位置。

---

## 布罗德曼分区的来源

Korbinian Brodmann (1868-1918) 通过 Nissl 染色观察皮层各区域的细胞密度、层状结构差异，将大脑皮层划分为 52 个区域。编号从 BA1 开始，按切片顺序递增——BA1 是他找到的第一个独特细胞模式，BA2 是第二个，以此类推。这解释了为什么功能相关的区域 (如 BA44 和 BA45 共同构成布罗卡区) 有时数字相邻，而其他功能相关区域 (如 BA4 运动皮层 和 BA17 视皮层) 却相隔甚远。

Brodmann 后来将自己的 52 个分区归入 **11 个 Regionen** (大区):
- Regio postcentralis (BA1-3)     — 体感
- Regio praecentralis (BA4)       — 运动
- Regio frontalis (BA6-11, 44-47) — 额叶
- Regio insularis (BA13-16)       — 岛叶
- Regio parietalis (BA5, 7, 39-40) — 顶叶
- Regio temporalis (BA20-22, 37-42, 52) — 颞叶
- Regio occipitalis (BA17-19)     — 枕叶
- Regio cingularis (BA23-33)      — 扣带
- Regio retrosplenialis (BA26-30) — 压后
- Regio rhinencephali (BA34-36)   — 嗅脑
- Regio orbitalis (BA11-12)       — 眶额

现代神经科学沿用编号出于便利/标准化，但功能性划分常跨越或细分这些边界。

---

## 完整 BA 分区 × NotMe v4.0 架构对照

### 额叶 (Frontal Lobe) ← cerebrum/frontal_lobe/

| BA | 名称 | 功能 | NotMe 模块 | 状态 |
|----|------|------|-----------|------|
| BA4 | 初级运动皮层 (M1) | 执行随意运动 | `frontal_lobe/motor_cortex.py` | 待实现 |
| BA6 | 前运动皮层 + SMA | 运动规划/序列 | `frontal_lobe/motor_cortex.py` | 待实现 |
| BA8 | 额叶眼区 (FEF) | 眼动/注意定向 | `parietal_lobe/spatial_attention.py` | 待实现 |
| BA9 | 背外侧前额叶 (dlPFC) | 工作记忆/执行 | `frontal_lobe/prefrontal.py` | **已实现** |
| BA10 | 额极 (FPC) | 认知分支/multitasking | `frontal_lobe/prefrontal.py` | 待实现 |
| BA11 | 眶额皮层 (OFC) | 价值评估/奖赏 | `frontal_lobe/orbitofrontal.py` | 待实现 |
| BA12 | 眶额皮层 | 同 BA11 | `frontal_lobe/orbitofrontal.py` | 待实现 |
| BA44 | 布罗卡区 (pars opercularis) | 语法/语音产出 | `frontal_lobe/broca.py` | **已实现** |
| BA45 | 布罗卡区 (pars triangularis) | 语义检索 | `frontal_lobe/broca.py` | **已实现** |
| BA46 | 背外侧前额叶 | 空间工作记忆 | `frontal_lobe/prefrontal.py` | 部分实现 |
| BA47 | 额下回眶部 | 语义处理 | `frontal_lobe/broca.py` | 部分实现 |

### 顶叶 (Parietal Lobe) ← cerebrum/parietal_lobe/

| BA | 名称 | 功能 | NotMe 模块 | 状态 |
|----|------|------|-----------|------|
| BA1,2,3 | 初级体感皮层 (S1) | 触觉/痛觉/温度 | `parietal_lobe/somatosensory.py` | 待实现 |
| BA5,7 | 后顶叶联合区 | 空间注意力 | `parietal_lobe/spatial_attention.py` | 待实现 |
| BA39 | 角回 | 阅读/数学/语义 | `parietal_lobe/tpj.py` | 待实现 |
| BA40 | 缘上回 | 语音/动作观察 | `parietal_lobe/tpj.py` | 待实现 |
| BA43 | 中央下区 | 味觉 | `parietal_lobe/somatosensory.py` | 待实现 |

### 颞叶 (Temporal Lobe) ← cerebrum/temporal_lobe/

| BA | 名称 | 功能 | NotMe 模块 | 状态 |
|----|------|------|-----------|------|
| BA20 | 颞下回 | 视觉物体识别 | `temporal_lobe/it_cortex.py` | 待实现 |
| BA21 | 颞中回 | 语义记忆 | `temporal_lobe/wernicke.py` | 部分实现 |
| BA22 | 韦尼克区 (后部) | 语言理解 | `temporal_lobe/wernicke.py` | **已实现** |
| BA37 | 梭状回 | 面孔/文字识别 | `temporal_lobe/fusiform.py` | 待实现 |
| BA38 | 颞极 | 社会-情绪处理 | `temporal_lobe/wernicke.py` | 部分实现 |
| BA41 | 初级听皮层 (A1) | 听觉输入 | `temporal_lobe/auditory_cortex.py` | 待实现 |
| BA42 | 次级听皮层 (A2) | 听觉整合 | `temporal_lobe/auditory_cortex.py` | 待实现 |
| BA52 | 副听区 | 听觉 | `temporal_lobe/auditory_cortex.py` | 待实现 |

### 枕叶 (Occipital Lobe) ← cerebrum/occipital_lobe/

| BA | 名称 | 功能 | NotMe 模块 | 状态 |
|----|------|------|-----------|------|
| BA17 | 初级视皮层 (V1) | 边缘/方向/空间频率 | `occipital_lobe/v1.py` + `visual_pathway.py` | **已实现** |
| BA18 | 次级视皮层 (V2) | 轮廓整合 | `occipital_lobe/v2.py` + `visual_pathway.py` | **已实现** |
| BA19 | 第三视皮层 (V3/V4/V5) | 颜色/运动/形状 | `occipital_lobe/v4.py` + `visual_pathway.py` | **已实现** |

### 边缘系统 (Limbic System) ← cerebrum/limbic_system/

| BA | 名称 | 功能 | NotMe 模块 | 状态 |
|----|------|------|-----------|------|
| BA23 | 后扣带回 (PCC) | DMN 核心节点 | `association/dmn.py` | 部分实现 |
| BA24 | 前扣带回 (ACC) | 冲突监测/共情 | `limbic_system/cingulate.py` | **已实现** |
| BA25 | 膝下扣带 | 情绪调节/抑郁 | `limbic_system/cingulate.py` | 待实现 |
| BA26-30 | 压后皮层 | 空间导航/情景记忆 | `limbic_system/hippocampus.py` | 待实现 |
| BA31-33 | 扣带回 | DMN/突显网络 | `association/dmn.py` | 部分实现 |
| BA34-36 | 内嗅/旁海马皮层 | 空间导航/记忆 | `limbic_system/hippocampus.py` | 待实现 |
| BA13-16 | 岛叶 | 内感受/自我意识 | `body/interoception.py` | 待实现 |

### 基底神经节 (Basal Ganglia) ← cerebrum/basal_ganglia/

| 结构 | 功能 | NotMe 模块 | 状态 |
|------|------|-----------|------|
| 尾状核 + 壳核 | 习惯学习 | `basal_ganglia/striatum.py` | 待实现 |
| 伏隔核 (NAc) | 奖赏/动机 | `basal_ganglia/striatum.py` | 待实现 |
| 苍白球 (GPe/GPi) | 动作门控输出 | `basal_ganglia/pallidum.py` | 待实现 |
| 底丘脑核 (STN) | 动作抑制 | `basal_ganglia/subthalamic.py` | 待实现 |
| MoE Gate | 动作选择/疲劳 | `basal_ganglia/action_gating.py` | **已实现** |

### 脑干 (Brainstem) ← brainstem_cerebellum/

| 核团 | 神经递质 | 功能 | NotMe 模块 | 状态 |
|------|---------|------|-----------|------|
| VTA (A10) | 多巴胺 | 奖赏预测误差 | `midbrain/vta.py` | 待实现 |
| SNc (A9) | 多巴胺 | 运动/习惯 | `midbrain/substantia_nigra.py` | 待实现 |
| 蓝斑核 (A6) | 去甲肾上腺素 | 唤醒/注意 | `pons/locus_coeruleus.py` | 待实现 |
| 中缝核 (B1-B9) | 血清素 | 情绪/睡眠 | (待添加) | 待实现 |

---

## 核心规则

1. **BA 分区是标注，不是层级** — 它是皮层表面的编号网格，不是层级包含关系
2. **编号不反映功能亲疏** — BA4(运动)和BA17(视觉)不相邻 ≠ 功能无关
3. **一个脑区可跨多个BA** — 布罗卡区 = BA44+BA45，韦尼克区 = BA22
4. **一个BA可跨多个功能** — BA37既是梭状回(面孔)也是 VWFA(文字)
5. **现代细分更精细** — Glasser et al. (2016) 的 HCP 分区已达 180 区/半球

---

## 参考文献

- Brodmann, K. (1909). *Vergleichende Lokalisationslehre der Grosshirnrinde.*
- Zilles, K., & Amunts, K. (2010). Centenary of Brodmann's map — conception and fate.
- Glasser, M. F., et al. (2016). A multi-modal parcellation of human cerebral cortex. *Nature.*
