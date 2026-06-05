# NotMe 项目状态报告

> **版本**: v4.1 — 大脑功能细分
> **日期**: 2026-06-05
> **上一版本**: v4.0 — 人脑层级结构
> **分支**: main

---

## 一、版本概述

v4.1 基于 [图2 — 大脑各结构功能分工](人脑结构调查_可视化图表集.html) 对 v4.0 架构进行两项核心改进：

1. **丘脑独立**: 按图2将丘脑从边缘系统中分离为独立 Level 3 结构
2. **消除代码重复**: 22 个根目录 `.py` 文件从完整副本转为 re-export shim

---

## 二、架构变更

### 2.1 脑区层级 (当前状态)

```
Level 1: CNS (cns/)                          6 文件
Level 2: Cerebrum (cerebrum/)                29 文件
  ├── frontal_lobe/     4  (prefrontal★ broca★ motor_cortex□ orbitofrontal□)
  ├── parietal_lobe/    3  (somatosensory□ spatial_attention□ tpj□)
  ├── temporal_lobe/    4  (wernicke★ auditory_cortex□ it_cortex□ fusiform□)
  ├── occipital_lobe/   6  (visual_pathway★ retina_lgn★ gestalt★ v1□ v2□ v4□)
  ├── limbic_system/    5  (hippocampus★ amygdala★ cingulate★ hypothalamus□ olfactory□)
  ├── basal_ganglia/    4  (action_gating★ striatum□ pallidum□ subthalamic□)
  ├── thalamus/         1  (thalamus□) ← v4.1 新增
  └── association/      2  (dmn★ crossmodal★)
Level 2: Brainstem+Cerebellum (brainstem_cerebellum/) 10 文件
  ├── midbrain/         3  (vta□ substantia_nigra□ superior_colliculus□)
  ├── pons/             2  (locus_coeruleus□ reticular_formation□)
  ├── medulla/          1  (autonomic□)
  ├── cerebellum/       2  (motor_coordination□ predictive_timing□)
  └── neuromodulatory/  2  (meta_learning★ plasticity□)
工程层: environments/ tools/ entry/ body/ spinal/
```

**★ = 已实现 (13) | □ = 占位 (22)**

### 2.2 文件变更明细

| 操作 | 文件 | 说明 |
|------|------|------|
| + 新建 | `cerebrum/thalamus/` | 丘脑独立 Level 3 |
| − 移除 | `cerebrum/limbic_system/thalamus.py` | 从边缘系统迁出 |
| + 恢复 | `layer0_gestalt.py` | v4.0 误删恢复 (3 个文件引用) |
| + 归位 | `cerebrum/occipital_lobe/gestalt.py` | Gestalt 归入枕叶视觉通路 |
| ~ 转换 | 22 个根目录 `.py` | 完整副本 → re-export shim |
| ~ 修复 | 8 个层级内部文件 | flat import → 包路径 |
| ~ 修复 | `cns/__init__.py` | `masked_cosine` 导入 bug |
| ~ 标注 | `visual_interface.py` | 废弃说明，指向新视觉模块 |

---

## 三、代码质量

### 3.1 重复代码消除

| 指标 | v4.0 | v4.1 |
|------|------|------|
| 重复文件对数 | 22 对 (SHA256 相同) | 0 |
| 根目录 shim | 0 (全部完整副本) | 22 (薄 re-export) |
| 层级内 flat import | 全项目 | 仅工程层遗留 |
| 权威代码位置 | 根目录 (实际运行) | cerebrum/cns/brainstem |

### 3.2 导入路径规范 (v4.1)

```
cns/*              → from cns.xxx import Y
cerebrum/**        → from cns.xxx / cerebrum.xxx import Y
brainstem/**       → from cns.xxx import Y
root shim          → from <hierarchy>.xxx import Y, Z
```

### 3.3 已知遗留

| 项目 | 状态 | 版本 |
|------|------|------|
| `environments/text_interface.py` | flat import `from data_types` | v5.0 |
| V1/V2/V4 独立模块 | 存根 (实现在 visual_pathway.py) | 待实现 |
| 脑干核团 (VTA/SN/LC) | 占位 + TODO | 待实现 |
| 下丘脑 Body ODE | 占位 + TODO | 待实现 |
| 丘脑感觉门控 | 占位 + TODO | 待实现 |

---

## 四、模块成熟度

### 4.1 已实现模块 (14)

| 模块 | 脑区 | 功能 | 行数 |
|------|------|------|------|
| `hippocampus.py` | 海马 | Hebb 集群记忆、睡眠巩固 | 452 |
| `cingulate.py` | 扣带回/ACC | 自由能计算、效价/唤醒 | 309 |
| `prefrontal.py` | 前额叶 dlPFC | EFE 行动选择 | 303 |
| `broca.py` | 布罗卡区 BA44/45 | Hebb 词序链生成 | 897 |
| `wernicke.py` | 韦尼克区 BA22 | 语言理解、对话记忆 | 546 |
| `amygdala.py` | 杏仁核 | Hebb 情感词汇网络 | 308 |
| `action_gating.py` | 基底节 | MoE 动作门控 | 79 |
| `dmn.py` | DMN | 自我模型 | 150 |
| `visual_pathway.py` | 枕叶 V1-V4 | Gabor 视觉通路 | 857 |
| `retina_lgn.py` | 视网膜→LGN | 图像 Gabor 编码 | 131 |
| `gestalt.py` | 枕叶 | Gestalt 知觉分组 | 634 |
| `meta_learning.py` | 神经调节 | 元学习、关键期 | 191 |
| `crossmodal.py` | 联合皮层 | 跨模态 Hebb 绑定 | 1591 |
| `agent.py` | CNS | 全系统整合 | 506 |

### 4.2 占位模块 (22)

全部含完整 TODO 清单、接口设计、参考文献，标注优先级。

---

## 五、验证结果

```
[OK] 24/24 根目录 shim 重导出通过
[OK]  9/9  cerebrum + brainstem 层级导入通过
[OK]  1/1  entry/main_dialogue.py 导入上下文通过
[OK]  1/1  Agent.step() 全链路 (L0→L1→L2→L3) 通过
[OK]  0   孤立 .pyc / 破损导入
```

端到端: Agent 创建 → step() → 自由能计算 → 行动选择 → 身体更新:
```
clusters=0 → step → F_total=0.069, valence=0.000, clusters=1
```

---

## 六、下一步计划 (v5.0 候选)

1. **工程层导入修复**: `text_interface.py` → 层级导入
2. **下丘脑实现**: BodyVector ODE 动态 setpoint + 驱力计算
3. **丘脑实现**: 感觉门控、TRN 注意力探照灯
4. **V1/V2/V4 独立**: 从 `visual_pathway.py` 拆分为独立模块
5. **脑干核团**: VTA (多巴胺)、蓝斑核 (NE) 神经调节模型
6. **根目录完全消除**: 仅从层级导入 (移除 shim 层)

---

## 七、运行

```bash
# 对话模式
python entry/main_dialogue.py

# 跨模态学习
python stage2_crossmodal.py --dataset coco --n 5000 --mode all
```

向后兼容: 根目录 `.py` 文件保留为 shim。`python main_dialogue.py` 透明重定向到层级代码。

---

> 报告生成: 2026-06-05 | v4.1
