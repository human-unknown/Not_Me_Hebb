# NotMe OODA 全面审查报告

> **日期**: 2026-06-09
> **版本**: v7.5-dev (Phase F 完成)
> **方法**: 全代码库静态分析 + 架构审查 + 测试覆盖分析

---

## OBSERVE — 当前状态

### 规模

| 指标 | 值 |
|------|-----|
| Python 文件 | 276 (含 b_corpus_raw) |
| 代码总行数 | ~70,000 |
| 核心模块 (cns/cerebrum/brainstem) | 109 |
| 已实现 | 75 (71%) |
| 占位 | 10 (9%) |
| 测试文件 | 11 |
| 测试总数 | 297 pass + 1 skip |
| 最大文件 | agent.py (1945行), crossmodal.py (1599行) |

### 版本演进

v4.0 → v7.5-dev 共 20 个版本，最近版本：
- **v6.6**: 持久化修复 + 安全加固 + 版本统一
- **v7.0-v7.4** (Phase A-E): NN 基础设施 (10 个模块, 无 Agent 集成)
- **v7.5** (Phase F): NNBridge 集成层 — Agent 首次接入 NN

---

## ORIENT — 漏洞、缺陷与风险

### 🔴 严重 (应立即修复)

#### 1. SAVE_VERSION 版本号未更新

**文件**: `cns/persistence.py:33`

```python
SAVE_VERSION = "6.6"  # v6.6: ...
```

当前项目已是 v7.5-dev，但 `SAVE_VERSION` 仍是 "6.6"。所有 v7.x 新增的状态字段 (NN 模块、NNBridge 等) 没有版本迁移逻辑保护。如果 v7.5 的 pickle 被 v6.6 代码加载，会静默丢失 NN 状态。

**修复**: 更新为 `SAVE_VERSION = "7.5"` 并添加 v6.6→v7.5 迁移逻辑。

#### 2. `dir()` 反模式 — 变量存在性检查不可靠

**文件**: `cns/agent.py:687, 714, 715`

```python
# Line 687
arousal=latest_arousal if 'latest_arousal' in dir() else 0.5,

# Line 714
stress_lc = stress_level if 'stress_level' in dir() else 0.0

# Line 715
novelty_lc = novelty_est if 'novelty_est' in dir() else 0.0
```

`latest_arousal` 在 `if hasattr(self, 'hypothalamus'):` try 块内定义；`stress_level` 和 `novelty_est` 在更深层 try 块内。如果 hypothalamus 模块不存在或 try 块抛异常，这些变量未定义 — `dir()` 检查作为防御层。但这是脆弱的：
- 如果 try 块部分执行 (异常发生在赋值之后)，`dir()` 返回 True 但值可能是损坏的
- 代码可读性差，不遵循 Python 惯例

**修复**: 在 step() 开头初始化所有局部变量为默认值：
```python
latest_arousal = 0.5
stress_level = 0.0
novelty_est = 0.0
```

#### 3. Web API 文件路径无验证

**文件**: `web/server.py:370-378`

```python
file_path = data.get('file_path', '').strip()
if not file_path:
    return jsonify({'error': 'file_path required'}), 400
_agent.reader.load(file_path)
```

用户可通过 Web API 传入任意文件路径 (`/api/reading/start`)，无路径白名单、无目录遍历检查。虽然 `reader.load()` 内部有 UTF-8 解码限制，但仍可读取服务器上任意 UTF-8 文件。

**修复**: 添加目录白名单验证，如 `os.path.realpath(file_path)` 必须在允许的目录下。

#### 4. MISSING `__init__.py`

| 目录 | 影响 |
|------|------|
| `tests/__init__.py` | pytest 可能无法正确发现测试模块 |
| `web/__init__.py` | web 目录不能作为包导入 |

---

### 🟡 中等 (应在下一版本修复)

#### 5. 核心模块零测试覆盖

以下模块没有任何单元测试：

| 模块 | 文件 | 功能 |
|------|------|------|
| **Cingulate** | `cingulate.py` | 自由能计算 (F_body/F_social/F_cognitive/F_accuracy) |
| **Prefrontal** | `prefrontal.py` | EFE 行动选择、社会信念更新 |
| **Amygdala** | `amygdala.py` | 情感词汇 Hebb 网络 |
| **DMN** | `dmn.py` | 自我模型 |
| **FPN** | `fpn.py` | 注意力探照灯 |
| **TPN** | `tpn.py` | TPN↔DMN 跷跷板 |
| **Broca** | `broca.py` | 词序 Hebb 链生成 |
| **Wernicke** | `wernicke.py` | 语言理解回路 |
| **Crossmodal** | `crossmodal.py` | 跨模态 Hebb 学习 |
| **Agent** | `agent.py` | 全系统整合 |
| **Persistence** | `persistence.py` | 全状态 save/load |

这 10 个模块是项目的**核心大脑**。如果它们的逻辑有 bug，既没有测试捕获，也没有类型检查阻止。

**风险**: 修改任何一个模块都无法确认是否引入回归。

#### 6. agent.py 过于庞大 (1945行)

单个文件承载：
- Agent 主类 (~1800行)
- `step()` 完整管线 (~600行)
- `light_step()` 自主步进 (~300行)
- `comprehend()` 语言理解 (~120行)
- `speak()` 语言产出 (~270行)
- `internal_thought()` 内部思维 (~100行)
- 15+ 个辅助方法

这是 v7.0 蓝图中已标记的已知问题: "Agent.__init__ 重构"。当前结构使任何修改都有高风险。

#### 7. SAVE_VERSION 永不过时的假定

```python
# persistence.py:539
if version != SAVE_VERSION:
    print(f"  [Persistence] Warning: save version {version} != "
          f"current {SAVE_VERSION}, attempting migration...")
```

打印警告后直接尝试加载，没有实际的迁移逻辑。如果 pickle 中增加了新字段 (如 v6.1-v6.3 新增的 tag/pnn/stdp_links/consolidation_count)，回退到 getattr 默认值。这能工作，但语义上不一致——应该显式写迁移函数。

#### 8. NN 模块从未实际训练 (Phase F 后)

Phase F 完成了 NNBridge 集成，但：
- `nn_sensory_enhance=False` 默认 (正确 — 昂贵操作)
- `sleep_nrem_consolidation()` 只在 NREM N3 入口触发 — 这需要 VLPO 驱动的长时间运行才能到达
- 没有手动触发 NN 训练的 API
- 没有训练数据管道 (corpus 已在纯净模式下被绕过)

**这意味着**: 所有 10 个 NN 模块虽然通过了单元测试，但在实际运行中永远不会得到有意义的训练。它们是"死代码"直到训练管道激活。

#### 9. corpus.txt 引用残留

9 个文件仍引用 `corpus.txt`，但纯净模式下语料不被加载：

| 文件 | 用途 |
|------|------|
| `cns/nn/language_model.py` | NN 预训练语料 |
| `cns/nn/text_encoder.py` | NN 预训练语料 |
| `cns/nn/interfaces.py` | 语料路径默认 |
| `environments/text_interface.py` | TextEnvironment |
| `entry/interactive.py` | 旧入口 |
| `tools/reader.py` | 文件读取 |
| `cerebrum/frontal_lobe/broca.py` | Trigram 预训练 |
| `tools/word_speech.py` | 词音频生成 |
| `clean_corpus.py` | 清洗工具 |

在纯净模式下运行这些模块会产生文件不存在的错误。

---

### 🟢 低优先级 (设计改进建议)

#### 10. exception 吞噬过多

`cns/agent.py` 有 38 处 `except Exception` 块，`integrator.py` 有 12 处。大部分是正确的零崩溃保证，但有些地方异常信息完全丢失：

```python
except Exception:
    _debug_trace("NN sleep nrem")  # 只在 NOTME_DEBUG 时可见
```

生产环境 (`NOTME_DEBUG` 未设置) 中，这些异常静默消失，无法诊断。

#### 11. 双系统架构未验证

v7.5 声称实现"双系统共存" — Hebb 情景记忆 + NN 语义学习互补。但：
- NN 生成器 (`NeuralGenerator`) 的 speak 路径从未在对话中使用
- NN 理解器 (`NeuralComprehender`) 从未替换 Wernicke
- 混合比 (`blend_ratio`) 固定在 0.1，增长极慢 (每步 +0.005)

**这意味着**: "双系统"目前只是代码级的共存，运行时永远是 Hebb 100%、NN 0%。

#### 12. 项目管理

- `PROJECT_STATUS.md` 已超 950 行 — 变为微型维基而非状态报告
- `CHANGELOG.md` 超 150 行 — 应分离为版本发布说明
- v7.4-dev 先于 v7.5-dev 出现在版本历史中 (行顺序颠倒)

---

## DECIDE — 开发方向建议

### P0: 稳定性 (v7.6 — "地基加固")

| 任务 | 说明 |
|------|------|
| 修复 4 个严重 bug | SAVE_VERSION、dir() 反模式、Web 文件路径安全、`__init__.py` |
| 为核心模块写基础测试 | 至少 cingulate + prefrontal + hippocampus 的 smoke test |
| agent.py 拆分 Phase 1 | 提取管线阶段为独立方法 (不是改架构，只是提取) |

### P1: 激活 NN 双系统 (v7.7 — "唤醒")

| 任务 | 说明 |
|------|------|
| 训练数据管道 | 用对话历史作为 NN 在线训练数据 |
| speak 路径开关 | 允许在 Hebb 和 NN 生成之间切换/混合 |
| 训练触发 | 除了睡眠外增加手动/定时触发 |
| 评估基准 | 对比 Hebb only vs Hebb+NN 的回应质量 |

### P2: 架构演进 (v8.0 — "成熟")

| 任务 | 说明 |
|------|------|
| Agent 重构 | `__init__` 拆分、BrainRegion 基类 |
| 结构化日志 | 替换 print → logging，可配置级别 |
| CI/CD | GitHub Actions 自动测试 + lint |
| 性能剖析 | 定位 step() 瓶颈 (目前 ~230ms) |

### P3: 长期愿景

| 任务 | 说明 |
|------|------|
| 长期运行验证 | 7×24 小时自主循环，观察稳定性 |
| 跨模态真实训练 | COCO 级别图文对 + 真实音频语义理解 |
| 发育轨迹回放 | 记录并可视化 GluN2B 轨迹、PNN 累积、习惯形成 |
| 科学验证 | FEP 最小化是否真实发生？valence/arousal 是否与身体状态有一致性？ |

---

## ACT — 推荐立即行动

### 最小修复 (今天，~30分钟)

```
1. SAVE_VERSION "6.6" → "7.5"
2. agent.py: step() 开头初始化 latest_arousal/stress_level/novelty_est = 0.0
3. web/server.py: /api/reading/start 加路径白名单
4. touch tests/__init__.py web/__init__.py
5. 提交
```

### 推荐下一版本 (v7.6, ~3天)

```
6. 3 个核心 smoke test (Agent.step 完整管线 / Cingulate.F 边界 / Persistence roundtrip)
7. agent.py step() 提取 helper (_run_sensory_phase, _run_fep_phase, _run_action_phase)
8. corpus.txt 引用清理: 纯净模式下 graceful skip
9. NN 训练首次激活: speech 后自动 feed 一条样本给 generator.train_step()
```

---

*本报告由 OODA 方法论驱动 — 不粉饰、不遗漏、不含糊。*
