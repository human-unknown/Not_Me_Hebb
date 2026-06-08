# NotMe Changelog

## v7.5-dev (2026-06) — Phase F: 整合与打磨 (NN ↔ Agent 全系统集成)

### Phase F: 整合与打磨 (2026-06-08)
- 新增 `cns/nn/integrator.py` — NNBridge
  - Agent ↔ NN 模块单一集成点: Agent 通过 5 个钩子调用委托 NN 关注
  - 懒初始化: NN 模块仅在首次使用时创建 (快速启动)
  - `enhance_sensory(s)` — 可选的 NN 编码器增强感知 (混合比从 0.1 开始增长)
  - `record_step(F, valence, ...)` — 每步记录指标到 ExperienceTracker
  - `record_dialogue(user, resp)` — 对话轮次追踪 (词汇 + 个性化)
  - `get_nn_lr_modulation(rpe, da)` — VTA RPE → NN 学习率调制 (阻尼 0.7×)
  - `get_nn_explore_params(tonic_ne, ...)` — LC NE → temperature/dropout 调制
  - `sleep_nrem_consolidation()` — NREM 深睡 → NN 梯度更新 (Hebb 集群回放)
  - `sleep_rem_consolidation()` — REM → 生成器情感偏置衰减 (情感去毒)
  - `save_checkpoint(dir)` / `load_checkpoint(dir)` — 统一 checkpoint
  - `get_status()` / `get_training_summary()` — Web UI 数据
  - 零崩溃保证: 所有 NN 操作包裹在 try/except 中
- 更新 `cns/agent.py` — +NNBridge 集成
  - `Agent.__init__`: +`nn_bridge=None` + `_nn_config=None` (默认禁用, 零开销)
  - `Agent.enable_nn(config)`: 创建 NNBridge 并开启集成
  - `Agent.step()`: 5 个钩子调用 (感官增强 → 指标记录 → VTA 调制 → LC 调制 → 睡眠巩固)
  - 睡眠整合: NREM N3 → NN 梯度更新, REM → 情感衰减
  - 所有钩子失败静默 — NN 故障永不崩溃 Agent
- 更新 `brainstem_cerebellum/midbrain/vta.py` — +`nn_lr_multiplier`
  - VTA.process() 返回新增 `nn_lr_multiplier` 键 (Hebb LR 的阻尼 0.7× 版本)
  - 范围 [0.2, 2.0] — 比 Hebb 保守, 防止灾难性遗忘
- 更新 `brainstem_cerebellum/pons/locus_coeruleus.py` — +`nn_temperature` + `nn_dropout`
  - LC.process() 返回新增 `nn_temperature` [0.3, 1.5] 和 `nn_dropout` [0.05, 0.5]
  - 探索模式 → 高温 (多样生成), 利用模式 → 低温 (确定生成)
  - 低 Yerkes-Dodson → 高 dropout (不确定性)
- 更新 `cns/nn/trainer.py` — +`get_training_history()`
  - 返回 Web UI 的逐模块损失历史 (用于前端 sparkline 图表)
- 更新 `cns/nn/config.py` — +2 集成标志
  - `nn_enabled: bool = False` — NN 集成主开关
  - `nn_sensory_enhance: bool = False` — NN 编码器增强感官 (昂贵)
- 更新 `cns/persistence.py` — NN 持久化集成
  - `save_agent()` 自动保存 NN 模块 (pickle + .pt 双格式)
  - `restore_agent()` 自动加载 NN checkpoint
  - `save_nn_modules()` / `load_nn_modules()` 委托给 NNBridge
- 更新 `web/server.py` — +NN 状态
  - `_build_status()` 新增 `nn` 字段 (NNBridge 状态 + 训练摘要)
  - SSE 实时推送包含 NN 训练数据 (每 500ms)
- 更新 `web/static/index.html` — +NN 训练面板
  - 运行状态面板新增 "◈ NN 训练 ◈" 迷你面板
  - `updateNNPanel()` 函数渲染融合比/LR/温度/模块数
  - 版本号 v6.5 → v7.5
- 更新 `cns/nn/__init__.py` — +NNBridge 导出, v7.4 → v7.5
- 新增 43 个集成测试 (`tests/test_nn_integration.py`)
  - TestNNBridge: 12 (init/ensure/感官增强/指标/VTA调制/LC调制/状态)
  - TestVTANNModulation: 3 (nn_lr_multiplier/范围/正RPE)
  - TestLCNNModulation: 3 (nn_temperature/nn_dropout/范围)
  - TestAgentIntegration: 8 (默认禁用/enable_nn/步进无崩溃/指标/VTA接线/LC接线/桥接)
  - TestWebIntegration: 5 (NN状态/训练历史/摘要/键/禁用状态)
  - TestEmotionalConsistency: 5 (valence范围/arousal范围/F有限/Hebb集群/动作有效)
  - TestSleepNNIntegration: 3 (NREM不崩溃/REM不崩溃/无agent)
  - TestPersistenceIntegration: 3 (保存创建目录/加载无崩溃/往返)
  - TestConfigIntegration: 2 (nn_enabled标志/自定义配置)
- 保持 FEP / 身体稳态 / Hebb 模块 / D=516 布局 零修改
- 保持双系统架构 — Hebb 情景记忆 + NN 语义学习 互补共存

## v7.4-dev (2026-06) — Phase E: 训练与体验闭环 (训练编排器 + 可观察指标)

### Phase E: 训练与体验闭环 (2026-06-08)
- 新增 `cns/nn/trainer.py` — Trainer
  - 统一训练编排器, 替代各模块重复的 pretrain() boilerplate
  - `register(module)` — 注册 NeuralModule 进行训练
  - `pretrain(module_name, corpus)` — 多 epoch 预训练 (shuffle + batch + progress + LR调度)
  - `pretrain_all(corpus, configs)` — 顺序预训练所有已注册模块
  - `online_finetune(module_name, batch)` — 对话中单步低 LR 微调 (独立优化器)
  - `save_checkpoint(dir)` / `load_checkpoint(dir)` — 统一 checkpoint 管理 (所有模块 + trainer 状态)
  - `get_summary()` — 跨所有模块的训练摘要
  - LR 调度支持: cosine annealing / step decay / none
  - 回调模式: `callback(epoch, batch_idx, losses)` 供 UI 进度报告
  - 不替代模块自带的 pretrain() — 双向兼容
- 新增 `cns/nn/metrics.py` — ExperienceTracker + TrainingMetrics
  - **ExperienceTracker**: 可观察指标追踪
    - `record_step(**metrics)` — 灵活记录任意步指标 (F_body/valence/arousal/...)
    - `record_dialogue_turn(user, response, metrics)` — 完整对话轮次记录
    - `get_summary(window)` — 最近 N 步摘要 (avg F/valence/arousal/response)
    - `get_trends(window)` — 趋势分析 (improving/stable/declining, 线性回归斜率)
    - `to_csv(path)` / `to_json(path)` — CSV/JSON 导出 (兼容 telemetry.py)
    - `from_json(path)` — 从 JSON 恢复
    - 字符级词汇追踪 (中文适配) + 用户词频统计 (个性化)
    - `get_user_profile()` — 个性化偏好分析
  - **TrainingMetrics**: 轻量训练进度追踪
    - loss/ppl/LR 历史 + EMA 平滑
    - `is_improving()` / `is_converged()` — 改进/收敛检测
    - `get_best()` — 最佳 loss + epoch
    - `reset()` — 全部指标重置
- 更新 `cns/nn/config.py` — +4 训练配置字段
  - `pretrain_epochs: int = 10` — 默认预训练 epoch 数
  - `online_lr: float = 1e-4` — 在线微调学习率 (远低于预训练)
  - `lr_scheduler: str = "none"` — LR 调度策略 ("cosine"/"step"/"none")
  - `checkpoint_interval: int = 5` — 每 N 个 epoch 保存 checkpoint
- 更新 `cns/nn/__init__.py` — +Trainer / +ExperienceTracker / +TrainingMetrics
- 新增 54 个训练层测试 (`tests/test_nn_training.py`)
  - TestTrainer: 20 测试 (init/register/pretrain/callback/online/checkpoint/summary/pretrain_all/...)
  - TestExperienceTracker: 16 测试 (init/record/summary/trends/vocab/profile/export/roundtrip/...)
  - TestTrainingMetrics: 11 测试 (init/record/is_improving/is_converged/get_best/reset/...)
  - TestIntegration: 7 测试 (trainer+tracker/full_pretrain/online_flow/persistence/trainable/version/config)
- 保持 Agent.step() / FEP / 身体稳态 / 睡眠系统 零修改
- 保持已有模块 pretrain() 方法不变 — Trainer 调用 module.train_step() 内部

## v7.3-dev (2026-06) — Phase D: 语言系统重铸 (神经语言模型)

### Phase D: 语言层 (2026-06-08)
- 新增 `cns/nn/language_model.py` — NeuralGenerator
  - Char-level 自回归 Transformer 解码器 (GPT 式, d_model=256, 4层, 8头, ~5M参数)
  - 因果自注意力 + 输出投影 (weight tying with input embedding)
  - Valence/Arousal 情感条件生成 (特殊 token: [V_POS]/[V_NEG]/[A_HIGH]/[A_LOW])
  - Temperature + top-k 采样解码, 确定性种子
  - 语料预训练 (next-char CE loss + perplexity)
  - 替换: Broca trigram Hebb 链 — 补充 (非替换) 双系统架构
- 新增 `cns/nn/comprehender.py` — NeuralComprehender
  - 记忆增强文本理解: text encoder + memory cross-attention
  - N400 = 1-cosine_sim(input, predicted) — 语义预测误差
  - P600 = char 熵 + 长度因子 — 句法处理成本 (占位)
  - 上下文窗口 (EMA + 时间衰减) + 记忆融合 (加权混合)
  - 替换: Wernicke 记忆检索理解 — 补充 (非替换) 双系统架构
- 新增 `cns/nn/angular_gyrus_nn.py` — NeuralAngularGyrus
  - 2×Conv1d seq2vec 模型 (64→128→256 + GlobalAvgPool, ~200K参数)
  - 字符序列 → 音素/发音特征向量 (64-dim)
  - 配对训练 MSE (char_seqs, phoneme_vecs)
  - 替换: Hebb AngularGyrus (grapheme_to_phoneme) — 补充 (非替换) 双系统架构
- 更新 `cns/nn/__init__.py` — +NeuralGenerator / +NeuralComprehender / +NeuralAngularGyrus
- 新增 40 个语言层测试 (`tests/test_nn_language.py`)
- PhraseStructure / PhonologicalLoop / ArcuateFasciculus 保留不变
- 保持 Agent.step() / FEP / 身体稳态 / 睡眠系统 零修改

## v7.2-dev (2026-06) — Phase C: 记忆层升级 (神经记忆 + 跨模态对比学习)

### Phase C: 记忆层 (2026-06-08)
- 新增 `cns/nn/semantic_store.py` — NeuralSemanticStore
  - FAISS/numpy 向量数据库, 支持 ANN 检索 (cosine similarity)
  - 使用 TrainableTextEncoder 编码文本 → 64d L2-normalized embedding
  - 元数据管理: {text, valence, arousal, timestamp, source, count}
  - 重复检测 (自动合并), 容量驱逐, 年龄遗忘
  - 批量插入, 索引重建, save/load 持久化
  - 补充 (非替换) Hebb SemanticMemory — 双系统架构
- 新增 `cns/nn/crossmodal_nn.py` — CrossModalNN
  - CLIP 式对比学习: Text(64d)→Linear→128d 与 Visual(308d)→Linear→128d 共享空间
  - InfoNCE 对称损失 (温度 τ=0.07), 精度追踪
  - freeze_encoders 模式 (默认冻结, 仅训练投影头 ~50K 参数)
  - Text→Image 和 Image→Text 检索
  - 余弦相似度矩阵计算
  - 补充 (非替换) Hebb crossmodal.py — 双系统架构
- 更新 `cns/nn/__init__.py` — +NeuralSemanticStore / +CrossModalNN
- 新增 37 个记忆层测试 (`tests/test_nn_memory.py`)
- 纹状体 RL 保留为简单 D1/D2 (per 蓝图: "保留为简单RL")
- 保持 Agent.step() / FEP / 身体稳态 / 睡眠系统 零修改

## v7.1-dev (2026-06) — Phase B: 感知层替换 (学习特征提取)

### Phase B: 感知编码器 (2026-06-08)
- 新增 `cns/nn/text_encoder.py` — TrainableTextEncoder
  - 2层 char-level Transformer (d_model=128, nhead=4, max_len=128)
  - MLM 预训练 on corpus.txt → 64-dim L2-normalized 输出
  - 字符级 tokenization (~5000 字表, 4 特殊 token)
  - 替换: MiniLM-L6-v2 (384d) → PCA (64d) 外部模型
- 新增 `cns/nn/visual_encoder.py` — TrainableVisualEncoder
  - 4层 CNN backbone (3→32→64→128→256) + 7 头子通路投影
  - 子通路: M(96), P(112), K(48), IT(16), SC(16), Pulvinar(12), Binding(8) = 308d
  - 自动编码器训练 (reconstruction MSE)
  - 替换: Gabor 滤波 V1→V2→V4→IT 手写管线
- 新增 `cns/nn/audio_encoder.py` — TrainableAudioEncoder
  - 3层 CNN on Mel spectrogram (1→32→64→128) + 4 头子模块投影
  - 子模块: CN(32), SOC(24), IC(24), AC(16) = 96d
  - scipy Mel 计算 (torchaudio 备选)
  - 替换: Mel→耳蜗核→SOC→IC→AC 手写管线
- 更新 `cns/nn/__init__.py` — +TrainableTextEncoder / +TrainableVisualEncoder / +TrainableAudioEncoder
- 新增 48 个编码器测试 (`tests/test_nn_encoders.py`)
- 保持 Agent.step() / FEP / 身体稳态 / 睡眠系统 零修改

## v7.0-dev (2026-06) — Phase A: 神经网络支撑层

### Phase A: 基础架构 (2026-06-08)
- 新增 `cns/nn/` 包 — PyTorch 基础设施
  - `config.py`: NNConfig dataclass (device/dtype/training开关/学习率)
  - `base.py`: NeuralModule 抽象基类 (forward/train_step/save/load)
  - `bridge.py`: numpy↔tensor 桥接 + 自动设备检测 (CPU/CUDA/MPS)
  - `interfaces.py`: TextEncoder(64d) / VisualEncoder(308d) / AudioEncoder(96d) 抽象接口
- 更新 `cns/params.py` — +DEFAULT_NN_PARAMS (10个NN超参数)
- 更新 `cns/persistence.py` — +save_nn_modules / load_nn_modules / has_nn_checkpoint
- 更新 `requirements.txt` — +torch>=2.0.0
- 新增 39 个 NN 模块测试 (`tests/test_nn_base.py`)
- 保持 Agent.step() / FEP / 身体稳态 / 睡眠系统 无修改

### v7.0 前准备 (2026-06-08)
- 全面代码审计 — 93 模块, 59 Theta 参数, D=516
- VLPO 测试修复 — min_stable 适配 v6.5 的 150步 (15→150)
- LGN 测试修复 — 添加 sys.path 设置
- v6 验收测试修复 — Theta 参数数量 32→59
- v7.0 ML 改造蓝图: `docs/superpowers/plans/2026-06-08-v7-ml-transformation-blueprint.md`
- 测试覆盖率: 76/76 (100%) — 所有套件通过
- 版本号: v6.6 → v7.0-dev

## v6.6 (2026-06) — 持久化优化 + 代码质量提升
- Step counter 持久化修复 (agent.meta.step_count 同步)
- SCN 时间改用步数线性映射 (不再倒计时/卡住)
- Visual 训练结果持久化 (不再每次启动重训)
- F_social 修复 — 使用 Hebb 情感词典代替手工规则
- 移除 "无LLM" 原则声明
- 27 处异常块加调试 traceback
- 版本号统一到 v6.6
- Web API 路径遍历安全修复
- 测试辅助函数提取到 tests/conftest.py
- 补全 requirements.txt + 新增 requirements-dev.txt

## v6.5 (2026-05) — Web 前端大改 + sleep-wake social interrupt
- Web 前端 overhaul (Flask REST API + SSE 实时推送)
- sleep-wake 社交中断机制
- 实时传感器流集成 (camera + microphone)
- AutonomousLoop 后台线程模式

## v6.4 (2026-04) — 长期常驻学习
- AutonomousLoop 自主时间流引擎
- InternalLife: 走神回忆 + 内部独白
- Reader: 文件阅读 + 疲劳模型
- Telemetry: CSV 遥测记录
- light_step() 轻量自主步进

## v6.3 (2026-03) — 睡眠与时间维度
- SCN 昼夜节律时钟 (TTFL + Process S)
- VLPO 睡眠-觉醒调控 (触发器 + NREM/REM 双相)
- 双相睡眠: NREM 突触缩小 + REM 情绪去刺痛
- α 注意门控 + 类淋巴清除
- 56 Theta 参数 (L7: +8 睡眠/节律)

## v6.2 (2026-02) — 记忆巩固优化
- 突触标签捕获 (STC 假说)
- 激活持续性 (persistence)
- 巩固锁定 (consolidation lock)
- 56 Theta 参数 (L6: +8 记忆巩固)

## v6.1 (2026-01) — 发育优化
- STDP 时序学习 (pre→post = LTP)
- GluN2B 发育期 NMDA 亚基
- PNN 周围神经网络包裹
- 保护信号 (CD47-SIRPα)
- 沉默突触候选集群
- 可塑性调节器 (整合 GluN2B + 事件 + 稳态 + 神经调质)

## v6.0 (2025-12) — 记忆系统扩展
- 语义记忆 (皮层知识存储: 慢学慢衰大容量)
- 纹状体程序性记忆与习惯学习
- D1/D2 通路平衡
- 动作自动化

## v5.7 (2025-11) — 发育年龄系统 + 会话持久化
- 4 阶段发育年龄系统 (Piaget/Vygotsky)
- 会话全状态 save/load
- 多模态同步输入总线 (text+vision+audio+pain)
- 摄像头/麦克风实时流 (OpenCV+sounddevice)
- Rich 终端 UI (Claude Code 风格)
- 纯净模式 (零预训练, 从对话在线学习)
- D=516 (text[64] | vision[308] | audio[96] | pain[48])

## v5.6 (2025-10) — 语言系统
- 弓状束 (Wernicke↔Broca 腹侧+背侧双通路)
- 语音回路 (Baddeley 模型: ~7组块, ~2s消退)
- 短语结构网络 (BA44 层级句法)
- 角回阅读通路 (视觉字形→语音)
- 运动皮层发音规划 (M1+SMA, 16维发音特征)
- TPJ 语用理解 (心理理论 + 意图推断)
- N400/P600 语言预测误差
- Theta: 24→26 (+w_semantic +w_syntactic)

## v5.5 (2025-09) — 神经调节系统
- 下丘脑稳态 (SetpointModel + DriveSystem + HPA)
- VTA 奖赏预测误差 (事件驱动学习率)
- 蓝斑核 NE (phasic/tonic + SNR + Yerkes-Dodson)

## v5.4 (2025-08) — 痛觉系统
- 7 条痛觉知觉规律
- 脊髓背角闸门控制
- 双通路 (外侧感觉-辨别 + 内侧情感-动机)
- PAG→RVM 下行调控闭环
- D=516

## v5.3 (2025-07) — 真实音频输入
- AudioInput (WAV/MP3/FLAC/麦克风)
- Mel 频谱 → 全听觉管线
- 替换语义代理模式

## v5.2 (2025-06) — 听觉层级全接入
- 耳蜗核→SOC→IC→MGB→听皮层
- 15 条知觉规律 + 真实声学特征
- D=468

## v5.1 (2025-05) — D 维度全局切换
- D=330→372 (V5 视觉层级全管线)
- V4 扩展 (shape 32 + color 16)
- Gestalt 集成 (19d)
- 旧 ImageEncoder 完全替换

## v5.0 (2025-04) — 视觉管线全接入
- 6 条并行子通路 (M/P/K + Pulvinar + Dorsal)
- IT 皮层物体识别 (Hebb 学习)
- MT/MST 运动/光流检测
- SC 上丘视觉反射
- FPN 跨通路绑定
