# NotMe Changelog

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
