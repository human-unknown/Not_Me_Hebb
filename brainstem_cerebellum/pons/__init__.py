"""
脑桥 (Pons)  [Level 3]

功能：呼吸节律生成 · 睡眠/觉醒切换 · 去甲肾上腺素(NE)调控 · 听觉双耳整合

核心核团:
├── 蓝斑核 (Locus Coeruleus)     — NE 释放 → 调控唤醒度、注意力、应激
├── 桥网状结构 (Pontine RF)      — 睡眠切换 (REM-on/off 神经元)
├── 中缝核 (Raphe Nuclei)        — 5-HT(血清素) → 情绪、睡眠、食欲
├── 呼吸中枢 (Pneumotaxic)       — 呼吸节律调节
├── 桥核 (Pontine Nuclei)        — 皮层-小脑通路中继
├── 耳蜗核 (Cochlear Nucleus)    — v5.2: 听觉CNS第一中继站, tonotopic编码
├── 上橄榄复合体 (SOC)            — v5.2: 双耳ITD/ILD定位, 双重理论
└── 外侧丘系核 (LL Nuclei)       — v5.2: 时间模式 + 双耳GABA抑制

子模块:
├── locus_coeruleus.py       蓝斑核 — NE 调控、唤醒度调制、SNR增强 ★v5.5
├── vlpo.py                  视前区腹外侧核 — 睡眠-觉醒触发器开关 + NREM/REM振荡 ★v6.3
├── reticular_formation.py   网状结构 — 觉醒/睡眠状态机 [待实现]
├── cochlear_nucleus.py      耳蜗核 — 听觉频谱分解 + 相位锁定 ★v5.2
├── superior_olivary.py      上橄榄复合体 — 双耳定位 (MSO/LSO/MNTB) ★v5.2
└── lateral_lemniscus.py     外侧丘系 — 时间增强 + 双耳抑制 ★v5.2
"""
