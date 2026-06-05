"""
tpn.py — 任务正网络 (Task Positive Network)

对应脑区: 任务正网络 (TPN) — 含多个皮层区域的活动模式网络
所属层级: 大脑 → 联合皮层 (Level 3)
脑区标签: dlPFC · dACC · PPC · AI (前岛叶)

功能职责 (图3 规则4 — 注意力瓶颈):
  - 任务执行 — 当注意力聚焦外部任务时激活
  - DMN 对立 — TPN 与 DMN 呈跷跷板式互相抑制
  - 外源性注意 — 自下而上的刺激驱动注意
  - 认知努力 — 困难任务时 TPN 激活增强

核心节点:
  - dlPFC (背外侧前额叶) — 执行控制、任务集维护
  - dACC (背侧前扣带回)  — 冲突监测、错误检测
  - PPC (后顶叶皮层)     — 感觉运动整合、空间注意
  - AI (前岛叶)          — 内感受意识、突显检测
  - pre-SMA (前辅助运动区) — 任务集切换、动作选择

网络动态 (跷跷板模型):
  TPN ↑ ⟹ DMN ↓  (任务中: 注意力聚焦外部)
  DMN ↑ ⟹ TPN ↓  (走神/休息: 自我参照思维激活)

  跷跷板切换由突显网络 (Salience Network = AI + dACC) 控制:
    显著刺激出现 → 突显网络激活 → 抑制 DMN → 激活 TPN → 注意力转向任务

与 FPN 的关系:
  FPN (额顶网络) 是 TPN 的"受控"子组件 — 负责内源性自上而下注意
  TPN 是更广泛的任务相关激活模式 — 包含 FPN + dACC + AI + pre-SMA

功能层级:
  突显网络 (Salience)  → 检测显著事件，决定何时切换
       ↓
  TPN (任务正网络)    → 维持任务状态，抑制走神
       ↓
    ├─ FPN (额顶)     → 选择性注意力探照灯 (内源性)
    ├─ DAN (背侧注意)  → 空间注意定向
    └─ VAN (腹侧注意)  → 刺激驱动注意重定向

在 NotMe 中的待实现功能:
  1. 跷跷板动态: TPN_activation = sigmoid(task_demand - mind_wandering_baseline)
     - 高认知负荷 → TPN ↑, DMN ↓
     - 低唤醒/无聊 → TPN ↓, DMN ↑
  2. 突显切换: 接收 cingulate 的冲突/新颖信号 → 决定是否切换 TPN↔DMN
     - salience = max(conflict_signal, novelty_signal)
     - if salience > threshold: switch_to_TPN()
  3. 认知努力调节: 困难任务 → 提升 FPN 探照灯增益
     - effort = expected_free_energy / baseline
     - gain_multiplier = 1 + tanh(effort - 1)
  4. 任务疲劳: 长时间 TPN 激活 → 累积疲劳 → 切换倾向增强
     - fatigue[t+1] = fatigue[t] + tpn_activation * dt - fatigue_decay

神经学病症对应:
  - ADHD: TPN↔DMN 跷跷板不稳定 → 频繁走神/注意力漂移
  - 自闭症 (ASD): TPN↔DMN 切换困难 → 过度专注/重复行为
  - 精神分裂: 突显网络失调 → 内部/外部边界模糊
  - 抑郁症: DMN 过度活跃 (反刍思维) → TPN 难以维持

接口设计 (预留):
  class TaskPositiveNetwork:
      def toggle_tpn_dmn(task_demand, mind_wandering) -> (tpn_act, dmn_act)
      def salience_switch(conflict, novelty, urgency) -> switch_signal
      def cognitive_effort(task_difficulty) -> effort_level
      def fatigue_update(tpn_duration) -> fatigue_signal

参考:
  - Fox, M. D., et al. (2005). The human brain is intrinsically organized into
    dynamic, anticorrelated functional networks. PNAS.
  - Seeley, W. W., et al. (2007). Dissociable intrinsic connectivity networks
    for salience processing and executive control. J. Neuroscience.
  - 图3 规则4: 注意力瓶颈 — TPN vs DMN 跷跷板互相抑制
  - 图4: 同时发生的并行通路 — DMN 在任务中被抑制
"""

import numpy as np
from typing import Optional


class TaskPositiveNetwork:
    """任务正网络 — 执行任务时激活，与 DMN 呈跷跷板对立 (图3 规则4)。

    待实现功能:
      - TPN↔DMN 跷跷板动态
      - 突显切换 (接收 cingulate 信号)
      - 认知努力调节
      - 任务疲劳累积
    """

    def __init__(self):
        # TPN 激活度 (0-1)
        self.tpn_activation: float = 0.5

        # DMN 激活度 (0-1) — 跷跷板对面
        self.dmn_activation: float = 0.5

        # 跷跷板耦合强度
        self.seesaw_strength: float = 0.3

        # 认知努力水平
        self.cognitive_effort: float = 0.5

        # 任务疲劳 (累积项)
        self.task_fatigue: float = 0.0

        # 突显信号 (来自 cingulate / insula)
        self.salience_signal: float = 0.0

    def update_seesaw(
        self,
        task_demand: float = 0.5,
        mind_wandering_baseline: float = 0.3,
        salience: float = 0.0,
    ) -> tuple[float, float]:
        """更新 TPN↔DMN 跷跷板动态 (图3 规则4)。

        TPN 与 DMN 互相抑制:
          TPN[t+1] = TPN[t] + α·(task_demand - TPN[t]) - β·DMN[t]
          DMN[t+1] = DMN[t] + α·(mind_wandering - DMN[t]) - β·TPN[t]

        突显网络可以强制切换:
          if salience > threshold: TPN ↑↑, DMN ↓↓

        Args:
            task_demand: 当前任务需求 (0-1, 高 = 难任务)
            mind_wandering_baseline: 走神基线倾向
            salience: 突显信号 (来自 dACC/insula)

        Returns:
            (tpn_activation, dmn_activation)
        """
        α = 0.1  # 驱动适应率
        β = self.seesaw_strength  # 互相抑制强度

        # 跷跷板核心方程
        tpn_new = self.tpn_activation + α * (task_demand - self.tpn_activation) - β * self.dmn_activation
        dmn_new = self.dmn_activation + α * (mind_wandering_baseline - self.dmn_activation) - β * self.tpn_activation

        # 突显切换: 显著事件 → 强制激活 TPN
        if salience > 0.6:
            tpn_new += 0.5 * salience
            dmn_new -= 0.5 * salience

        # 夹紧到 [0, 1]
        self.tpn_activation = float(np.clip(tpn_new, 0.0, 1.0))
        self.dmn_activation = float(np.clip(dmn_new, 0.0, 1.0))

        # 更新疲劳: TPN 持续时间越长越疲劳
        self.task_fatigue += self.tpn_activation * 0.01 - 0.005  # 缓慢累积 + 衰减
        self.task_fatigue = float(np.clip(self.task_fatigue, 0.0, 1.0))

        return self.tpn_activation, self.dmn_activation

    def compute_effort(self, expected_free_energy: float, baseline: float = 1.0) -> float:
        """根据期望自由能计算认知努力水平。

        困难任务 (高 EFE) → 提升努力 → 增强 FPN 探照灯增益。

        Args:
            expected_free_energy: 期望自由能 G
            baseline: 基线自由能水平

        Returns:
            effort: 认知努力水平 (0-1, 越高越努力)
        """
        ratio = expected_free_energy / (baseline + 1e-8)
        self.cognitive_effort = float(np.tanh(ratio - 1.0) * 0.5 + 0.5)
        return self.cognitive_effort

    def should_switch_to_tpn(self, salience_threshold: float = 0.6) -> bool:
        """突显网络判断是否应切换到 TPN (任务模式)。

        Args:
            salience_threshold: 切换阈值

        Returns:
            should_switch: True = 切换到任务模式 (激活 TPN, 抑制 DMN)
        """
        return self.salience_signal > salience_threshold

    def should_switch_to_dmn(self, fatigue_threshold: float = 0.8) -> bool:
        """疲劳判断是否应切换回 DMN (休息/走神)。

        Args:
            fatigue_threshold: 疲劳阈值

        Returns:
            should_switch: True = 释放任务模式 (激活 DMN)
        """
        return self.task_fatigue > fatigue_threshold or self.tpn_activation < 0.3

    def receive_salience(
        self, conflict_signal: float = 0.0, novelty_signal: float = 0.0, urgency: float = 0.0
    ):
        """接收来自 cingulate/insula 的突显信号。

        突显网络 (dACC + AI) 检测冲突/新颖/紧迫, 驱动 TPN↔DMN 切换。

        Args:
            conflict_signal: 冲突监测信号 (来自 dACC)
            novelty_signal: 新颖性信号 (来自环境/海马)
            urgency: 紧迫性 (来自 amygdala / body state)
        """
        self.salience_signal = float(
            np.clip(max(conflict_signal, novelty_signal) + 0.3 * urgency, 0.0, 1.0)
        )

    def get_state(self) -> dict:
        """获取 TPN 当前状态。"""
        return {
            "tpn_activation": self.tpn_activation,
            "dmn_activation": self.dmn_activation,
            "cognitive_effort": self.cognitive_effort,
            "task_fatigue": self.task_fatigue,
            "salience_signal": self.salience_signal,
        }
