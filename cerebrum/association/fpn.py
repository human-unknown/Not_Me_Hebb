"""
fpn.py — 额顶网络 (Frontoparietal Network / Central Executive Network)

对应脑区: 额顶网络 (FPN / CEN)
所属层级: 大脑 → 联合皮层 (Level 3)
脑区标签: dlPFC (BA9/46) · PPC (BA7/40) · FEF (BA8)

功能职责 (图3 规则4 — 注意力瓶颈):
  - 选择性注意的"探照灯" — 增强任务相关信号，抑制干扰信号
  - 工作记忆维护 — 在延迟期内保持信息活跃
  - 认知控制 — 目标导向的行为调节
  - 注意力定向 — 内源性 (自上而下) 注意转移

核心节点:
  - dlPFC (背外侧前额叶 BA9/46) — 执行控制、工作记忆中央执行器
  - PPC (后顶叶皮层 BA7/40)     — 空间注意定向、感觉运动整合
  - FEF (额叶眼区 BA8)          — 眼动控制、注意转移
  - IPS (顶内沟)                — 数量、空间表征

关键机制:
  1. 探照灯效应 (Searchlight): dlPFC 发送自上而下的增益信号
     → 增强目标特征在感觉皮层的表征
     → 抑制非目标特征的竞争
  2. 容量限制: 意识带宽 ~50 bits/s (感官输入 ~11M bits/s)
     → FPN 作为瓶颈 — 一次只能深度处理 1 个任务
  3. 与 DMN 的跷跷板: FPN 激活时 DMN 被抑制 (任务中)
     → 由突显网络 (前岛叶 + dACC) 负责切换

神经学病症对应:
  - ADHD: FPN 探照灯不稳定 → 注意力频繁漂移
  - 忽视症 (Neglect): PPC 受损 → 对侧空间忽视
  - 精神分裂: FPN-DMN 跷跷板失调 → 思维插入、被动体验

在 NotMe 中的待实现功能:
  1. 注意力增益调制: 根据当前任务目标调节感知通道增益
     - attended = sensory_input * gain_vector(target_features)
     - gain_vector 由 dlPFC 根据任务上下文计算
  2. 工作记忆延迟期维护: 在行动选择期间保持目标表征
     - recurrent_activation[t+1] = tanh(W·h[t] + keep_gate)
  3. 抗干扰滤波: 抑制与当前目标无关的感觉输入
     - filtered = input * (1 - suppress_mask)
  4. 任务切换: 响应突显网络信号，重配置注意力模板

接口设计 (预留):
  class FrontoparietalNetwork:
      def attention_searchlight(sensory_input, task_goal) -> attended_input
      def maintain_working_memory(goal_state, delay) -> persistent_activity
      def task_switch(new_goal, salience_signal) -> attention_template
      def filter_distractors(inputs, target_mask) -> filtered_inputs

参考:
  - Corbetta, M., & Shulman, G. L. (2002). Control of goal-directed and
    stimulus-driven attention in the brain. Nature Reviews Neuroscience.
  - Duncan, J. (2010). The multiple-demand (MD) system of the primate brain.
    Trends in Cognitive Sciences.
  - 图3 规则4: 注意力瓶颈 — 选择性注意、FPN探照灯
"""

import numpy as np
from typing import Optional, Tuple


class FrontoparietalNetwork:
    """额顶网络 — 选择性注意的探照灯 (图3 规则4)。

    待实现功能:
      - 注意力增益调制 (探照灯效应)
      - 工作记忆延迟期维护
      - 抗干扰滤波
      - DMN↔TPN 跷跷板中的任务执行侧
    """

    def __init__(self, input_dim: int = 330, wm_capacity: int = 4):
        """
        Args:
            input_dim: 感知输入维度 (D=330)
            wm_capacity: 工作记忆容量 (经典 7±2, 但核心瓶颈 ~4 项)
        """
        self.input_dim = input_dim
        self.wm_capacity = wm_capacity

        # 注意力模板: 当前任务目标的特征权重
        self.attention_template: np.ndarray = np.ones(input_dim, dtype=np.float32)

        # 工作记忆槽位
        self.wm_slots: list = []  # 最多 wm_capacity 个活跃项

        # 抑制掩码: 被抑制的特征维度
        self.suppression_mask: np.ndarray = np.zeros(input_dim, dtype=np.float32)

        # TPN 激活度 (0-1, 与 DMN 跷跷板)
        self.tpn_activation: float = 0.5

    def gate_attention(
        self, sensory: np.ndarray, goal_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """注意力门控 — FPN 探照灯增强目标特征。

        图3 规则4: 选择性注意作为"探照灯"。
        增强 goal_mask 对应的特征维度，抑制其余。

        Args:
            sensory: 感知输入 s ∈ R^D
            goal_mask: 任务目标特征权重 (可选, 默认用当前模板)

        Returns:
            attended: 注意力调制后的感知输入
        """
        if goal_mask is None:
            goal_mask = self.attention_template

        # 探照灯: 加权增强目标信号
        gain = 1.0 + goal_mask  # 目标特征增益 > 1
        attended = sensory * gain

        # 侧抑制: 非目标区域被压制
        attended = attended * (1.0 - 0.3 * (1.0 - goal_mask / (goal_mask.max() + 1e-8)))

        return attended.astype(np.float32)

    def update_template(self, task_goal_features: np.ndarray, lr: float = 0.1):
        """更新注意力模板 — EMA 朝向当前任务目标特征。

        Args:
            task_goal_features: 当前任务目标的特征向量
            lr: 模板更新速率
        """
        self.attention_template = (
            1.0 - lr
        ) * self.attention_template + lr * task_goal_features
        self.attention_template = np.clip(self.attention_template, 0.0, 2.0)

    def maintain_wm(self, item: np.ndarray):
        """维护工作记忆 — 在容量限制内保持信息活跃。

        Args:
            item: 工作记忆项目 (任意维度的特征向量)
        """
        if len(self.wm_slots) < self.wm_capacity:
            self.wm_slots.append(item.copy())
        else:
            # 最弱项目被替换
            self.wm_slots.pop(0)
            self.wm_slots.append(item.copy())

    def filter_distractors(
        self, inputs: np.ndarray, target_features: np.ndarray
    ) -> np.ndarray:
        """抗干扰滤波 — 抑制与目标无关的感觉输入。

        Args:
            inputs: 感觉输入
            target_features: 目标特征的掩码 (1=相关, 0=无关)

        Returns:
            filtered: 滤除干扰后的输入
        """
        self.suppression_mask = np.where(target_features > 0.5, 0.0, 0.7)
        filtered = inputs * (1.0 - self.suppression_mask)
        return filtered.astype(np.float32)

    def get_activation(self) -> float:
        """返回 FPN 当前激活度。"""
        return self.tpn_activation

    def set_channel_gains(self, task_type: str = 'default') -> dict:
        """v5.0: 按 M/P/K 通道 + 脑区设置增益权重.

        Args:
            task_type: 当前任务类型
                'motion' → M 通道增益↑ (运动检测优先)
                'shape'  → P 通道增益↑ (形状分析优先)
                'color'  → K 通道增益↑ (颜色处理优先)
                'default' → 均匀增益

        Returns:
            dict with per-area per-channel gain values
        """
        gains = {
            'V1': {'M': 1.0, 'P': 1.0, 'K': 1.0},
            'V2': {'thick': 1.0, 'pale': 1.0, 'thin': 1.0},
            'V4': {'shape': 1.0, 'color': 1.0},
            'MT': {'direction': 1.0},
            'IT': {'object': 1.0},
        }

        if task_type == 'motion':
            gains['V1']['M'] = 1.8
            gains['V2']['thick'] = 1.8
            gains['MT']['direction'] = 2.0
            gains['V1']['P'] = 0.6
            gains['V2']['pale'] = 0.6
        elif task_type == 'shape':
            gains['V1']['P'] = 1.8
            gains['V2']['pale'] = 1.8
            gains['V4']['shape'] = 2.0
            gains['IT']['object'] = 1.5
        elif task_type == 'color':
            gains['V1']['K'] = 1.8
            gains['V2']['thin'] = 1.8
            gains['V4']['color'] = 2.0
        # 'default': all 1.0

        return gains

    def compute_spatial_focus(self, sensory_or_v1: np.ndarray,
                               n_positions: int = 16) -> np.ndarray:
        """v5.0: 从 V1 段信号中提取 FPN 的空间注意力焦点.

        Returns:
            (n_positions,) 空间注意力权重, sum=1
        """
        spatial_focus = np.ones(n_positions, dtype=np.float32)

        # 从 V1 段提取每个网格细胞的响应强度
        if len(sensory_or_v1) >= n_positions * 4:
            for i in range(n_positions):
                offset = i * 4
                if offset + 4 <= len(sensory_or_v1):
                    cell_strength = float(np.mean(np.abs(
                        sensory_or_v1[offset:offset + 4])))
                    spatial_focus[i] = 1.0 + cell_strength * 3.0

        total = float(np.sum(spatial_focus)) + 1e-8
        return spatial_focus / total
