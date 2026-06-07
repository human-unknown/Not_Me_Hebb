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

    def __init__(self, input_dim: int = None, wm_capacity: int = 4):
        """
        Args:
            input_dim: 感知输入维度 (默认 data_types.D, 当前 D_V54=516)
            wm_capacity: 工作记忆容量 (经典 7±2, 但核心瓶颈 ~4 项)
        """
        if input_dim is None:
            from cns.data_types import D as _D
            input_dim = _D
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
        self, sensory: np.ndarray, goal_mask: Optional[np.ndarray] = None,
        alpha_strength: float = 0.4, step_count: int = 0,
    ) -> np.ndarray:
        """注意力门控 — FPN 探照灯增强目标特征 + α 节律功能抑制。

        图3 规则4: 选择性注意作为"探照灯"。
        增强 goal_mask 对应的特征维度，抑制其余。

        v6.3: α 节律 (8-13 Hz) 功能抑制:
          - 注意通道: α抑制低 (高γ, 低α) → 信号增强
          - 非注意通道: α抑制高 (低γ, 高α) → 功能抑制
          - α 在功能上是"主动抑制"——不是缺乏处理，而是主动阻挡

        Args:
            sensory: 感知输入 s ∈ R^D
            goal_mask: 任务目标特征权重 (可选, 默认用当前模板)
            alpha_strength: α 节律对非注意通道的抑制强度 [0, 1]
            step_count: 全局步数 (用于 α 振荡相位)

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

        # ---- v6.3: α 节律功能抑制 ----
        # α 振荡相位 (10 Hz 等价, sin 函数)
        alpha_phase = 0.5 + 0.5 * np.sin(2.0 * np.pi * 10.0 * step_count * 0.03)
        # 注意通道: α 抑制弱 (高γ, 低α)
        alpha_attention = 1.0 - alpha_strength * 0.15 * alpha_phase * (2.0 - goal_mask)
        # 非注意通道: α 抑制强 (低γ, 高α)
        alpha_suppress = 1.0 - alpha_strength * alpha_phase * (1.0 - goal_mask /
                                                                (goal_mask.max() + 1e-8))
        # 综合 α 调制
        alpha_mod = np.where(goal_mask > 0.5, alpha_attention, alpha_suppress)
        attended = attended * np.clip(alpha_mod, 0.1, 1.5)

        return attended.astype(np.float32)

    def alpha_gate_attention(self, sensory: np.ndarray,
                             attention_mask: Optional[np.ndarray] = None,
                             alpha_strength: float = 0.4,
                             step_count: int = 0) -> dict:
        """v6.3: α 节律注意门控 — 独立调用接口.

        对应神经机制:
          - α 节律 (8-13 Hz, 枕区主导) = 功能性抑制
          - 闭眼/放松 → α 增强 (广泛抑制)
          - 注意集中 → α 在非注意区增强, 在注意区减弱
          - 这就是"α 阻断" (alpha blocking) —— 被注意的通道 α 下降

        Args:
            sensory: 感知输入
            attention_mask: 注意力掩码 (1=注意, 0=忽略)
            alpha_strength: α 抑制强度
            step_count: 全局步数

        Returns:
            dict with gated_sensory, alpha_level, suppression_map
        """
        if attention_mask is None:
            attention_mask = self.attention_template

        # α 水平: 基于当前任务参与度
        base_alpha = alpha_strength * (1.5 - self.tpn_activation)
        # α 振荡相位
        alpha_phase = 0.5 + 0.5 * np.sin(2.0 * np.pi * 10.0 * step_count * 0.03)

        # 通道级 α 抑制
        # 注意通道: γ 主导, α 低
        attended_channels = attention_mask > 0.5
        unattended_channels = ~attended_channels

        suppression = np.ones_like(sensory)
        # 非注意通道: α 高 → 功能抑制
        suppression[unattended_channels] = (
            1.0 - base_alpha * alpha_phase)
        # 注意通道: α 低 → 信号畅通 (但仍有微弱调制)
        suppression[attended_channels] = (
            1.0 - base_alpha * 0.1 * alpha_phase)

        suppression = np.clip(suppression, 0.15, 1.5)
        gated = sensory * suppression

        return {
            'gated_sensory': gated.astype(np.float32),
            'alpha_level': float(base_alpha),
            'alpha_phase': float(alpha_phase),
            'suppression_map': suppression,
            'mean_suppression': float(np.mean(suppression)),
            'attended_suppression': float(np.mean(suppression[attended_channels]))
                if np.any(attended_channels) else 1.0,
        }

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

    # ================================================================
    # v6.0: 中央执行器 (Central Executive)
    # ================================================================

    def allocate_attention(self, task_demand: float,
                          wm_load: float = 0.0,
                          novelty: float = 0.0) -> dict:
        """Central executive: allocate limited attentional resources across subsystems.

        High task demand → more resources to current task
        High novelty → redirect resources to new stimulus
        High WM load → reduce new information intake

        Args:
            task_demand: task demand intensity [0, 1]
            wm_load: working memory load [0, 1]
            novelty: novelty signal [0, 1]

        Returns:
            dict: {attn_focus, suppression_level, switch_triggered, switch_cost}
        """
        # Base attention focus = task demand
        attn_focus = task_demand

        # WM load modulation: high load → reduce new info intake
        effective_attn = attn_focus * (1.0 - wm_load * 0.5)

        # Novelty modulation: high novelty → possible attention switch
        switch_triggered = False
        if novelty > 0.6 and task_demand < 0.5:
            effective_attn = novelty * 0.8
            self._task_switch_cost = 0.3
            switch_triggered = True
        else:
            self._task_switch_cost = getattr(self, '_task_switch_cost', 0.0) * 0.9

        # Distractor suppression = f(task focus)
        self._distractor_suppression = 0.2 + 0.6 * task_demand

        return {
            'attn_focus': float(effective_attn),
            'suppression_level': float(self._distractor_suppression),
            'switch_triggered': switch_triggered,
            'switch_cost': float(self._task_switch_cost),
        }

    def filter_distractors_v2(self, sensory: np.ndarray) -> np.ndarray:
        """Suppress sensory input irrelevant to current task goal.

        Uses current attention template and distractor suppression strength.
        Non-target dimensions are attenuated (but not zeroed — preserves
        response to sudden threats).

        Args:
            sensory: (D,) raw sensory input

        Returns:
            (D,) filtered sensory input
        """
        template = self.attention_template.copy()
        level = getattr(self, '_distractor_suppression', 0.3)

        # Build suppression mask: low-template dims are suppressed
        template_range = template.max() - template.min()
        if template_range > 0:
            suppress_mask = ((template - template.min()) / (template_range + 1e-8))
        else:
            suppress_mask = np.ones_like(template)
        suppress_mask = level + (1.0 - level) * suppress_mask

        return (sensory * suppress_mask).astype(np.float32)

    def coordinate_subsystems(self,
                             phonological_load: float = 0.0,
                             visuospatial_load: float = 0.0,
                             episodic_buffer_load: float = 0.0) -> dict:
        """Coordinate WM subsystems — allocate attention across them.

        Phonological loop and visuospatial sketchpad compete for
        attentional resources. The central executive decides priority.

        Returns:
            dict: attention weights for each subsystem
        """
        total_load = (phonological_load + visuospatial_load
                     + episodic_buffer_load)
        if total_load < 0.01:
            return {'phonological': 0.33, 'visuospatial': 0.33,
                   'episodic_buffer': 0.34}

        return {
            'phonological': phonological_load / max(total_load, 0.01),
            'visuospatial': visuospatial_load / max(total_load, 0.01),
            'episodic_buffer': episodic_buffer_load / max(total_load, 0.01),
        }
