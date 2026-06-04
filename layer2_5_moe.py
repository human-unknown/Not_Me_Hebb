"""
layer2_5_moe.py —— L2.5 混合专家门控 (MoE)
自由能原理智能体 — M1 单智能体生存

单一智能体内多套 Theta 配置的动态加权。
门控根据 F 状态向量决定各模块激活权重。
包含模块疲劳机制 → 自然行为模式轮替。
"""

import numpy as np
from data_types import AgentBelief
from utils import softmax


class MoEGate:
    """混合专家门控

    核心动态:
    1. F 分量驱动调制: F_body↑ → body 相关专家激活
    2. 内部冲突 = G(a) 分歧 → 方差大 → F_cognitive 上升
    3. 模块疲劳: budget 消耗 → 自然轮替
    """

    def __init__(self, n_experts: int = 3):
        self.n_experts = n_experts

        # 各专家的预算 [0, 1]，初始均分
        self.budgets = np.ones(n_experts) / n_experts

        # 各专家的近期表现追踪
        self.performance = np.ones(n_experts)

        # 专家索引含义（M1 中为概念性）:
        # 0 = body-focused (稳态维持)
        # 1 = social-focused (社会交互，M1 少用)
        # 2 = cognitive-focused (探索学习)

    def compute_weights(
        self,
        z: np.ndarray,
        beliefs: AgentBelief,
        step: int,
    ) -> np.ndarray:
        """计算各专家权重

        w_i = softmax(budget_i * performance_i)

        Returns:
            weights: shape (n_experts,), sum = 1
        """
        scores = self.budgets * self.performance
        weights = softmax(scores)
        return weights

    def get_confidence(self, weights: np.ndarray) -> float:
        """从权重计算置信度
        权重分布越均匀 → 多专家一致 → 置信度高
        权重分布越尖锐 → 单一专家主导 → 可能过拟合
        """
        if len(weights) == 1:
            return 1.0
        # 归一化熵
        max_entropy = np.log(self.n_experts)
        entropy = -np.sum(weights * np.log(weights + 1e-8))
        normalized_entropy = entropy / (max_entropy + 1e-8)
        return float(normalized_entropy)

    def update_budgets(self, action: int, outcome: float):
        """根据行动结果更新预算

        outcome 应反映行动好坏（M1 中用 -G_eff 或 reward 近似）
        好结果 → performance 上升 → 该专家预算增加
        差结果 → performance 下降 → 该专家预算减少

        每条轨迹的预算不低于 0.05（保留最低激活可能性）
        """
        # 将 outcome 映射为奖励信号 (G 越小越好 → reward 越大)
        reward = float(np.exp(-abs(outcome) * 0.1))

        # EMA 更新表现
        self.performance = 0.9 * self.performance + 0.1 * reward

        # 更新预算
        self.budgets = np.clip(
            self.budgets + 0.01 * (reward - self.budgets),
            0.05, 1.0
        )
        # 归一化
        self.budgets /= np.sum(self.budgets)

    def fatigue_step(self):
        """疲劳恢复：未激活的专家缓慢恢复预算
        每个时间步调用，推动行为模式自然轮替
        """
        # 所有专家预算向均分靠拢（轻微回弹）
        target = np.ones(self.n_experts) / self.n_experts
        self.budgets = 0.99 * self.budgets + 0.01 * target
        self.budgets /= np.sum(self.budgets)
