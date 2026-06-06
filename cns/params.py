"""
params.py —— 默认参数常量
自由能原理智能体 — M1 单智能体生存
"""

# 默认 Theta 参数字典（23 个）
DEFAULT_THETA_DICT = {
    # L0 (6): 生成模型
    'sigma_z': 0.1,
    'sigma_x': 1.0,
    'decay_rate': 0.01,
    'cluster_threshold': 0.85,
    'learn_rate_l0': 0.05,
    'pe_lr_scale': 0.0,
    # L1 (9): 自由能权重
    'w_body': 1.0,
    'w_social': 1.0,
    'w_cognitive': 1.0,
    'eta_valence': 0.5,
    'eta_arousal': 0.5,
    'habituation_tau': 10.0,
    'negativity_bias': 1.5,
    'w_accuracy': 0.5,
    'w_F_signal': 0.1,
    # L2 (5): 策略推理
    'gamma': 0.95,
    'exploration_bonus': 0.1,
    'temperature': 1.0,
    'n_policy_samples': 16,
    'urgency_weight': 0.3,
    # L3 (4): 元学习
    'meta_lr': 0.01,
    'grad_epsilon': 0.001,
    'plasticity_decay': 0.999,
    'critical_window': 1000,
    # L4 (6): v6.0 记忆系统 (语义 + 程序性)
    'semantic_threshold': 0.45,
    'semantic_learn_rate': 0.01,
    'semantic_decay_rate': 0.003,
    'habit_threshold': 0.3,
    'habit_automation_rate': 0.05,
    'd1_d2_balance': 0.5,
}

# 参数边界 [min, max] —— 为 M4 参数扫描预留
PARAM_BOUNDS = {
    'sigma_z': (0.1, 5.0),
    'sigma_x': (0.1, 5.0),
    'decay_rate': (0.001, 0.1),
    'cluster_threshold': (0.5, 0.99),
    'learn_rate_l0': (0.001, 0.5),
    'pe_lr_scale': (0.0, 10.0),
    'w_body': (0.1, 5.0),
    'w_social': (0.1, 5.0),
    'w_cognitive': (0.1, 5.0),
    'eta_valence': (0.1, 2.0),
    'eta_arousal': (0.1, 2.0),
    'habituation_tau': (1.0, 100.0),
    'negativity_bias': (0.5, 5.0),
    'w_accuracy': (0.05, 2.0),
    'w_F_signal': (0.01, 1.0),
    'gamma': (0.1, 0.99),
    'exploration_bonus': (0.01, 1.0),
    'temperature': (0.1, 10.0),
    'n_policy_samples': (4, 64),
    'urgency_weight': (0.05, 1.0),
    'meta_lr': (0.001, 0.1),
    'grad_epsilon': (0.0001, 0.01),
    'plasticity_decay': (0.99, 0.9999),
    'critical_window': (100, 5000),
    # v6.0: 语义 + 程序性参数边界
    'semantic_threshold': (0.2, 0.8),
    'semantic_learn_rate': (0.001, 0.1),
    'semantic_decay_rate': (0.0001, 0.01),
    'habit_threshold': (0.05, 0.8),
    'habit_automation_rate': (0.01, 0.2),
    'd1_d2_balance': (0.1, 0.9),
}
