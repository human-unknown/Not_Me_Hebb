"""
data_types.py — 向后兼容 re-export shim (v4.1)
实际代码位置: cns/data_types.py
"""

from cns.data_types import (
    # 全局常量
    D, H, K, A, S_CORE, N_AGENTS,
    # 核心数据类型
    Theta, SensoryVector, Cluster, FreeEnergy, Action, AgentBelief,
    SeedPackage, BodyVector,
    # 动作定义
    ACTION_NAMES, ACTION_DIRECTIONS,
    # 验证函数
    validate_theta,
)
