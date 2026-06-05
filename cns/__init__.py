"""
cns — 中枢神经系统 (Central Nervous System)  [Level 1]

人脑层级结构最外层：包含大脑与脊髓，指挥全身一切神经活动。

在 NotMe 项目中，CNS 层包含：
- agent.py:     全系统整合 — 组装所有脑区模块，提供统一的 step() 接口
- data_types.py: 全局数据结构 — Theta, BodyVector, FreeEnergy, Cluster 等
- params.py:     默认参数与参数边界
- type_aliases.py: 类型别名
- utils.py:      工具函数

这是唯一不直接对应单一脑区的层级——它是整个系统的"外壳"。
"""

from cns.data_types import (
    D, H, K, A, S_CORE, N_AGENTS,
    Theta, Action, FreeEnergy, AgentBelief, BodyVector,
    Cluster, ACTION_DIRECTIONS, ACTION_NAMES,
)
from cns.params import DEFAULT_THETA_DICT, PARAM_BOUNDS
from cns.type_aliases import *
from cns.utils import exp_moving_average
from cns.agent import Agent

__all__ = [
    'D', 'H', 'K', 'A', 'S_CORE', 'N_AGENTS',
    'Theta', 'Action', 'FreeEnergy', 'AgentBelief', 'BodyVector',
    'Cluster', 'ACTION_DIRECTIONS', 'ACTION_NAMES',
    'DEFAULT_THETA_DICT', 'PARAM_BOUNDS',
    'Agent',
]
