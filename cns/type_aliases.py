"""
types.py —— 类型别名
自由能原理智能体 — M1 单智能体生存
"""

import numpy as np

# 向量与矩阵
Vector = np.ndarray    # 1D array
Matrix = np.ndarray    # 2D array

# 复合类型（用于类型标注）
from typing import Union, Optional
from cns.data_types import Cluster

ClusterList = list     # list[Cluster]
