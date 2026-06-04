"""
utils.py —— 工具函数
自由能原理智能体 — M1 单智能体生存
"""

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """余弦相似度，带数值保护"""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    denom = norm_a * norm_b + 1e-8
    return float(dot / denom)


def clip_gradient(g: np.ndarray, max_norm: float = 1.0) -> np.ndarray:
    """梯度裁剪，防止梯度爆炸"""
    g_norm = np.linalg.norm(g)
    if g_norm > max_norm:
        return g * (max_norm / g_norm)
    return g


def normalize(v: np.ndarray) -> np.ndarray:
    """向量归一化到单位长度"""
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return v
    return v / norm


def exp_moving_average(new_val: float, old_val: float, alpha: float) -> float:
    """指数移动平均：EMA = alpha * new + (1-alpha) * old"""
    return alpha * new_val + (1 - alpha) * old_val


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Softmax with temperature, numerically stable"""
    x_adj = x / max(temperature, 1e-8)
    x_max = np.max(x_adj)
    exp_x = np.exp(x_adj - x_max)
    return exp_x / (np.sum(exp_x) + 1e-8)


def tanh_clip(x: float) -> float:
    """tanh 映射到 [-1, 1]"""
    return float(np.tanh(x))


def sigmoid(x: float) -> float:
    """sigmoid 映射到 [0, 1]"""
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -50, 50))))


def ensure_non_negative(x: float, eps: float = 1e-8) -> float:
    """确保非负"""
    return max(x, eps)
