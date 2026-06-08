"""
bridge.py — NumPy ↔ PyTorch 桥接层 (v7.0 Phase A)

提供无损的 numpy↔tensor 转换和自动设备检测,
使感知层可以在 numpy (现有) 和 tensor (神经网络) 之间无缝切换。

设计原则:
  - 所有函数接受 np.ndarray 返回 np.ndarray (对外透明)
  - 内部转为 tensor 计算，结果转回 numpy
  - 不强制所有代码使用 tensor — 保持向后兼容
"""

import numpy as np
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)

# 惰性导入 PyTorch — 允许在不安装 torch 时优雅降级
_torch = None


def _get_torch():
    """惰性获取 torch 模块, 首次调用时尝试导入."""
    global _torch
    if _torch is None:
        try:
            import torch as _t
            _torch = _t
        except ImportError:
            raise ImportError(
                "PyTorch is required for cns.nn. "
                "Install it with: pip install torch"
            )
    return _torch


def get_device(device_str: str = "auto") -> str:
    """解析设备字符串, 返回实际可用设备.

    Args:
        device_str: 'auto', 'cpu', 'cuda', 'cuda:0', 'mps'

    Returns:
        实际设备字符串 (如 'cpu', 'cuda:0')
    """
    torch = _get_torch()

    if device_str == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    # 验证指定设备可用
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        logger.warning(f"CUDA requested but not available, falling back to CPU")
        return "cpu"
    if device_str == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        logger.warning(f"MPS requested but not available, falling back to CPU")
        return "cpu"

    return device_str


def numpy_to_torch(
    arr: np.ndarray,
    device: str = "cpu",
    dtype: Optional[str] = None,
) -> "torch.Tensor":
    """NumPy 数组 → PyTorch 张量.

    Args:
        arr: 输入 numpy 数组
        device: 目标设备
        dtype: 目标精度 (None=保持原精度)

    Returns:
        PyTorch tensor (requires_grad=False)
    """
    torch = _get_torch()

    tensor = torch.from_numpy(np.asarray(arr, dtype=np.float32 if dtype is None else None))

    # 类型映射
    if dtype is not None:
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "float64": torch.float64,
            "int32": torch.int32,
            "int64": torch.int64,
        }
        tensor = tensor.to(dtype=dtype_map.get(dtype, torch.float32))

    resolved_device = get_device(device)
    return tensor.to(resolved_device)


def torch_to_numpy(tensor: "torch.Tensor") -> np.ndarray:
    """PyTorch 张量 → NumPy 数组.

    Args:
        tensor: PyTorch tensor (任意设备)

    Returns:
        numpy array (float32)
    """
    return tensor.detach().cpu().numpy().astype(np.float32)


def ensure_numpy(x: Union[np.ndarray, "torch.Tensor", list]) -> np.ndarray:
    """确保输出为 numpy 数组 (通用转换)."""
    torch = _get_torch()
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)
    if isinstance(x, torch.Tensor):
        return torch_to_numpy(x)
    if isinstance(x, list):
        return np.array(x, dtype=np.float32)
    raise TypeError(f"Cannot convert {type(x)} to numpy array")


def ensure_tensor(
    x: Union[np.ndarray, "torch.Tensor", list],
    device: str = "cpu",
    dtype: Optional[str] = None,
) -> "torch.Tensor":
    """确保输出为 PyTorch 张量 (通用转换)."""
    torch = _get_torch()
    if isinstance(x, torch.Tensor):
        resolved_device = get_device(device)
        return x.to(resolved_device)
    if isinstance(x, np.ndarray):
        return numpy_to_torch(x, device=device, dtype=dtype)
    if isinstance(x, list):
        arr = np.array(x, dtype=np.float32)
        return numpy_to_torch(arr, device=device, dtype=dtype)
    raise TypeError(f"Cannot convert {type(x)} to tensor")


def is_torch_available() -> bool:
    """检查 PyTorch 是否可用 (不触发导入)."""
    global _torch
    if _torch is not None:
        return True
    try:
        import torch
        _torch = torch
        return True
    except ImportError:
        return False
