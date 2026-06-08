"""
cns.nn — 神经网络支撑层 (v7.0 Phase A)

提供 PyTorch 基础设施，使感知层可以运行神经网络。

模块:
  - config.py:     NNConfig — 全局配置 (device, dtype, model_dir, training开关)
  - base.py:       NeuralModule — 所有ML模块的抽象基类
  - bridge.py:     numpy↔tensor 桥接 + 自动设备检测
  - interfaces.py: TextEncoder / VisualEncoder / AudioEncoder 抽象接口
"""

from cns.nn.config import NNConfig, DEFAULT_NN_CONFIG
from cns.nn.base import NeuralModule
from cns.nn.bridge import (
    get_device,
    numpy_to_torch,
    torch_to_numpy,
    ensure_numpy,
    ensure_tensor,
    is_torch_available,
)
from cns.nn.interfaces import (
    TextEncoder,
    VisualEncoder,
    AudioEncoder,
    PerceptionEncoder,
)

__all__ = [
    # Config
    "NNConfig",
    "DEFAULT_NN_CONFIG",
    # Base
    "NeuralModule",
    # Bridge
    "get_device",
    "numpy_to_torch",
    "torch_to_numpy",
    "ensure_numpy",
    "ensure_tensor",
    "is_torch_available",
    # Interfaces
    "TextEncoder",
    "VisualEncoder",
    "AudioEncoder",
    "PerceptionEncoder",
]
