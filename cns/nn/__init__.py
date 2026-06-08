"""
cns.nn — 神经网络支撑层 (v7.2 Phase C)

提供 PyTorch 基础设施，使感知层和记忆层可以运行神经网络。

模块:
  - config.py:         NNConfig — 全局配置 (device, dtype, model_dir, training开关)
  - base.py:           NeuralModule — 所有ML模块的抽象基类
  - bridge.py:         numpy↔tensor 桥接 + 自动设备检测
  - interfaces.py:     TextEncoder / VisualEncoder / AudioEncoder 抽象接口
  - text_encoder.py:   TrainableTextEncoder (Phase B)
  - visual_encoder.py: TrainableVisualEncoder (Phase B)
  - audio_encoder.py:  TrainableAudioEncoder (Phase B)
  - semantic_store.py: NeuralSemanticStore (Phase C)
  - crossmodal_nn.py:  CrossModalNN (Phase C)
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

# Phase B: concrete encoder implementations
from cns.nn.text_encoder import TrainableTextEncoder
from cns.nn.visual_encoder import TrainableVisualEncoder
from cns.nn.audio_encoder import TrainableAudioEncoder

# Phase C: memory layer
from cns.nn.semantic_store import NeuralSemanticStore
from cns.nn.crossmodal_nn import CrossModalNN

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
    # Phase B: concrete encoders
    "TrainableTextEncoder",
    "TrainableVisualEncoder",
    "TrainableAudioEncoder",
    # Phase C: memory layer
    "NeuralSemanticStore",
    "CrossModalNN",
]
