"""
config.py — 神经网络配置 (v7.0 Phase A)

NNConfig 管理所有 ML 模块的运行时配置:
  - device (cpu/cuda)
  - dtype (float32/float16)
  - 模型存储路径
  - 训练开关
  - 默认学习率
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NNConfig:
    """神经网络全局配置 — 所有 NeuralModule 共享.

    Attributes:
        device: 计算设备 ('cpu', 'cuda', 'cuda:0', 'mps', 或 'auto')
        dtype: 张量精度 (默认 float32)
        model_dir: 模型权重存档目录
        training_enabled: 全局训练开关 (在线学习控制)
        learning_rate: 默认学习率 (各模块可 override)
        grad_clip: 梯度裁剪阈值 (0 = 不裁剪)
        log_verbose: 是否打印调试信息
    """
    device: str = "auto"
    dtype: str = "float32"
    model_dir: str = ".notme/models"
    training_enabled: bool = True
    learning_rate: float = 1e-3
    grad_clip: float = 1.0
    log_verbose: bool = False

    # 子模块独立学习率 (Phase B-D 使用, Phase A 预留)
    text_lr: Optional[float] = None
    visual_lr: Optional[float] = None
    audio_lr: Optional[float] = None
    lm_lr: Optional[float] = None

    # Phase E: training orchestration
    pretrain_epochs: int = 10           # default pretraining epochs
    online_lr: float = 1e-4             # online fine-tuning learning rate
    lr_scheduler: str = "none"          # "none", "cosine", "step"
    checkpoint_interval: int = 5        # save checkpoint every N epochs (0 = only final)

    # Phase F: integration flags
    nn_enabled: bool = False             # master switch for NN integration
    nn_sensory_enhance: bool = False     # use NN encoders to enhance sensory (expensive)

    # Phase G (v7.7): dual-system speak path
    nn_generate_enabled: bool = False    # enable NN generator in speak() pipeline
    nn_generate_weight: float = 0.3      # NN blend weight [0,1] (0=Hebb, 0.5=mix, 1=NN)

    def effective_lr(self, module_lr: Optional[float] = None) -> float:
        """返回模块的有效学习率, 模块级 > 全局."""
        return module_lr if module_lr is not None else self.learning_rate


# 默认配置实例 (可在 Agent.__init__ 中覆盖)
DEFAULT_NN_CONFIG = NNConfig()
