"""
base.py — NeuralModule 基类 (v7.0 Phase A)

所有 ML 模块的统一抽象接口:
  - forward:   前向推理 (sensory → features)
  - train_step: 单步训练 (在线学习)
  - save/load:  模型持久化 (PyTorch .pt 格式)
  - to_numpy:   确保输出为 numpy 数组 (与现有 Agent 兼容)

设计约束:
  - 输入输出均为 numpy 数组 (与 Agent.step() 兼容)
  - 内部可用 PyTorch 张量加速
  - 不训练时 zero_grad + no_grad 确保效率
"""

from abc import ABC, abstractmethod
import os
import logging
from typing import Optional, Dict, Any, Union
import numpy as np

from cns.nn.config import NNConfig, DEFAULT_NN_CONFIG
from cns.nn.bridge import (
    _get_torch, get_device, numpy_to_torch, torch_to_numpy, ensure_numpy,
)

logger = logging.getLogger(__name__)


class NeuralModule(ABC):
    """神经网络模块抽象基类.

    每个子类实现一个特定的 ML 功能 (编码器、语言模型等).
    统一接口使得 Agent 可以透明地替换手写特征提取器。

    Usage:
        class MyEncoder(NeuralModule):
            def _build_network(self):
                self._net = nn.Linear(64, 64)

            def forward(self, x):
                t = self.to_tensor(x)
                return self.to_numpy(self._net(t))
    """

    def __init__(
        self,
        name: str,
        config: Optional[NNConfig] = None,
        trainable: bool = True,
    ):
        """初始化神经模块.

        Args:
            name: 模块名 (用于存档文件命名)
            config: 全局 NN 配置 (None 则使用默认)
            trainable: 是否可训练 (False = 冻结权重)
        """
        self.name = name
        self.config = config or DEFAULT_NN_CONFIG
        self.trainable = trainable
        self._train_mode: bool = False

        # 解析设备
        self._device_str = get_device(self.config.device)
        self._torch = _get_torch()
        self._device = self._torch.device(self._device_str)

        # 子类提供的网络层
        self._net: Optional[Any] = None
        self._optimizer: Optional[Any] = None

        # 元信息
        self._step_count: int = 0
        self._total_trained: int = 0
        self._version: str = "7.0"

        # 构建网络
        self._build_network()

    # ================================================================
    # 抽象接口 (子类必须实现)
    # ================================================================

    @abstractmethod
    def _build_network(self):
        """构建内部神经网络结构 (子类必须实现).

        将网络赋值给 self._net (torch.nn.Module).
        可训练的层应正确注册为 nn.Module 属性或放入 nn.ModuleList.
        """
        ...

    # ================================================================
    # 核心接口
    # ================================================================

    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向推理 — numpy in, numpy out.

        Args:
            x: 输入 (np.ndarray, shape depends on subclass)

        Returns:
            输出 (np.ndarray, float32)
        """
        if self._net is None:
            return x

        with self._torch.no_grad():
            tensor_x = ensure_numpy(x)  # ensure numpy first
            tensor_x = numpy_to_torch(tensor_x, device=self._device_str)
            result = self._forward_impl(tensor_x)
            if isinstance(result, self._torch.Tensor):
                return torch_to_numpy(result)
            return result

    @abstractmethod
    def _forward_impl(self, x: "torch.Tensor") -> "torch.Tensor":
        """前向推理实现 (子类必须实现) — tensor in, tensor out."""
        ...

    def train_step(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """单步训练 (在线学习).

        Args:
            batch: 训练数据字典, 至少包含 'input' 键,
                  可选 'target', 'mask' 等.

        Returns:
            损失字典 {'loss': float, ...}
        """
        if self._net is None or not self.trainable:
            return {"loss": 0.0}

        if not self.config.training_enabled:
            return {"loss": 0.0}

        self._train_mode = True
        self._net.train()

        # 准备数据
        tensor_batch = {
            k: numpy_to_torch(v, device=self._device_str)
            for k, v in batch.items()
            if isinstance(v, np.ndarray)
        }

        # 子类实现
        losses = self._train_step_impl(tensor_batch)

        self._step_count += 1
        self._total_trained += 1

        self._train_mode = False
        return losses

    def _train_step_impl(
        self, batch: Dict[str, "torch.Tensor"]
    ) -> Dict[str, float]:
        """训练步骤实现 — 子类可 override, 默认不做任何事."""
        return {"loss": 0.0}

    # ================================================================
    # 持久化接口
    # ================================================================

    def save(self, path: Optional[str] = None) -> str:
        """保存模型权重到 .pt 文件.

        Args:
            path: 保存路径 (None = 自动生成 model_dir/name.pt)

        Returns:
            实际保存路径
        """
        if path is None:
            os.makedirs(self.config.model_dir, exist_ok=True)
            path = os.path.join(self.config.model_dir, f"{self.name}.pt")

        self._torch.save(
            {
                "name": self.name,
                "version": self._version,
                "step_count": self._step_count,
                "total_trained": self._total_trained,
                "trainable": self.trainable,
                "state_dict": self._net.state_dict() if self._net else {},
                "optimizer": (
                    self._optimizer.state_dict() if self._optimizer else {}
                ),
            },
            path,
        )
        logger.debug(f"[{self.name}] Saved to {path}")
        return path

    def load(self, path: Optional[str] = None) -> bool:
        """加载模型权重.

        Args:
            path: 加载路径 (None = 自动使用 model_dir/name.pt)

        Returns:
            是否加载成功
        """
        if path is None:
            path = os.path.join(self.config.model_dir, f"{self.name}.pt")

        if not os.path.exists(path):
            logger.warning(f"[{self.name}] Model file not found: {path}")
            return False

        checkpoint = self._torch.load(
            path, map_location=self._device_str, weights_only=False
        )

        if self._net and "state_dict" in checkpoint:
            self._net.load_state_dict(checkpoint["state_dict"])

        if self._optimizer and "optimizer" in checkpoint:
            self._optimizer.load_state_dict(checkpoint["optimizer"])

        self._step_count = checkpoint.get("step_count", 0)
        self._total_trained = checkpoint.get("total_trained", 0)
        logger.debug(f"[{self.name}] Loaded from {path}")
        return True

    # ================================================================
    # 工具方法
    # ================================================================

    def to_tensor(self, x: Union[np.ndarray, list]) -> "torch.Tensor":
        """numpy → tensor 快捷方法."""
        return numpy_to_torch(
            ensure_numpy(x) if not isinstance(x, np.ndarray) else x,
            device=self._device_str,
        )

    def to_numpy(self, tensor: "torch.Tensor") -> np.ndarray:
        """tensor → numpy 快捷方法."""
        return torch_to_numpy(tensor)

    def train(self, mode: bool = True):
        """设置训练/评估模式."""
        self._train_mode = mode
        if self._net:
            self._net.train(mode)

    def eval(self):
        """设置评估模式."""
        self.train(False)

    @property
    def device(self) -> str:
        return self._device_str

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def total_trained(self) -> int:
        return self._total_trained

    @property
    def has_network(self) -> bool:
        """网络是否已构建."""
        return self._net is not None

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={self.name!r}, "
            f"device={self._device_str}, trainable={self.trainable}, "
            f"steps={self._step_count})"
        )
