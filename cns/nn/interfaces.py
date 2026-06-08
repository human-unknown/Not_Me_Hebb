"""
interfaces.py — 感知编码器抽象接口 (v7.0 Phase A)

定义 TextEncoder, VisualEncoder, AudioEncoder 的抽象契约,
为 Phase B 的具体实现提供统一的类型签名。

所有编码器遵循相同的模式:
  - encode(x) → np.ndarray: 单样本编码
  - encode_batch(xs) → np.ndarray: 批量编码
  - output_dim: 输出维度 (必须匹配 D=516 中的通道宽度)
"""

from abc import abstractmethod
from typing import List, Optional, Union
import numpy as np

from cns.nn.base import NeuralModule
from cns.nn.config import NNConfig


class TextEncoder(NeuralModule):
    """文本编码器抽象接口.

    输入: 原始文本 (str) 或 预处理token序列
    输出: 语义向量 s[0:64], 64维

    替换: MiniLM-L6-v2 → PCA 64d (外部预训练模型)
    目标: 小型可训练 Transformer, 从 corpus.txt 预训练
    """

    # 输出维度 (固定, 匹配 D=516 布局)
    OUTPUT_DIM: int = 64

    def __init__(self, config: Optional[NNConfig] = None, trainable: bool = True):
        super().__init__(name="text_encoder", config=config, trainable=trainable)

    @abstractmethod
    def encode(self, text: str) -> np.ndarray:
        """编码单条文本.

        Args:
            text: 输入文本 (中文/英文)

        Returns:
            语义向量 (64,) float32
        """
        ...

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """编码批量文本 (默认逐条编码, 子类可 override 为 batch 推理)."""
        results = [self.encode(t) for t in texts]
        return np.stack(results, axis=0).astype(np.float32)

    @property
    def output_dim(self) -> int:
        return self.OUTPUT_DIM


class VisualEncoder(NeuralModule):
    """视觉编码器抽象接口.

    输入: 图像 (np.ndarray, H×W×C uint8 或 float32)
    输出: 视觉特征向量 s[64:372], 308维

    替换: Gabor滤波 V1→V2→V4→IT 手写管线
    目标: 小型 CNN (ResNet-18 前几层) 或 TinyViT

    保留概念:
      - 6条子通路 (M/P/K/Pulvinar/Dorsal/Binding) →
        可以是6个独立小通路或一个网络的6个分头输出
    """

    # M/P/K + IT + SC + Pulvinar + Binding = 96 + 112 + 48 + 16 + 16 + 12 + 8
    OUTPUT_DIM: int = 308
    # 子通路维度分配 (Phase B 实现, Phase A 预留)
    SUBPATHWAYS = {
        "m_pathway": 96,      # 运动/空间 (M-V1, M-V2, MT, MST)
        "p_pathway": 112,     # 形状/细节 (P-V1, P-V2, V4-shape)
        "k_pathway": 48,      # 颜色 (K-V1, K-V2, V4-color)
        "it": 16,             # 物体识别
        "sc": 16,             # 上丘快速通路
        "pulvinar": 12,       # 丘脑枕捷径
        "binding": 8,         # FPN 绑定信号
    }

    def __init__(self, config: Optional[NNConfig] = None, trainable: bool = True):
        super().__init__(name="visual_encoder", config=config, trainable=trainable)

    @abstractmethod
    def encode(self, image: np.ndarray) -> np.ndarray:
        """编码单张图像.

        Args:
            image: 输入图像 (H×W×C), uint8 [0-255] 或 float32 [0-1]

        Returns:
            视觉特征向量 (308,) float32
        """
        ...

    def encode_batch(self, images: np.ndarray) -> np.ndarray:
        """编码批量图像 (B×H×W×C)."""
        results = [self.encode(images[i]) for i in range(len(images))]
        return np.stack(results, axis=0).astype(np.float32)

    @property
    def output_dim(self) -> int:
        return self.OUTPUT_DIM


class AudioEncoder(NeuralModule):
    """听觉编码器抽象接口.

    输入: 音频波形 (np.ndarray, samples) 或 Mel 频谱
    输出: 听觉特征向量 s[372:468], 96维

    替换: Mel频谱→耳蜗核→SOC→IC→MGB→AC 手写管线
    目标: 小型音频模型 (CNN on Mel 或 Wav2Vec2-tiny)

    保留:
      - 双耳定位 (ITD/ILD) → 显式计算 + 神经网络特征拼接
      - 听觉场景分析 → 注意力机制流分离
    """

    # 耳蜗核 32 + SOC 24 + IC 24 + AC 16 = 96
    OUTPUT_DIM: int = 96
    # 子模块维度分配 (Phase B 实现, Phase A 预留)
    SUBMODULES = {
        "cochlear_nucleus": 32,   # tonotopic spectrum
        "soc": 24,                # binaural spatial (ITD + ILD)
        "ic": 24,                 # integrated frequency×space×time
        "auditory_cortex": 16,    # auditory objects/scene
    }

    def __init__(self, config: Optional[NNConfig] = None, trainable: bool = True):
        super().__init__(name="audio_encoder", config=config, trainable=trainable)

    @abstractmethod
    def encode(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """编码单段音频.

        Args:
            audio: 音频波形 (samples,) 或 (samples, channels)
            sample_rate: 采样率 (Hz)

        Returns:
            听觉特征向量 (96,) float32
        """
        ...

    def encode_batch(
        self, audios: List[np.ndarray], sample_rates: Optional[List[int]] = None
    ) -> np.ndarray:
        """编码批量音频."""
        if sample_rates is None:
            sample_rates = [16000] * len(audios)
        results = [self.encode(a, sr) for a, sr in zip(audios, sample_rates)]
        return np.stack(results, axis=0).astype(np.float32)

    @property
    def output_dim(self) -> int:
        return self.OUTPUT_DIM


# ================================================================
# 感知编码器类型别名
# ================================================================

PerceptionEncoder = Union[TextEncoder, VisualEncoder, AudioEncoder]
