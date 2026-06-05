"""
image_encoder.py —— 单图像 Gabor 视觉编码器
自由能原理智能体

管线 (与 stage2_crossmodal.py 完全一致):
  图像 (H×W×3 RGB) → GaborFilterBank (image_size=128, grid=4)
    → V1[0:96] + V2[0:64] + V4[0:64] + Color[0:42]
    → build_crossmodal_sensory() → D-dim 感知向量

用途: 为 main_dialogue.py 提供"看到图片"能力。
"""

import numpy as np
from PIL import Image
from typing import Optional

from data_types import D
from layer0_visual import GaborFilterBank

# 与 stage2_crossmodal.py 保持一致的布局常量
TEXT_WIDTH  = 64
V1_WIDTH    = 96
V2_WIDTH    = 64
V4_WIDTH    = 64
COLOR_WIDTH = 42

TEXT_START,  TEXT_END  = 0,   64
V1_START,    V1_END    = 64,  64 + 96
V2_START,    V2_END    = 160, 160 + 64
V4_START,    V4_END    = 224, 224 + 64
COLOR_START, COLOR_END = 288, 288 + 42


class ImageEncoder:
    """单图像 Gabor 视觉编码器。

    GaborFilterBank 是全局共享的 (无状态)——多次编码之间
    只累积 Hebb gain 统计，不影响正确性。
    """

    def __init__(self, image_size: int = 128):
        self.image_size = image_size
        self._gabor: Optional[GaborFilterBank] = None

    @property
    def gabor(self) -> GaborFilterBank:
        if self._gabor is None:
            self._gabor = GaborFilterBank(image_size=self.image_size, grid_size=4)
        return self._gabor

    def encode_from_path(self, image_path: str) -> dict:
        """从文件路径编码图像。

        Returns:
            dict with keys: v1 (96d), v2 (64d), v4 (64d), color (42d)
        """
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img, dtype=np.uint8)
        return self.encode(img_np)

    def encode(self, image: np.ndarray) -> dict:
        """编码单张图像 (uint8 H×W×3 或 float [0,255])。

        Returns:
            dict with keys: v1 (96d), v2 (64d), v4 (64d), color (42d)
        """
        if image.dtype != np.uint8:
            img_np = np.clip(image, 0, 255).astype(np.uint8)
        else:
            img_np = image

        v1_raw = self.gabor.encode(img_np, learn=False)
        v2_raw = self.gabor.encode_v2(img_np)
        v4_raw = self.gabor.encode_v4(img_np)
        color_raw = self.gabor.encode_color(img_np)

        return {
            'v1':    v1_raw[:V1_WIDTH].astype(np.float32),
            'v2':    v2_raw[:V2_WIDTH].astype(np.float32),
            'v4':    v4_raw[:V4_WIDTH].astype(np.float32),
            'color': color_raw[:COLOR_WIDTH].astype(np.float32),
        }


def build_visual_sensory(vis_features: dict,
                         text_emb: np.ndarray = None,
                         normalize_channels: bool = True) -> np.ndarray:
    """构建包含视觉特征的感知向量 (与 stage2_crossmodal 相同布局)。

    s[0:64]    = text embedding (optional, 默认 zeros)
    s[64:160]  = V1 Gabor
    s[160:224] = V2 Gabor
    s[224:288] = V4 Gabor
    s[288:330] = Color opponent

    normalize_channels: 若 True，每个通道 L2 归一化，防止 V4 主导。
                        V4 原始范数 ~1.0，V1/V2/Color ~0.01，
                        不归一化时所有图像 cosine > 0.93 (无区分力)。
    """
    s = np.zeros(D, dtype=np.float32)

    if text_emb is not None:
        flen = min(len(text_emb), TEXT_WIDTH)
        s[TEXT_START:TEXT_START + flen] = text_emb[:flen]

    v1 = vis_features.get('v1')
    if v1 is not None:
        flen = min(len(v1), V1_WIDTH)
        vec = v1[:flen].copy()
        if normalize_channels:
            n = np.linalg.norm(vec)
            if n > 1e-8:
                vec = vec / n
        s[V1_START:V1_START + flen] = vec

    v2 = vis_features.get('v2')
    if v2 is not None:
        flen = min(len(v2), V2_WIDTH)
        vec = v2[:flen].copy()
        if normalize_channels:
            n = np.linalg.norm(vec)
            if n > 1e-8:
                vec = vec / n
        s[V2_START:V2_START + flen] = vec

    v4 = vis_features.get('v4')
    if v4 is not None:
        flen = min(len(v4), V4_WIDTH)
        vec = v4[:flen].copy()
        if normalize_channels:
            n = np.linalg.norm(vec)
            if n > 1e-8:
                vec = vec / n
        s[V4_START:V4_START + flen] = vec

    color = vis_features.get('color')
    if color is not None:
        flen = min(len(color), COLOR_WIDTH)
        vec = color[:flen].copy()
        if normalize_channels:
            n = np.linalg.norm(vec)
            if n > 1e-8:
                vec = vec / n
        s[COLOR_START:COLOR_START + flen] = vec

    return s


def make_visual_mask() -> np.ndarray:
    """视觉通道 mask: s[64:330]=True, 其余=False。

    用于 masked recall — 以纯视觉查询检索跨模态集群。
    """
    mask = np.zeros(D, dtype=bool)
    mask[64:330] = True
    return mask


def make_text_mask() -> np.ndarray:
    """文本通道 mask: s[0:64]=True, 其余=False。"""
    mask = np.zeros(D, dtype=bool)
    mask[0:64] = True
    return mask
