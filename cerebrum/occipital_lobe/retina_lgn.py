"""
retina_lgn.py —— 视网膜→LGN 图像编码器 (v5.1: V5 感知布局)
自由能原理智能体

v5.1 更新:
  - build_visual_sensory() 输出 D=372 V5 布局 (M/P/K × 脑区)
  - make_visual_mask() 覆盖 V5 视觉范围 s[64:372]
  - ImageEncoder 保留用于图像加载, M/P/K raw 输出供下游使用
  - 全视觉管线推荐使用 VisualHierarchy.process()

管线:
  图像 (H×W×3 RGB) → GaborFilterBank (image_size=128, grid=4)
    → M_raw[1024] / P_raw[1024] / K_raw[1024]  (视网膜分型输出)
    → build_visual_sensory() → D-dim (372) V5 感知向量
"""

import numpy as np
from PIL import Image
from typing import Optional

from cns.data_types import (
    D,
    TEXT_V5_WIDTH, TEXT_V5_START,
    M_V1_WIDTH, M_V2_WIDTH, MT_WIDTH, MST_WIDTH,
    M_V1_START, M_V1_END, M_V2_START, M_V2_END,
    MT_START, MT_END, MST_START, MST_END,
    P_V1_WIDTH, P_V2_WIDTH, V4_SHAPE_WIDTH,
    P_V1_START, P_V1_END, P_V2_START, P_V2_END,
    V4_SHAPE_START, V4_SHAPE_END,
    K_V1_WIDTH, K_V2_WIDTH, V4_COLOR_WIDTH,
    K_V1_START, K_V1_END, K_V2_START, K_V2_END,
    V4_COLOR_START, V4_COLOR_END,
    IT_WIDTH, IT_START, IT_END,
    SC_WIDTH, SC_START, SC_END,
    PULVINAR_WIDTH, PULVINAR_START, PULVINAR_END,
    BINDING_WIDTH, BINDING_START, BINDING_END,
)
from cerebrum.occipital_lobe.visual_pathway import GaborFilterBank


# ---- v5.0 compatible aliases (ImageEncoder legacy keys use these widths) ----
# Old V1 = M_V1 + P_V1 + K_V1 (non-contiguous in V5, but summed width for truncation)
_V1_LEGACY_WIDTH = M_V1_WIDTH + P_V1_WIDTH + K_V1_WIDTH   # 96
_V2_LEGACY_WIDTH = M_V2_WIDTH + P_V2_WIDTH + K_V2_WIDTH   # 64
_V4_LEGACY_WIDTH = V4_SHAPE_WIDTH + V4_COLOR_WIDTH         # 48
_COLOR_LEGACY_WIDTH = V4_COLOR_WIDTH                        # 16


class ImageEncoder:
    """单图像 Gabor 视觉编码器 (v5.1: 推荐用 VisualHierarchy 替代).

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
        """从文件路径编码图像.

        Returns:
            dict with keys: M (1024d), P (1024d), K (1024d),
                 v1, v2, v4, color (legacy, deprecated)
        """
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img, dtype=np.uint8)
        return self.encode(img_np)

    def encode(self, image: np.ndarray) -> dict:
        """编码单张图像 (uint8 H×W×3 或 float [0,255]).

        Returns:
            dict with keys:
                M (1024d), P (1024d), K (1024d) — v5.0 M/P/K pathway raw outputs
                v1, v2, v4, color — legacy (deprecated, for backward compat)
        """
        if image.dtype != np.uint8:
            img_np = np.clip(image, 0, 255).astype(np.uint8)
        else:
            img_np = image

        # v5.0: M/P/K retinal ganglion cell-type outputs
        M_raw = self.gabor.encode_M(img_np)
        P_raw = self.gabor.encode_P(img_np)
        K_raw = self.gabor.encode_K(img_np)

        # Legacy encodings (preserved for backward compatibility)
        v1_raw = self.gabor.encode(img_np, learn=False)
        v2_raw = self.gabor.encode_v2(img_np)
        v4_raw = self.gabor.encode_v4(img_np)
        color_raw = self.gabor.encode_color(img_np)

        return {
            'v1':    v1_raw[:_V1_LEGACY_WIDTH].astype(np.float32),
            'v2':    v2_raw[:_V2_LEGACY_WIDTH].astype(np.float32),
            'v4':    v4_raw[:_V4_LEGACY_WIDTH].astype(np.float32),
            'color': color_raw[:_COLOR_LEGACY_WIDTH].astype(np.float32),
            # v5.0 M/P/K channel outputs (1024d raw each)
            'M':     M_raw.astype(np.float32),
            'P':     P_raw.astype(np.float32),
            'K':     K_raw.astype(np.float32),
        }


def _pool_to(raw: np.ndarray, target_len: int, offset: int = 0) -> np.ndarray:
    """从 raw 向量中取切片 + 零填充到 target_len."""
    segment = raw[offset:offset + target_len]
    if len(segment) < target_len:
        out = np.zeros(target_len, dtype=np.float32)
        out[:len(segment)] = segment
        return out
    return segment.astype(np.float32)


def _place_segment(s: np.ndarray, start: int, vec: np.ndarray,
                   normalize: bool = True):
    """Place a normalized segment into sensory vector at start index."""
    v = vec.copy()
    if normalize:
        n = np.linalg.norm(v)
        if n > 1e-8:
            v = v / n
    flen = min(len(v), len(s) - start)
    s[start:start + flen] = v[:flen]


def build_visual_sensory(vis_features: dict,
                         text_emb: np.ndarray = None,
                         normalize_channels: bool = True) -> np.ndarray:
    """构建包含视觉特征的感知向量 (v5.1 V5 布局: D=372).

    M/P/K 三通路 × 脑区层级:
      s[0:64]     = text embedding
      s[64:96]    = M_V1 (32d)   — M通路 V1 层状输出
      s[96:112]   = M_V2 (16d)   — M通路 V2 粗条纹
      s[112:144]  = MT (32d)     — 中颞区方向能量
      s[144:160]  = MST (16d)    — 内上颞区光流模式
      s[160:208]  = P_V1 (48d)   — P通路 V1 层状输出
      s[208:240]  = P_V2 (32d)   — P通路 V2 苍白条纹
      s[240:272]  = V4_shape(32d)— V4 形状编码
      s[272:288]  = K_V1 (16d)   — K通路 V1 斑块输出
      s[288:304]  = K_V2 (16d)   — K通路 V2 细条纹
      s[304:320]  = V4_color(16d)— V4 颜色编码
      s[320:336]  = IT (16d)     — 下颞区物体编码
      s[336:352]  = SC (16d)     — 上丘显著性图
      s[352:364]  = Pulvinar(12d)— 丘脑枕快速通路
      s[364:372]  = Binding (8d) — FPN 特征绑定

    Args:
        vis_features: ImageEncoder.encode() 的输出 dict
        text_emb: 文本嵌入 (可选, 64d)
        normalize_channels: 逐段 L2 归一化

    Returns:
        D-dim (372) 感知向量
    """
    s = np.zeros(D, dtype=np.float32)

    if text_emb is not None:
        flen = min(len(text_emb), TEXT_V5_WIDTH)
        s[TEXT_V5_START:TEXT_V5_START + flen] = text_emb[:flen]

    # ---- M 通路 (运动/空间): 从 M_raw 抽取各段 ----
    M_raw = vis_features.get('M')
    if M_raw is not None:
        _place_segment(s, M_V1_START, _pool_to(M_raw, M_V1_WIDTH, 0),
                       normalize_channels)
        _place_segment(s, M_V2_START, _pool_to(M_raw, M_V2_WIDTH, 32),
                       normalize_channels)
        _place_segment(s, MT_START, _pool_to(M_raw, MT_WIDTH, 48),
                       normalize_channels)
        _place_segment(s, MST_START, _pool_to(M_raw, MST_WIDTH, 80),
                       normalize_channels)

    # ---- P 通路 (形状/细节): 从 P_raw 抽取各段 ----
    P_raw = vis_features.get('P')
    if P_raw is not None:
        _place_segment(s, P_V1_START, _pool_to(P_raw, P_V1_WIDTH, 0),
                       normalize_channels)
        _place_segment(s, P_V2_START, _pool_to(P_raw, P_V2_WIDTH, 48),
                       normalize_channels)
        _place_segment(s, V4_SHAPE_START, _pool_to(P_raw, V4_SHAPE_WIDTH, 80),
                       normalize_channels)

    # ---- K 通路 (颜色): 从 K_raw 抽取各段 ----
    K_raw = vis_features.get('K')
    if K_raw is not None:
        _place_segment(s, K_V1_START, _pool_to(K_raw, K_V1_WIDTH, 0),
                       normalize_channels)
        _place_segment(s, K_V2_START, _pool_to(K_raw, K_V2_WIDTH, 16),
                       normalize_channels)
        _place_segment(s, V4_COLOR_START, _pool_to(K_raw, V4_COLOR_WIDTH, 32),
                       normalize_channels)

    # ---- IT/SC/Pulvinar/Binding: 零初始化 (由 VisualHierarchy 补充) ----
    # 在 build_visual_sensory 快速路径中这些段保持为零;
    # 当使用 VisualHierarchy.process() 时会被真实值覆盖.

    return s


def make_visual_mask() -> np.ndarray:
    """视觉通道 mask: s[64:372]=True, 其余=False (v5.1 V5 范围).

    用于 masked recall — 以纯视觉查询检索跨模态集群.
    """
    mask = np.zeros(D, dtype=bool)
    mask[M_V1_START:BINDING_END] = True
    return mask


def make_text_mask() -> np.ndarray:
    """文本通道 mask: s[0:64]=True, 其余=False."""
    mask = np.zeros(D, dtype=bool)
    mask[TEXT_V5_START:TEXT_V5_START + TEXT_V5_WIDTH] = True
    return mask
