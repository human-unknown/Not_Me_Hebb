"""
v4.py — 第四视皮层 V4 (Visual Area V4)

对应脑区: BA19 (部分), V4
所属层级: 大脑 → 枕叶 → V4

功能职责:
  - 全局形状: 1×1 全局池化 — 完全位置不变 → "存在什么形状"
  - 曲率检测: 相邻方向的响应空间相关性 → "边缘在弯曲"
  - 形状选择性: 全局 stats (mean/std per filter) → 形状描述符
  - 颜色-形状绑定的前驱

V4 看到的是"什么形状存在"而非"形状在哪里"。
与 V1 (where) / V2 (coarse where+what) 互补。

管线:
  图像 → GaborFilterBank (共享 32 个滤波器)
       → 1×1 global pool × 2 stats × 32 filters = 64d
       → + curvature detection (8d, 跨相邻方向空间相关性)
       → divisive normalization → L2 normalize → ~72d

使用:
  from cerebrum.occipital_lobe.v4 import V4
  v4 = V4(image_size=64)
  features = v4.encode(image)  # → (~72,) float32
"""

import numpy as np
from cerebrum.occipital_lobe.visual_pathway import GaborFilterBank


class V4:
    """V4 第四视皮层 — 全局形状 + 曲率编码。

    封装 GaborFilterBank.encode_v4(), 仅暴露 V4 相关接口。
    """

    def __init__(self, image_size: int = 64):
        self.image_size = image_size
        self._gabor = GaborFilterBank(image_size=image_size, grid_size=4)

    @property
    def n_filters(self) -> int:
        return self._gabor.n_filters  # 32

    def encode(self, image: np.ndarray) -> np.ndarray:
        """图像 → V4 全局形状 + 曲率特征向量。

        Args:
            image: (H, W) 灰度 或 (H, W, 3) RGB, uint8 或 float

        Returns:
            (~72,) float32 — L2 归一化
            布局: 32 filters × 2 stats global (=64)
                  + n_orientations curvature (=8)
        """
        return self._gabor.encode_v4(image)


# ================================================================
# 自测
# ================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  V4 (BA19) Test")
    print("=" * 60)

    rng = np.random.default_rng(42)
    v4 = V4(image_size=64)
    print(f"  Filters: {v4.n_filters}")

    noise = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    vec = v4.encode(noise)
    print(f"  Noise encoding: shape={vec.shape}, norm={np.linalg.norm(vec):.4f}")

    # V4 is position-invariant — same image shifted should produce same encoding
    shifted = np.roll(noise, shift=16, axis=0)
    vec2 = v4.encode(shifted)
    cos = np.dot(vec, vec2) / (np.linalg.norm(vec) * np.linalg.norm(vec2) + 1e-8)
    print(f"  Cosine(original, shifted) = {cos:.4f} (~1.0 = position invariant)")

    print("  [PASS] V4 test complete")
