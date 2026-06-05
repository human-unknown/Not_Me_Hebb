"""
v2.py — 次级视皮层 V2 (Secondary Visual Cortex)

对应脑区: BA18 (旁纹状皮层, V2)
所属层级: 大脑 → 枕叶 → V2

功能职责:
  - 粗网格空间: 2×2 网格 (比 V1 更大的感受野, 32×32 per cell)
  - 方向交互: 相邻方向响应乘积 → 角点检测
  - 方向对比度: max(response) - min(response) per cell → 朝向纯度
  - 轮廓整合的前驱: 方向对比度和交互为轮廓连续性提供线索

管线:
  图像 → GaborFilterBank (共享 V1 的 32 个滤波器)
       → 2×2 grid × 2 stats × 32 filters = 256d
       → + 方向交互 (16d) + 方向对比度 (4d)
       → divisive normalization → L2 normalize → ~276d

使用:
  from cerebrum.occipital_lobe.v2 import V2
  v2 = V2(image_size=64)
  features = v2.encode(image)  # → (~276,) float32
"""

import numpy as np
from cerebrum.occipital_lobe.visual_pathway import GaborFilterBank


class V2:
    """V2 次级视皮层 — 粗网格 + 方向交互编码。

    封装 GaborFilterBank.encode_v2(), 仅暴露 V2 相关接口。
    """

    def __init__(self, image_size: int = 64):
        self.image_size = image_size
        self._gabor = GaborFilterBank(image_size=image_size, grid_size=4)

    @property
    def n_filters(self) -> int:
        return self._gabor.n_filters  # 32

    def encode(self, image: np.ndarray) -> np.ndarray:
        """图像 → V2 粗网格 + 方向交互特征向量。

        Args:
            image: (H, W) 灰度 或 (H, W, 3) RGB, uint8 或 float

        Returns:
            (~276,) float32 — L2 归一化
            布局: 32 filters × 4 cells × 2 stats (=256)
                  + 4 cells × (n_orient//2) cross-orient (=16)
                  + 4 cells orient-contrast (=4)
        """
        return self._gabor.encode_v2(image)

    # 诊断: 委托给内部的 GaborFilterBank
    @property
    def gains(self) -> np.ndarray:
        return self._gabor.gains


# ================================================================
# 自测
# ================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  V2 (BA18) Test")
    print("=" * 60)

    rng = np.random.default_rng(42)
    v2 = V2(image_size=64)
    print(f"  Filters: {v2.n_filters}")

    noise = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    vec = v2.encode(noise)
    print(f"  Noise encoding: shape={vec.shape}, norm={np.linalg.norm(vec):.4f}")

    # 纹理 vs 平滑
    stripe = np.zeros((64, 64, 3), dtype=np.uint8)
    for i in range(0, 64, 4):
        stripe[:, i:i + 2] = 255
    vec_s = v2.encode(stripe)
    cos = np.dot(vec, vec_s) / (np.linalg.norm(vec) * np.linalg.norm(vec_s) + 1e-8)
    print(f"  Cosine(noise, stripe) = {cos:.4f}")

    print("  [PASS] V2 test complete")
