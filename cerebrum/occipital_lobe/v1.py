"""
v1.py — 初级视皮层 V1 (Primary Visual Cortex / Striate Cortex)

对应脑区: BA17 (纹状皮层, V1)
所属层级: 大脑 → 枕叶 → V1

功能职责:
  - 边缘检测: 方向选择性 (orientation selectivity)
  - 空间频率: 4 个尺度的细节 (σ=2,4,6,8)
  - 4×4 空间网格池化: 保留 retinotopic 位置信息
  - Hebb 可塑性: 经常激活的滤波器 → 更敏感

V1 是视觉皮层第一站 — 来自 LGN 的信号首次在这里被皮层处理。

管线:
  图像 → GaborFilterBank (4 scales × 8 orientations = 32 filters)
       → 4×4 grid × 2 stats (mean, std) = 1024d raw
       → divisive normalization → Hebb gain modulation → L2 normalize

使用:
  from cerebrum.occipital_lobe.v1 import V1
  v1 = V1(image_size=64, grid_size=4)
  features = v1.encode(image)  # → (1024,) float32
"""

import numpy as np
from cerebrum.occipital_lobe.visual_pathway import GaborFilterBank


class V1:
    """BA17 初级视皮层 — 边缘/方向选择性 Gabor 编码。

    封装 GaborFilterBank.encode(), 仅暴露 V1 相关接口:
      - encode(): 4×4 网格池化, 1024d raw features
      - get_gain_profile(): Hebb 增益诊断
      - reset_gains(): 重置可塑性
    """

    def __init__(self, image_size: int = 64, grid_size: int = 4):
        self.image_size = image_size
        self.grid_size = grid_size
        self._gabor = GaborFilterBank(image_size=image_size, grid_size=grid_size)

    # ---- 属性 ----
    @property
    def n_filters(self) -> int:
        return self._gabor.n_filters  # 32

    @property
    def raw_dim(self) -> int:
        return self._gabor.raw_dim    # 1024

    # ---- 编码 ----
    def encode(self, image: np.ndarray, learn: bool = False) -> np.ndarray:
        """图像 → V1 边缘/方向特征向量。

        Args:
            image: (H, W) 灰度 或 (H, W, 3) RGB, uint8 或 float
            learn: 是否更新 Hebb 增益

        Returns:
            (raw_dim,) float32 — L2 归一化, 4×4 网格池化
        """
        return self._gabor.encode(image, learn=learn)

    # ---- Hebb 可塑性 ----
    def get_gain_profile(self) -> dict:
        """返回 Gabor 滤波器增益概况 (诊断用)。"""
        return self._gabor.get_gain_profile()

    def reset_gains(self):
        """重置 Hebb 增益到 1.0。"""
        self._gabor.reset_gains()

    # ---- 诊断 ----
    def filter_info(self, idx: int) -> dict:
        """返回指定滤波器的参数信息。"""
        return self._gabor.filter_info(idx)


# ================================================================
# 自测
# ================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  V1 (BA17) Test")
    print("=" * 60)

    rng = np.random.default_rng(42)
    v1 = V1(image_size=64, grid_size=4)
    print(f"  Filters: {v1.n_filters}, Raw dim: {v1.raw_dim}")

    noise = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    vec = v1.encode(noise)
    print(f"  Noise encoding: shape={vec.shape}, norm={np.linalg.norm(vec):.4f}")

    # Hebb learn
    for _ in range(30):
        v1.encode(noise, learn=True)
    p = v1.get_gain_profile()
    print(f"  After 30 learns: mean_gain={p['mean_gain']:.3f}, std={p['std_gain']:.3f}")

    print("  [PASS] V1 test complete")
