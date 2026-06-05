"""
layer0_visual.py —— Gabor 滤波器组 + Hebb V1 视觉皮层
自由能原理智能体 — 阶段 0: 视觉基础

零训练参数，纯数学信号处理。
Gabor 滤波器的地位 = 生物的"视网膜 + V1 简单细胞":
  不是认知模型，是将外界物理信号(光)转为神经脉冲的视觉传感器。

生物等价:
  Gabor 函数  = V1 简单细胞的感受野数学模型
  4 个空间尺度 = 不同大小的感受野
  8 个方向     = 不同方向选择性
  4×4 网格池化 = 空间位置保留 (模拟 V1 的 retinotopic map)
  Hebb 增益    = 感受野可塑性 (经常激活 → 更敏感)

输出:
  32 滤波器 × 16 网格 × 2 统计量 (mean, std) = 1024 dims
  → PCA 降到 64 dims (由 VisualEnvironment 负责)
"""

import numpy as np
from scipy.fft import fft2, ifft2
from scipy.ndimage import uniform_filter


class GaborFilterBank:
    """仿 V1 简单细胞的 Gabor 滤波器组 + 空间网格池化。

    数学定义:
      G(x, y) = exp(-(x'² + γ²y'²)/(2σ²)) × cos(2πx'/λ + ψ)
      其中 x' = x·cos(θ) + y·sin(θ), y' = -x·sin(θ) + y·cos(θ)

    参数:
      4 scales × 8 orientations = 32 Gabor filters
      4×4 spatial grid pooling  = 保留 retinotopic 位置信息
      32 × 16 × 2 = 1024 raw features → PCA → 64 dims
    """

    def __init__(self, n_scales: int = 4, n_orientations: int = 8,
                 image_size: int = 64, grid_size: int = 4):
        self.n_scales = n_scales
        self.n_orientations = n_orientations
        self.n_filters = n_scales * n_orientations  # 32
        self.image_size = image_size
        self.grid_size = grid_size                  # 4×4

        # 原始特征维度 (PCA 前)
        self.raw_dim = self.n_filters * (grid_size * grid_size) * 2
        # 32 × 16 × 2 = 1024

        # 空间尺度和方向
        self.sigmas = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32)
        self.thetas = np.linspace(0, np.pi, n_orientations,
                                  endpoint=False, dtype=np.float32)

        # ---- M/P/K 通路参数 (v5.0) ----
        # M 通路 (parasol): 大 σ, 低空间频率, 高时间频率 → 运动/粗略空间
        self.M_sigmas = np.array([4.0, 6.0, 8.0, 12.0], dtype=np.float32)
        self.M_thetas = np.linspace(0, np.pi, n_orientations,
                                      endpoint=False, dtype=np.float32)

        # P 通路 (midget): 小 σ, 高空间频率 → 精细形状/纹理
        self.P_sigmas = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        self.P_thetas = np.linspace(0, np.pi, n_orientations,
                                      endpoint=False, dtype=np.float32)

        # K 通路 (bistratified): 中 σ, 中空间频率 → 颜色
        self.K_sigmas = np.array([2.0, 3.0, 5.0, 7.0], dtype=np.float32)
        self.K_thetas = np.linspace(0, np.pi, n_orientations,
                                      endpoint=False, dtype=np.float32)

        # M/P/K 滤波器核 (lazy built)
        self._M_kernels = None
        self._P_kernels = None
        self._K_kernels = None

        # ---- Hebb 可塑性 ----
        self.gains = np.ones(self.n_filters, dtype=np.float32)
        self.gain_lr: float = 0.005

        # Divisive normalization params (Module B: brightness/color constancy)
        self.surround_sigma: float = image_size / 16.0  # ~4 for 64×64
        self.semi_saturation: float = 0.1               # σ², V1-like

        # 构建核 + 预计算 FFT
        self.kernels = self._build_kernels()
        self._build_surround_kernel()
        self.n_encodes: int = 0

        # Module D: 视觉预测编码状态
        self._has_last: bool = False
        self._last_v1: np.ndarray = None
        self._last_v2: np.ndarray = None
        self._last_v4: np.ndarray = None

        # 网格边界 (预计算)
        grid_bounds = np.linspace(0, image_size, grid_size + 1, dtype=int)
        self.n_cells = grid_size * grid_size  # 16
        self._grid_slices = []
        for gy in range(grid_size):
            for gx in range(grid_size):
                self._grid_slices.append((
                    slice(grid_bounds[gy], grid_bounds[gy + 1]),
                    slice(grid_bounds[gx], grid_bounds[gx + 1]),
                ))

    # ================================================================
    # Gabor 核生成
    # ================================================================

    def _gabor_kernel(self, sigma: float, theta: float,
                      ksize: int = 21, psi: float = 0.0) -> np.ndarray:
        """生成单个 Gabor 滤波器核。

        Args:
            sigma: Gaussian envelope width
            theta: preferred orientation (radians)
            ksize: kernel size
            psi: phase offset (0=cos/even, π/2=sin/odd)
        """
        lam = sigma * 2.0
        gamma = 0.5

        half = ksize // 2
        y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float32)

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        x_theta = x * cos_t + y * sin_t
        y_theta = -x * sin_t + y * cos_t

        gaussian = np.exp(-(x_theta ** 2 + gamma ** 2 * y_theta ** 2)
                          / (2.0 * sigma ** 2))
        sinusoidal = np.cos(2.0 * np.pi * x_theta / lam + psi)
        kernel = gaussian * sinusoidal

        kernel -= kernel.mean()
        std = np.std(kernel)
        if std > 1e-8:
            kernel /= std
        return kernel.astype(np.float32)

    def _build_kernels(self) -> list[np.ndarray]:
        """构建 Gabor 核 (cos-phase, ψ=0) + 预计算 FFT (128×128 padding)。"""
        kernels = []
        for sigma in self.sigmas:
            ksize = int(sigma * 5)
            ksize = max(5, min(41, ksize))
            if ksize % 2 == 0:
                ksize += 1
            for theta in self.thetas:
                kernel = self._gabor_kernel(sigma, theta, ksize, psi=0.0)
                kernels.append(kernel)

        # 预计算 FFT
        self.fft_size = self.image_size * 2  # 128
        self._kernel_ffts = []
        for kernel in kernels:
            k_h, k_w = kernel.shape
            padded = np.zeros((self.fft_size, self.fft_size), dtype=np.float32)
            start_h = (self.fft_size - k_h) // 2
            start_w = (self.fft_size - k_w) // 2
            padded[start_h:start_h + k_h, start_w:start_w + k_w] = kernel
            self._kernel_ffts.append(fft2(padded).conj())

        return kernels

    def _build_kernels_for(self, sigmas, thetas) -> list[np.ndarray]:
        """Build Gabor kernels for a given (sigmas, thetas) parameter set.

        Follows the same pattern as _build_kernels() but accepts custom
        sigma/theta arrays for M/P/K pathway-specific tuning.

        Args:
            sigmas: array of Gaussian envelope widths
            thetas: array of preferred orientations (radians)

        Returns:
            list of Gabor kernels (cos-phase, ψ=0)
        """
        kernels = []
        for sigma in sigmas:
            ksize = int(sigma * 5)
            ksize = max(5, min(41, ksize))
            if ksize % 2 == 0:
                ksize += 1
            for theta in thetas:
                kernel = self._gabor_kernel(sigma, theta, ksize, psi=0.0)
                kernels.append(kernel)
        return kernels

    def _ensure_channel_kernels(self):
        """Lazy-build M/P/K channel kernels if not yet built."""
        if self._M_kernels is None:
            self._M_kernels = self._build_kernels_for(self.M_sigmas, self.M_thetas)
        if self._P_kernels is None:
            self._P_kernels = self._build_kernels_for(self.P_sigmas, self.P_thetas)
        if self._K_kernels is None:
            self._K_kernels = self._build_kernels_for(self.K_sigmas, self.K_thetas)

    # ================================================================
    # Divisive Normalization (Module B: brightness/color constancy)
    # ================================================================

    def _build_surround_kernel(self):
        """构建 Gaussian surround 核用于 divisive normalization。

        V1 surround suppression: 每个神经元的响应被其空间邻域的
        总能量归一化 → 实现亮度恒常性和对比度归一化。
        """
        radius = max(1, self.image_size // 16)  # ~4
        ksize = radius * 2 + 1                 # ~9
        y, x = np.mgrid[-radius:radius + 1,
                        -radius:radius + 1].astype(np.float32)
        gaussian = np.exp(-(x ** 2 + y ** 2)
                          / (2.0 * self.surround_sigma ** 2))
        gaussian /= gaussian.sum()

        # Pad to FFT size
        padded = np.zeros((self.fft_size, self.fft_size), dtype=np.float32)
        start = (self.fft_size - ksize) // 2
        padded[start:start + ksize, start:start + ksize] = gaussian
        self._surround_kernel_fft = fft2(padded).conj()

    def _divisive_normalize(self, response: np.ndarray
                            ) -> np.ndarray:
        """Divisive normalization: V1 surround suppression.

        R_norm = R / (σ² + local_sq_energy)
        local_sq_energy = Gaussian_blur(R²)

        效果: 局部高对比度区域 → 归一化抑制 → 类似亮度恒常性。
        同一物体在亮/暗环境下产生更一致的响应。
        """
        h, w = response.shape
        # 局部平方能量 (仿生物 V1: Σ w_ij * R_j²)
        sq_response = response.astype(np.float32) ** 2

        # 如果响应图尺寸与 image_size 不同 (如 pulvinar 降采样),
        # 用对应的核尺寸做卷积，回退到 scipy uniform_filter
        if h != self.image_size:
            from scipy.ndimage import uniform_filter
            radius = max(1, h // 16)
            local_sq_energy = uniform_filter(
                sq_response, size=radius * 2 + 1)
        else:
            padded = np.zeros((self.fft_size, self.fft_size),
                              dtype=np.float32)
            padded[:h, :w] = sq_response
            energy_full = np.real(
                ifft2(fft2(padded) * self._surround_kernel_fft))
            local_sq_energy = np.abs(energy_full[:h, :w])

        # Normalize
        return response / (self.semi_saturation ** 2
                           + np.sqrt(local_sq_energy) + 1e-8)

    # ================================================================
    # 图像编码
    # ================================================================

    def encode(self, image: np.ndarray, learn: bool = False
               ) -> np.ndarray:
        """图像 → raw_dim 视觉特征向量 (PCA 前, 4x4 网格池化 = V1)。

        管线:
        1. 预处理: RGB→灰度, resize, 归一化 [-1, 1]
        2. FFT 卷积: 1×image FFT + 32×kernel IFFT
        3. 4×4 网格池化: 每个网格 cell → (mean, std)
        4. Hebb 增益调制 + L2 归一化

        Returns:
            (raw_dim,) float32 — 需要 PCA 降维
        """
        gray = self._preprocess(image)
        fft_size = self.fft_size
        gray_padded = np.zeros((fft_size, fft_size), dtype=np.float32)
        gray_padded[:self.image_size, :self.image_size] = gray
        image_fft = fft2(gray_padded)

        n_cells = self.grid_size * self.grid_size  # 16
        features = np.zeros(self.n_filters * n_cells * 2, dtype=np.float32)
        raw_means = np.zeros(self.n_filters, dtype=np.float32)

        for i in range(self.n_filters):
            resp_full = np.real(ifft2(image_fft * self._kernel_ffts[i]))
            response = resp_full[:self.image_size, :self.image_size]
            # Module B: divisive normalization (brightness constancy)
            response = self._divisive_normalize(response)
            raw_means[i] = float(np.mean(np.abs(response)))

            modulated = response * self.gains[i]
            for ci, (sy, sx) in enumerate(self._grid_slices):
                cell = modulated[sy, sx]
                features[i * n_cells * 2 + ci * 2] = float(np.mean(cell))
                features[i * n_cells * 2 + ci * 2 + 1] = float(np.std(cell))

        self.n_encodes += 1
        if learn:
            self._hebb_update(raw_means)

        norm = np.linalg.norm(features)
        if norm > 1e-8:
            features /= norm

        return features.astype(np.float32)

    # ================================================================
    # M/P/K 通路编码 (v5.0) — 三条并行通路的独立编码
    # ================================================================

    def _encode_channel(self, image: np.ndarray, kernels: list,
                        learn: bool = False) -> np.ndarray:
        """Generic channel encoder: convolve image with kernels, pool, normalize.

        Used by encode_M(), encode_P(), encode_K() to produce pathway-specific
        feature vectors. Follows the same pipeline as encode() but uses
        channel-specific kernel sets.

        Args:
            image: (H, W) uint8 grayscale or opponent-channel image
            kernels: list of Gabor kernels (pre-built for this channel)
            learn: whether to update Hebb gains

        Returns:
            (n_kernels × n_cells × 2,) float32 — L2 normalized
        """
        gray = self._preprocess(image)
        fft_size = self.fft_size
        gray_padded = np.zeros((fft_size, fft_size), dtype=np.float32)
        gray_padded[:self.image_size, :self.image_size] = gray
        image_fft = fft2(gray_padded)

        # Precompute FFTs for these kernels if not cached
        n_k = len(kernels)
        kernel_ffts = []
        for kernel in kernels:
            k_h, k_w = kernel.shape
            padded = np.zeros((fft_size, fft_size), dtype=np.float32)
            start_h = (fft_size - k_h) // 2
            start_w = (fft_size - k_w) // 2
            padded[start_h:start_h + k_h, start_w:start_w + k_w] = kernel
            kernel_ffts.append(fft2(padded).conj())

        n_cells = self.grid_size * self.grid_size  # 16
        features = np.zeros(n_k * n_cells * 2, dtype=np.float32)
        raw_means = np.zeros(n_k, dtype=np.float32)

        for i in range(n_k):
            resp_full = np.real(ifft2(image_fft * kernel_ffts[i]))
            response = resp_full[:self.image_size, :self.image_size]
            response = self._divisive_normalize(response)
            raw_means[i] = float(np.mean(np.abs(response)))

            # Use shared gains if available (first n_k entries)
            if i < len(self.gains):
                gain = self.gains[i]
            else:
                gain = 1.0
            modulated = response * gain

            for ci, (sy, sx) in enumerate(self._grid_slices):
                cell = modulated[sy, sx]
                features[i * n_cells * 2 + ci * 2] = float(np.mean(cell))
                features[i * n_cells * 2 + ci * 2 + 1] = float(np.std(cell))

        self.n_encodes += 1
        if learn:
            self._hebb_update(raw_means)

        norm = np.linalg.norm(features)
        if norm > 1e-8:
            features /= norm

        return features.astype(np.float32)

    def encode_M(self, image: np.ndarray, learn: bool = False) -> np.ndarray:
        """M 通路: 大 σ, 低空间频率 → 运动/粗略空间 (→ MT / 背侧通路).

        M-type retinal ganglion cells (parasol) have:
          - Large receptive fields (coarse spatial resolution)
          - High contrast sensitivity
          - Fast conduction velocity
          - No color opponency (achromatic)

        Uses M_sigmas=[4,6,8,12] for coarse-scale filtering.

        Args:
            image: (H, W) uint8 or (H, W, 3) RGB image
            learn: whether to update Hebb gains

        Returns:
            (n_M_kernels × 16 × 2,) float32 — L2 normalized
        """
        self._ensure_channel_kernels()
        if image.ndim == 3 and image.shape[2] >= 3:
            gray = np.mean(image.astype(np.float32), axis=2).astype(np.uint8)
        elif image.dtype == np.float32 or image.dtype == np.float64:
            gray = np.clip(image, 0, 255).astype(np.uint8)
        else:
            gray = image
        return self._encode_channel(gray, self._M_kernels, learn=learn)

    def encode_P(self, image: np.ndarray, learn: bool = False) -> np.ndarray:
        """P 通路: 小 σ, 高空间频率 → 精细形状/纹理 (→ V4 / 腹侧通路).

        P-type retinal ganglion cells (midget) have:
          - Small receptive fields (fine spatial resolution)
          - Red-green color opponency
          - Sustained responses
          - Slower conduction velocity

        Uses P_sigmas=[1,2,3,4] for fine-scale filtering.

        Args:
            image: (H, W) uint8 or (H, W, 3) RGB image
            learn: whether to update Hebb gains

        Returns:
            (n_P_kernels × 16 × 2,) float32 — L2 normalized
        """
        self._ensure_channel_kernels()
        if image.ndim == 3 and image.shape[2] >= 3:
            gray = np.mean(image.astype(np.float32), axis=2).astype(np.uint8)
        elif image.dtype == np.float32 or image.dtype == np.float64:
            gray = np.clip(image, 0, 255).astype(np.uint8)
        else:
            gray = image
        return self._encode_channel(gray, self._P_kernels, learn=learn)

    def encode_K(self, image: np.ndarray, learn: bool = False) -> np.ndarray:
        """K 通路: 中 σ, 蓝-黄颜色拮抗 → 颜色处理 (→ V4 颜色恒常).

        K-type retinal ganglion cells (bistratified) have:
          - Blue-yellow color opponency
          - Medium spatial resolution
          - May contribute to blindsight

        Converts RGB to Blue-Yellow opponent channel before Gabor filtering.
        Uses K_sigmas=[2,3,5,7] for mid-scale filtering.

        Args:
            image: (H, W, 3) RGB image (uint8 or float)
            learn: whether to update Hebb gains

        Returns:
            (n_K_kernels × 16 × 2,) float32 — L2 normalized
        """
        self._ensure_channel_kernels()
        if image.ndim == 3 and image.shape[2] >= 3:
            img = image.astype(np.float32)
            R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            # Blue-Yellow opponent channel: B - (R+G)/2
            BY = (B - (R + G) / 2.0)
            # Map to [0, 255] range for _preprocess
            BY = np.clip((BY + 128) * 0.7 + 64, 0, 255).astype(np.uint8)
        else:
            BY = image
        return self._encode_channel(BY, self._K_kernels, learn=learn)

    # ================================================================
    # V2 编码 — 粗网格 + 方向池化 (仿 V2 复杂细胞)
    # ================================================================

    def encode_v2(self, image: np.ndarray) -> np.ndarray:
        """V2 编码: 更大的感受野 (2x2 网格) + 跨方向交互。

        V2 与 V1 的关键差异:
        - 2×2 网格 (4 cells) vs 4×4 (16 cells) → 更大的空间不变性
        - 每个 cell 覆盖 32×32 像素 (V1: 16×16)
        - 额外计算跨方向交互: 相邻方向响应的乘积 (仿"角点检测")
        - 方向对比度: max(response) - min(response) per cell → "朝向纯度"

        Returns:
            (n_filters * 4 * 2 + extra,) float32 — ~300 dims
        """
        gray = self._preprocess(image)
        fft_size = self.fft_size
        gray_padded = np.zeros((fft_size, fft_size), dtype=np.float32)
        gray_padded[:self.image_size, :self.image_size] = gray
        image_fft = fft2(gray_padded)

        half_grid = self.image_size // 2  # 32
        v2_cells = [
            (slice(0, half_grid), slice(0, half_grid)),             # 左上
            (slice(0, half_grid), slice(half_grid, self.image_size)),  # 右上
            (slice(half_grid, self.image_size), slice(0, half_grid)),  # 左下
            (slice(half_grid, self.image_size), slice(half_grid, self.image_size)),  # 右下
        ]

        n_cells = 4
        # 基础: 32 filters x 4 cells x 2 stats = 256
        # 额外: 4 cells x 4 orient-pairs x 1 cross = 16
        # 额外: 4 cells x 1 orient-contrast = 4
        # 共 256 + 16 + 4 = 276
        extra_dim = n_cells * (self.n_orientations // 2) + n_cells
        features = np.zeros(
            self.n_filters * n_cells * 2 + extra_dim, dtype=np.float32)

        for i in range(self.n_filters):
            resp_full = np.real(ifft2(image_fft * self._kernel_ffts[i]))
            response = resp_full[:self.image_size, :self.image_size]
            # Module B: divisive normalization
            response = self._divisive_normalize(response)
            modulated = response * self.gains[i]

            for ci, (sy, sx) in enumerate(v2_cells):
                cell = modulated[sy, sx]
                features[i * n_cells * 2 + ci * 2] = float(np.mean(cell))
                features[i * n_cells * 2 + ci * 2 + 1] = float(np.std(cell))

        # ---- 方向交互特征 (V2 特有) ----
        base_offset = self.n_filters * n_cells * 2  # 256

        # 每个 cell 内的方向对比度
        for ci in range(n_cells):
            orient_means = np.zeros(self.n_orientations, dtype=np.float32)
            for oi in range(self.n_orientations):
                orient_means[oi] = np.mean([
                    features[i * n_cells * 2 + ci * 2]
                    for i in range(oi, self.n_filters, self.n_orientations)
                ])
            # 方向对比度: max - min
            features[base_offset + ci] = (
                float(np.max(orient_means)) - float(np.min(orient_means)))

        # 相邻方向交互 (角点检测)
        cross_offset = base_offset + n_cells  # 260
        for ci in range(n_cells):
            orient_means = np.zeros(self.n_orientations, dtype=np.float32)
            for oi in range(self.n_orientations):
                orient_means[oi] = np.mean([
                    features[i * n_cells * 2 + ci * 2]
                    for i in range(oi, self.n_filters, self.n_orientations)
                ])
            for oi in range(self.n_orientations // 2):
                o1 = oi
                o2 = (oi + 1) % self.n_orientations
                # 相邻方向响应乘积 (两者都强 → 角点)
                features[cross_offset + ci * (self.n_orientations // 2) + oi] = (
                    orient_means[o1] * orient_means[o2])

        # ---- L2 归一化 ----
        norm = np.linalg.norm(features)
        if norm > 1e-8:
            features /= norm

        return features.astype(np.float32)

    # ================================================================
    # V4 编码 — 全局形状 + 曲率 (仿 V4 形状选择性)
    # ================================================================

    def encode_v4(self, image: np.ndarray) -> np.ndarray:
        """V4 编码: 全局 (1x1) 池化 + 曲率检测。

        V4 与 V1/V2 的关键差异:
        - 1×1 全局池化: 完全位置不变 → "存在某种边缘" 而非 "边缘在哪里"
        - 曲率检测: 相邻方向的响应空间相关性 → "边缘在弯曲"
        - V4 看到的是"什么形状存在"而非"形状在哪里"

        Returns:
            (n_filters * 2 + n_orientations,) float32 — ~72 dims
        """
        gray = self._preprocess(image)
        fft_size = self.fft_size
        gray_padded = np.zeros((fft_size, fft_size), dtype=np.float32)
        gray_padded[:self.image_size, :self.image_size] = gray
        image_fft = fft2(gray_padded)

        n_f = self.n_filters  # 32
        n_orient = self.n_orientations  # 8

        # Part 1: Global (1x1) pooling — 32 x 2 = 64 dims
        global_features = np.zeros(n_f * 2, dtype=np.float32)

        # Store full response maps for curvature computation
        response_maps = np.zeros((n_f, self.image_size, self.image_size),
                                dtype=np.float32)

        for i in range(n_f):
            resp_full = np.real(ifft2(image_fft * self._kernel_ffts[i]))
            response = resp_full[:self.image_size, :self.image_size]
            # Module B: divisive normalization
            response = self._divisive_normalize(response)
            modulated = response * self.gains[i]
            response_maps[i] = modulated
            global_features[i * 2] = float(np.mean(modulated))
            global_features[i * 2 + 1] = float(np.std(modulated))

        # Part 2: Curvature detection — spatial correlation between
        # neighboring orientation responses. High correlation = straight edge
        # (both orientations fire together). Low correlation = curve.
        curvature_features = np.zeros(n_orient, dtype=np.float32)
        for oi in range(n_orient):
            o1 = oi
            o2 = (oi + 1) % n_orient
            # Get responses for all scales of these two orientations
            resp_o1 = np.zeros((self.n_scales, self.image_size, self.image_size),
                             dtype=np.float32)
            resp_o2 = np.zeros_like(resp_o1)
            for si in range(self.n_scales):
                fi = si * n_orient + o1
                resp_o1[si] = response_maps[fi]
                fi = si * n_orient + o2
                resp_o2[si] = response_maps[fi]
            # Spatial correlation of neighboring orientation pair
            # Normalized cross-correlation at zero lag
            o1_flat = resp_o1.ravel()
            o2_flat = resp_o2.ravel()
            o1_n = o1_flat - o1_flat.mean()
            o2_n = o2_flat - o2_flat.mean()
            denom = (np.std(o1_n) * np.std(o2_n) + 1e-8)
            curvature_features[oi] = float(np.dot(o1_n, o2_n) / denom
                                          / len(o1_flat))

        # Combine
        features = np.concatenate([global_features, curvature_features])
        dim = len(features)
        # 同时返回方便上游知道维度
        if not hasattr(self, '_v4_dim'):
            self._v4_dim = dim

        # L2 normalize
        norm = np.linalg.norm(features)
        if norm > 1e-8:
            features /= norm

        return features.astype(np.float32)

    # ================================================================
    # 色拮抗编码 — Red-Green + Blue-Yellow opponent channels
    # ================================================================

    def encode_color(self, image: np.ndarray) -> np.ndarray:
        """色拮抗编码: RG (红绿) + BY (蓝黄) 通过 Gabor 滤波器。

        生物类比:
          视网膜双拮抗细胞 → LGN Parvo/Konio 层 → V1 Blob 区
          RG = L - M (长波长 - 中波长锥体)
          BY = S - (L+M)/2 (短波长 - 亮度)

        管线:
          1. RGB → opponent channels
          2. 每个通道通过 32 个 Gabor 滤波器 (FFT 卷积)
          3. 2×2 网格池化 (与 V2 相同的空间粗糙度)
          4. mean + std per cell per filter

        Returns:
            (2 × 32 × 4 × 2 = 512,) float32 — L2 归一化
        """
        # ---- 颜色预处理 ----
        if image.ndim == 3 and image.shape[2] >= 3:
            img = image.astype(np.float32)
        elif image.ndim == 3:
            img = np.stack([image[:,:,0]] * 3, axis=-1).astype(np.float32)
        else:
            img = np.stack([image] * 3, axis=-1).astype(np.float32)

        # Resize to image_size
        if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
            from PIL import Image
            img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
            pil_img = Image.fromarray(img_uint8)
            pil_img = pil_img.resize((self.image_size, self.image_size),
                                     Image.LANCZOS)
            img = np.array(pil_img, dtype=np.float32)

        # Normalize to [-1, 1]
        img = (img / 255.0) * 2.0 - 1.0

        R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]

        # Opponent channels
        RG = R - G                        # Red-Green opponent
        BY = B - (R + G) * 0.5            # Blue-Yellow opponent

        opponent_channels = [RG, BY]

        # ---- Gabor 卷积 + 2x2 网格池化 ----
        fft_size = self.fft_size
        half_grid = self.image_size // 2
        v2_cells = [
            (slice(0, half_grid), slice(0, half_grid)),
            (slice(0, half_grid), slice(half_grid, self.image_size)),
            (slice(half_grid, self.image_size), slice(0, half_grid)),
            (slice(half_grid, self.image_size), slice(half_grid, self.image_size)),
        ]
        n_cells = 4

        # 2 channels × 32 filters × 4 cells × 2 stats = 512
        n_features = 2 * self.n_filters * n_cells * 2
        features = np.zeros(n_features, dtype=np.float32)

        for ch_idx, channel in enumerate(opponent_channels):
            ch_padded = np.zeros((fft_size, fft_size), dtype=np.float32)
            ch_padded[:self.image_size, :self.image_size] = channel
            ch_fft = fft2(ch_padded)

            ch_offset = ch_idx * self.n_filters * n_cells * 2

            for i in range(self.n_filters):
                resp_full = np.real(ifft2(ch_fft * self._kernel_ffts[i]))
                response = resp_full[:self.image_size, :self.image_size]
                # Module B: divisive normalization
                response = self._divisive_normalize(response)
                modulated = response * self.gains[i]

                filt_offset = ch_offset + i * n_cells * 2
                for ci, (sy, sx) in enumerate(v2_cells):
                    cell = modulated[sy, sx]
                    features[filt_offset + ci * 2] = float(np.mean(cell))
                    features[filt_offset + ci * 2 + 1] = float(np.std(cell))

        # ---- 跨通道归一化 (Module B: 颜色恒常性) ----
        # RG 和 BY 通道互相抑制: 一个通道强 → 抑制另一个
        # 模拟 V1 双拮抗细胞的颜色恒常性
        half_features = self.n_filters * n_cells * 2  # 256
        rg_norm = float(np.linalg.norm(features[:half_features]))
        by_norm = float(np.linalg.norm(features[half_features:]))
        features[:half_features] /= (1.0 + by_norm)
        features[half_features:] /= (1.0 + rg_norm)

        # ---- L2 归一化 ----
        norm = np.linalg.norm(features)
        if norm > 1e-8:
            features /= norm

        return features.astype(np.float32)

    # ================================================================
    # Pulvinar 编码 — 低空间频率捷径直通 IT
    # ================================================================

    def encode_pulvinar(self, image: np.ndarray) -> np.ndarray:
        """Pulvinar → IT 低空间频率捷径。

        生物类比:
          丘脑枕核 (Pulvinar) 接收视网膜直接输入并投射到 IT，
          完全绕过 V1-V2-V4。处理低空间频率 (模糊、粗糙) 信息，
          提供快速的"场景要旨"(gist) 识别。

        管线:
          1. 强高斯模糊 (σ=image_size/6) → 去除高空间频率
          2. 降采样到 image_size//4
          3. 仅使用大尺度 Gabor 滤波器 (σ=6,8) → 16 个滤波器
          4. 全局 1×1 池化 (无空间信息 — 纯"有什么"而非"在哪里")
          5. 16 × 2 stats = 32d → L2 归一化

        Returns:
            (32,) float32 — L2 归一化低空间频率特征
        """
        from scipy.ndimage import gaussian_filter

        # ---- 颜色预处理 ----
        if image.ndim == 3 and image.shape[2] >= 3:
            img = image.astype(np.float32)
        elif image.ndim == 3:
            img = np.stack([image[:, :, 0]] * 3, axis=-1).astype(np.float32)
        else:
            img = np.stack([image] * 3, axis=-1).astype(np.float32)

        # Resize to image_size
        if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
            from PIL import Image
            img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
            pil_img = Image.fromarray(img_uint8)
            pil_img = pil_img.resize((self.image_size, self.image_size),
                                     Image.LANCZOS)
            img = np.array(pil_img, dtype=np.float32)

        # ---- 低空间频率处理 ----
        # 转灰度 + 归一化
        gray = (0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1]
                + 0.1140 * img[:, :, 2])
        gray = (gray / 255.0) * 2.0 - 1.0

        # 强高斯模糊: σ = image_size / 6 (对 128×128 = σ≈21)
        blur_sigma = self.image_size / 6.0
        blurred = gaussian_filter(gray, sigma=blur_sigma)

        # 降采样到 1/4
        ds_factor = 4
        ds_size = self.image_size // ds_factor
        blurred_ds = blurred[::ds_factor, :ds_size][:ds_size, :]

        # ---- 仅大尺度 Gabor 编码 (σ=6,8, 最后 2 个尺度) ----
        large_scale_start = self.n_scales - 2  # 最后 2 个尺度
        n_large_scales = 2
        n_large_filters = n_large_scales * self.n_orientations  # 16

        # 为降采样图像准备 FFT
        fft_size = self.fft_size
        gray_padded = np.zeros((fft_size, fft_size), dtype=np.float32)
        gray_padded[:ds_size, :ds_size] = blurred_ds
        image_fft = fft2(gray_padded)

        features = np.zeros(n_large_filters * 2, dtype=np.float32)

        for si in range(large_scale_start, self.n_scales):
            for oi in range(self.n_orientations):
                fi = si * self.n_orientations + oi
                local_idx = (si - large_scale_start) * self.n_orientations + oi

                resp_full = np.real(ifft2(image_fft * self._kernel_ffts[fi]))
                response = resp_full[:ds_size, :ds_size]
                # Module B: divisive normalization
                response = self._divisive_normalize(response)
                modulated = response * self.gains[fi]

                # 全局 1×1 池化
                features[local_idx * 2] = float(np.mean(modulated))
                features[local_idx * 2 + 1] = float(np.std(modulated))

        # ---- L2 归一化 ----
        norm = np.linalg.norm(features)
        if norm > 1e-8:
            features /= norm

        return features.astype(np.float32)

    # ================================================================
    # Dorsal Stream 编码 — V1 → MT → MST ("在哪里"通路)
    # ================================================================

    def encode_dorsal(self, image: np.ndarray) -> np.ndarray:
        """背侧通路: V1 → MT → MST ("在哪里"流)。

        生物类比:
          MT (中颞叶): 方向选择性运动检测
          MST (内上颞叶): 光流模式 (膨胀/收缩/旋转)
          背侧通路处理空间位置和运动信息，独立于腹侧"是什么"通路。

        对静态图像的近似:
          MT: 每个朝向的 Gabor 响应空间梯度 → "隐含运动能量"
          MST: 每个朝向响应质心 (center of mass) → "特征在哪里"

        管线:
          1. 计算 32 个 Gabor 响应图
          2. 合并为 8 个朝向图 (对 4 个尺度取平均)
          3. MT-like: 每个朝向图的空间梯度幅度 → 4 象限池化
             8 orientations × 4 quadrants = 32d → 取 mean+std → 16d
          4. MST-like: 每个朝向图的响应质心 (cx, cy)
             8 orientations × 2 = 16d
          5. 拼接 16d + 16d = 32d → L2 归一化

        Returns:
            (32,) float32 — L2 归一化背侧通路特征
        """
        # ---- 预处理 ----
        gray = self._preprocess(image)

        fft_size = self.fft_size
        gray_padded = np.zeros((fft_size, fft_size), dtype=np.float32)
        gray_padded[:self.image_size, :self.image_size] = gray
        image_fft = fft2(gray_padded)

        # 存储 32 个响应图
        response_maps = np.zeros(
            (self.n_filters, self.image_size, self.image_size),
            dtype=np.float32)

        for i in range(self.n_filters):
            resp_full = np.real(ifft2(image_fft * self._kernel_ffts[i]))
            response = resp_full[:self.image_size, :self.image_size]
            # Module B: divisive normalization
            response = self._divisive_normalize(response)
            response_maps[i] = response

        # ---- 合并为 8 个朝向图 (跨尺度平均) ----
        orient_maps = np.zeros(
            (self.n_orientations, self.image_size, self.image_size),
            dtype=np.float32)
        for oi in range(self.n_orientations):
            for si in range(self.n_scales):
                fi = si * self.n_orientations + oi
                orient_maps[oi] += np.abs(response_maps[fi])
            orient_maps[oi] /= self.n_scales

        # ---- MT-like: 空间梯度能量 ----
        # 每朝向: mean gradient magnitude + max gradient magnitude
        mt_features = np.zeros(self.n_orientations * 2, dtype=np.float32)

        for oi in range(self.n_orientations):
            omap = orient_maps[oi]
            # 空间梯度 (Sobel-like)
            gy, gx = np.gradient(omap)
            grad_mag = np.sqrt(gx ** 2 + gy ** 2)

            mt_features[oi * 2] = float(np.mean(grad_mag))
            mt_features[oi * 2 + 1] = float(np.max(grad_mag))

        # ---- MST-like: 响应质心 (center of mass) ----
        mst_features = np.zeros(self.n_orientations * 2, dtype=np.float32)
        y_coords = np.arange(self.image_size, dtype=np.float32).reshape(-1, 1)
        x_coords = np.arange(self.image_size, dtype=np.float32).reshape(1, -1)

        for oi in range(self.n_orientations):
            omap = np.abs(orient_maps[oi])
            total_mass = omap.sum() + 1e-8

            cx = float(np.sum(x_coords * omap) / total_mass) / self.image_size
            cy = float(np.sum(y_coords * omap) / total_mass) / self.image_size

            mst_features[oi * 2] = cx      # 归一化质心 x [0, 1]
            mst_features[oi * 2 + 1] = cy  # 归一化质心 y [0, 1]

        # ---- 拼接 + L2 归一化 ----
        features = np.concatenate([mt_features, mst_features])
        norm = np.linalg.norm(features)
        if norm > 1e-8:
            features /= norm

        return features.astype(np.float32)

    # ================================================================
    # Saliency Map (Module C: 视觉显著性 + IOR)
    # ================================================================

    def compute_saliency(self, image: np.ndarray,
                         attention_precision: float = 0.5
                         ) -> tuple[np.ndarray, np.ndarray]:
        """自底向上显著性图 — Center-Surround 差异。

        生物类比:
          上丘 (Superior Colliculus) + V1 显著性图
          Center-surround 差异 → 定位"与众不同的区域"

        算法:
          1. 计算 32 个 Gabor 响应图
          2. 对每个滤波器: center-surround 差异
             center = 原始响应, surround = Gaussian 模糊 (σ=image_size/8)
          3. 跨滤波器求和 → 原始显著性图
          4. 跨尺度增强: fine scale (σ=2) vs coarse (σ=8) 差异
          5. 归一化到 [0, 1]
          6. L1 attention_precision 调制增益
          7. IOR mask: 指数衰减抑制已关注位置

        Args:
            image: (H, W, 3) uint8 RGB
            attention_precision: [0, 1] L1 注意力精度调制

        Returns:
            (saliency_map, attended_features)
            saliency_map: (H, W) float32 [0, 1]
            attended_features: weighted average of V1 features at salient locations
        """
        gray = self._preprocess(image)
        fft_size = self.fft_size
        gray_padded = np.zeros((fft_size, fft_size), dtype=np.float32)
        gray_padded[:self.image_size, :self.image_size] = gray
        image_fft = fft2(gray_padded)

        h, w = self.image_size, self.image_size

        # ---- Center-surround per filter ----
        saliency = np.zeros((h, w), dtype=np.float32)

        for i in range(self.n_filters):
            resp_full = np.real(ifft2(image_fft * self._kernel_ffts[i]))
            response = resp_full[:h, :w]
            # Divisive normalization (Module B)
            response = self._divisive_normalize(response)

            # Center: 原始响应绝对值
            center = np.abs(response)

            # Surround: Gaussian 模糊 (sigma = image_size/8)
            surround = uniform_filter(center, size=max(3, h // 4))

            # |center - surround| → 局部异常 = 显著
            cs_diff = np.abs(center - surround)
            saliency += cs_diff

        # ---- 跨尺度增强 ----
        # Fine scale: 小 sigma (σ=2) 的 Gabor 对细节敏感
        # Coarse scale: 大 sigma (σ=8) 的 Gabor 对整体敏感
        # |fine - coarse| → 多尺度差异增强显著性
        fine_idx = list(range(0, self.n_orientations))       # scale 0 (σ=2)
        coarse_idx = list(range(
            (self.n_scales - 1) * self.n_orientations,
            self.n_filters))                                 # scale 3 (σ=8)

        fine_map = np.zeros((h, w), dtype=np.float32)
        coarse_map = np.zeros((h, w), dtype=np.float32)
        for fi in fine_idx:
            resp_full = np.real(ifft2(image_fft * self._kernel_ffts[fi]))
            fine_map += np.abs(resp_full[:h, :w])
        for fi in coarse_idx:
            resp_full = np.real(ifft2(image_fft * self._kernel_ffts[fi]))
            coarse_map += np.abs(resp_full[:h, :w])

        cross_scale = np.abs(
            fine_map / (len(fine_idx) + 1e-8)
            - coarse_map / (len(coarse_idx) + 1e-8))
        saliency += cross_scale

        # ---- 归一化到 [0, 1] ----
        smax = saliency.max()
        if smax > 1e-8:
            saliency /= smax

        # ---- L1 attention_precision 调制 ----
        # 高精度 → 增益高 → 更锐利的显著性 (更选择性的注意)
        # 低精度 → 增益低 → 更平坦的显著性 (更分散的注意)
        gain = 0.5 + attention_precision
        saliency = np.tanh(gain * saliency)

        # ---- IOR (Inhibition of Return) ----
        if not hasattr(self, '_ior_mask'):
            self._ior_mask = np.ones((h, w), dtype=np.float32)
            self._ior_decay = 0.8   # 每步衰减因子
            self._ior_boost = 0.3   # 当前注视点的抑制量

        saliency_with_ior = saliency * self._ior_mask

        # ---- 更新 IOR mask: 当前峰值位置被抑制 ----
        peak_y, peak_x = np.unravel_index(
            np.argmax(saliency), saliency.shape)
        # 在峰值周围 8×8 区域施加抑制
        r = h // 8
        y0, y1 = max(0, peak_y - r), min(h, peak_y + r)
        x0, x1 = max(0, peak_x - r), min(w, peak_x + r)
        self._ior_mask[y0:y1, x0:x1] *= self._ior_boost
        # 全局衰减 → 抑制逐渐消失
        self._ior_mask = np.minimum(
            1.0, self._ior_mask + (1.0 - self._ior_mask) * (1.0 - self._ior_decay))

        # ---- 显著性加权的视觉特征 ----
        # 对 V1 encoding 做显著性加权
        attended_features = np.zeros(self.raw_dim, dtype=np.float32)
        weight_sum = 0.0
        for i in range(self.n_filters):
            resp_full = np.real(ifft2(image_fft * self._kernel_ffts[i]))
            response = np.abs(resp_full[:h, :w])
            # 显著性加权的响应
            weighted = response * saliency_with_ior
            for ci, (sy, sx) in enumerate(self._grid_slices):
                cell = weighted[sy, sx]
                attended_features[i * self.n_cells * 2 + ci * 2] = float(np.mean(cell))
                attended_features[i * self.n_cells * 2 + ci * 2 + 1] = float(np.std(cell))

        wsum = float(np.sum(weighted))
        if wsum > 1e-8:
            attended_features /= wsum * 0.01  # 缩放到合理范围

        return saliency_with_ior.astype(np.float32), attended_features.astype(np.float32)

    def reset_ior(self):
        """重置 IOR mask (切换图像时调用)"""
        self._ior_mask = np.ones((self.image_size, self.image_size),
                                 dtype=np.float32)

    # ================================================================
    # Predictive Coding Buffer (Module D: 视觉预测编码)
    # ================================================================

    def store_encoding(self, v1_vec: np.ndarray,
                       v2_vec: np.ndarray = None,
                       v4_vec: np.ndarray = None):
        """存储最近的编码向量用于预测误差计算。

        V1(t-1) → 预测 V1(t) — 时间平滑先验
        V2 → 预测 V1    — 层级反馈预测
        V4 → 预测 V2    — 层级反馈预测
        """
        self._last_v1 = v1_vec.copy()
        self._last_v2 = v2_vec.copy() if v2_vec is not None else None
        self._last_v4 = v4_vec.copy() if v4_vec is not None else None
        self._has_last = True

    def compute_prediction_error(self, v1_curr: np.ndarray,
                                  v2_curr: np.ndarray = None,
                                  v4_curr: np.ndarray = None
                                  ) -> dict[str, float]:
        """计算视觉预测误差。

        Returns:
            {temporal_v1_err, feedback_v1_err, feedback_v2_err, total}
        """
        result = {'temporal_v1_err': 0.0, 'feedback_v1_err': 0.0,
                  'feedback_v2_err': 0.0, 'total': 0.0}

        if not hasattr(self, '_has_last') or not self._has_last:
            return result

        # 时间预测误差: ||V1(t) - V1(t-1)||²
        if self._last_v1 is not None:
            diff = v1_curr - self._last_v1
            # 对齐维度
            min_len = min(len(v1_curr), len(self._last_v1))
            result['temporal_v1_err'] = float(
                np.sum(diff[:min_len] ** 2) / max(min_len, 1))

        # 层级反馈预测误差: ||V1 - feedback(V2)||²
        # feedback ≈ V2 的前 n 维 (粗略但生物合理 — 反馈连接保留低维结构)
        if self._last_v2 is not None and v2_curr is not None:
            n_dims = min(len(v1_curr), len(v2_curr))
            # V2 粗粒度 → 上采样到 V1 维度 (简单复制/插值)
            v2_upsampled = np.zeros(len(v1_curr), dtype=np.float32)
            v2_upsampled[:n_dims] = v2_curr[:n_dims]
            err = np.sum((v1_curr - v2_upsampled) ** 2) / max(len(v1_curr), 1)
            result['feedback_v1_err'] = float(err)

        if self._last_v4 is not None and v2_curr is not None:
            n_dims = min(len(v2_curr), len(v4_curr))
            v4_up = np.zeros(len(v2_curr), dtype=np.float32)
            v4_up[:n_dims] = v4_curr[:n_dims]
            err = np.sum((v2_curr - v4_up) ** 2) / max(len(v2_curr), 1)
            result['feedback_v2_err'] = float(err)

        result['total'] = (result['temporal_v1_err']
                           + result['feedback_v1_err'] * 0.3
                           + result['feedback_v2_err'] * 0.2)
        return result

    # ================================================================
    # 图像预处理
    # ================================================================

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """图像预处理: RGB→灰度, resize, 归一化 [-1, 1]。"""
        if image.ndim == 3 and image.shape[2] >= 3:
            gray = (0.2989 * image[:, :, 0].astype(np.float32)
                    + 0.5870 * image[:, :, 1].astype(np.float32)
                    + 0.1140 * image[:, :, 2].astype(np.float32))
        elif image.ndim == 3:
            gray = image[:, :, 0].astype(np.float32)
        else:
            gray = image.astype(np.float32)

        if gray.shape != (self.image_size, self.image_size):
            from PIL import Image
            gray_uint8 = np.clip(gray, 0, 255).astype(np.uint8)
            pil_img = Image.fromarray(gray_uint8)
            pil_img = pil_img.resize((self.image_size, self.image_size),
                                     Image.LANCZOS)
            gray = np.array(pil_img, dtype=np.float32)

        gray = (gray / 255.0) * 2.0 - 1.0
        return gray.astype(np.float32)

    # ================================================================
    # Hebb 可塑性
    # ================================================================

    def _hebb_update(self, raw_means: np.ndarray):
        """Hebb 增益: gain += lr × (response - gain)。"""
        self.gains = ((1.0 - self.gain_lr) * self.gains
                      + self.gain_lr * raw_means)
        self.gains = np.clip(self.gains, 0.1, 5.0)

    def reset_gains(self):
        """重置增益到 1.0。"""
        self.gains = np.ones(self.n_filters, dtype=np.float32)
        self.n_encodes = 0

    # ================================================================
    # 诊断
    # ================================================================

    def get_gain_profile(self) -> dict:
        gains_2d = self.gains.reshape(self.n_scales, self.n_orientations)
        return {
            'gains': self.gains.copy(),
            'gains_2d': gains_2d,
            'mean_gain': float(np.mean(self.gains)),
            'std_gain': float(np.std(self.gains)),
            'n_encodes': self.n_encodes,
            'top_filters': np.argsort(self.gains)[-8:][::-1].tolist(),
            'bottom_filters': np.argsort(self.gains)[:8].tolist(),
        }

    def filter_info(self, idx: int) -> dict:
        scale_idx = idx // self.n_orientations
        orient_idx = idx % self.n_orientations
        return {
            'index': idx,
            'sigma': float(self.sigmas[scale_idx]),
            'theta_deg': float(np.degrees(self.thetas[orient_idx])),
            'gain': float(self.gains[idx]),
        }


# ================================================================
# 自测
# ================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  GaborFilterBank Test (grid pooling)")
    print("=" * 60)

    gfb = GaborFilterBank(image_size=64, grid_size=4)
    print(f"  Filters: {gfb.n_filters} ({gfb.n_scales} scales x "
          f"{gfb.n_orientations} orientations)")
    print(f"  Grid: {gfb.grid_size}x{gfb.grid_size} = "
          f"{gfb.grid_size*gfb.grid_size} cells")
    print(f"  Raw output dim: {gfb.raw_dim} "
          f"(32 x {gfb.grid_size*gfb.grid_size} x 2)")

    # 测试: 随机 vs 条纹
    rng = np.random.default_rng(42)
    noise = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    vec_n = gfb.encode(noise)

    stripe = np.zeros((64, 64, 3), dtype=np.uint8)
    for i in range(0, 64, 4):
        stripe[:, i:i + 2] = 255
    vec_s = gfb.encode(stripe)

    cos = np.dot(vec_n, vec_s) / (np.linalg.norm(vec_n)
                                   * np.linalg.norm(vec_s) + 1e-8)
    print(f"\n  Random noise:    shape={vec_n.shape}, norm={np.linalg.norm(vec_n):.4f}")
    print(f"  Vertical stripes: shape={vec_s.shape}, norm={np.linalg.norm(vec_s):.4f}")
    print(f"  Cosine(noise, stripes) = {cos:.4f} (lower = better separation)")

    # 相同图像 → 应完全一致
    vec_s2 = gfb.encode(stripe)
    cos2 = np.dot(vec_s, vec_s2)
    print(f"  Cosine(stripe1, stripe2) = {cos2:.4f} (~1.0 = deterministic)")

    # Hebb 增益
    print("\n  --- Hebb plasticity ---")
    for _ in range(50):
        gfb.encode(stripe, learn=True)
    profile = gfb.get_gain_profile()
    top = profile['top_filters'][0]
    info = gfb.filter_info(top)
    print(f"  After 50 vertical stripes:")
    print(f"    gain mean={profile['mean_gain']:.3f}, std={profile['std_gain']:.3f}")
    print(f"    Top filter: sigma={info['sigma']}, "
          f"theta={info['theta_deg']:.1f} deg, gain={info['gain']:.3f}")

    # 速度
    import time
    N = 100
    t0 = time.perf_counter()
    for _ in range(N):
        gfb.encode(noise, learn=False)
    t1 = time.perf_counter()
    ms = (t1 - t0) / N * 1000
    print(f"\n  Speed: {ms:.1f}ms/image ({N} images in {t1-t0:.2f}s)")

    print("  [PASS] All tests complete")
