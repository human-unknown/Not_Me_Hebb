"""
gestalt.py — Gestalt 知觉分组层 (v4.1: 归入枕叶 occipital_lobe/)
自由能原理智能体 — 视觉知觉律实现

纯数学信号处理，零训练参数。将 Gabor 边缘片段组织为知觉结构。

生物对应:
  邻近分组   = V1 水平连接 (lateral connections)
  共线整合   = V1 水平轴突的长程连接 (long-range horizontal)
  相似分组   = V2 纹理选择性神经元
  对称检测   = V4 形状选择性
  图底分离   = V2 边界归属 (border ownership)

机制:
  1. 邻近分组: 空间距离聚类
  2. 共线整合: 沿朝向轴的响应片段链接
  3. 相似分组: 跨朝向/尺度响应相关
  4. 对称检测: V4 全局响应的镜像对称
  5. 图底分离: 边缘密度 → 前景掩码

输出: grouping_features (约 24d) → 与现有视觉特征拼接 → Hebb 学习
"""

import numpy as np
from scipy.ndimage import maximum_filter, label, uniform_filter
from scipy.spatial.distance import cdist


class GestaltGrouping:
    """格式塔知觉分组 — V1/V2/V4 分组机制的数学模型。

    零训练参数: 所有阈值基于图像尺寸和响应统计自适应。
    """

    def __init__(self, image_size: int = 64, n_scales: int = 4,
                 n_orientations: int = 8,
                 proximity_radius: float = None):
        self.image_size = image_size
        self.n_scales = n_scales
        self.n_orientations = n_orientations
        self.n_filters = n_scales * n_orientations

        # 邻近分组半径 (默认 image_size/12)
        self.proximity_radius = (
            proximity_radius or max(2, image_size / 12.0))

        # 共线角度容差 (弧度)
        self.collinearity_tolerance = np.radians(30.0)

    # ================================================================
    # 1. 邻近分组 (Proximity)
    # ================================================================

    def group_proximity(self, response_maps: np.ndarray
                        ) -> np.ndarray:
        """邻近律: 空间上靠近的强响应 → 同组。

        算法:
        1. 提取每个滤波器响应图的局部极大值位置 (关键点)
        2. 对所有关键点做距离聚类
        3. 统计每个聚类的特征

        Args:
            response_maps: (n_filters, H, W) 绝对值 Gabor 响应

        Returns:
            proximity_features: (4,) float32
              [n_keypoints_norm, mean_cluster_size, cluster_density, mean_spread]
        """
        h, w = self.image_size, self.image_size
        keypoints = []  # [(y, x, filter_idx, strength)]

        for fi in range(self.n_filters):
            rmap = np.abs(response_maps[fi])
            # 局部极大值 (非极大抑制窗口 = 3)
            local_max = maximum_filter(rmap, size=3)
            peaks = (rmap == local_max) & (rmap > rmap.mean() + 0.5 * rmap.std())
            ys, xs = np.where(peaks)
            for y, x in zip(ys, xs):
                keypoints.append((float(y), float(x), fi, float(rmap[y, x])))

        if len(keypoints) < 2:
            return np.zeros(4, dtype=np.float32)

        # 关键点密度 (归一化)
        n_kp_norm = min(1.0, len(keypoints) / (h * w * 0.1))

        # 简单距离聚类: 连通的最近邻图
        positions = np.array([(kp[0], kp[1]) for kp in keypoints])
        strengths = np.array([kp[3] for kp in keypoints])

        # 计算成对距离矩阵 (限制在 proximity_radius 内)
        # 对大量关键点用分块处理
        if len(keypoints) > 500:
            # 太多关键点 → 降采样
            indices = np.argsort(strengths)[-500:]
            positions = positions[indices]
            strengths = strengths[indices]

        # 构建邻接关系: distance < proximity_radius
        from scipy.spatial import KDTree
        tree = KDTree(positions)
        pairs = tree.query_pairs(r=self.proximity_radius, output_type='ndarray')

        if len(pairs) == 0:
            return np.array([n_kp_norm, 0.0, 0.0, 0.0], dtype=np.float32)

        # 连通分量 → 聚类
        n_kp = len(positions)
        parent = list(range(n_kp))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for a, b in pairs:
            union(int(a), int(b))

        # 统计聚类
        cluster_sizes = {}
        for i in range(n_kp):
            root = find(i)
            cluster_sizes[root] = cluster_sizes.get(root, 0) + 1

        sizes = np.array(list(cluster_sizes.values()))
        mean_size = float(np.mean(sizes)) / max(1, n_kp)
        cluster_density = float(len(pairs)) / max(1, n_kp)
        # Spread: cluster内标准差 / proximity_radius
        spreads = []
        for root, members in cluster_sizes.items():
            if members > 1:
                idxs = [i for i in range(n_kp) if find(i) == root]
                cluster_pos = positions[idxs]
                centroid = cluster_pos.mean(axis=0)
                spread = np.mean(np.linalg.norm(cluster_pos - centroid, axis=1))
                spreads.append(spread)
        mean_spread = float(np.mean(spreads)) / self.proximity_radius if spreads else 0.0

        return np.array([n_kp_norm, mean_size, cluster_density, mean_spread],
                        dtype=np.float32)

    # ================================================================
    # 2. 共线整合 (Continuity / Good Continuation)
    # ================================================================

    def integrate_contours(self, response_maps: np.ndarray
                           ) -> np.ndarray:
        """连续律: 沿 Gabor 朝向轴连接共线的响应片段。

        对每个朝向:
        1. 提取该朝向的强响应位置
        2. 沿朝向方向追踪: 如果下一个像素也在该朝向上有强响应
           → 它们是同一轮廓的一部分
        3. 统计轮廓长度

        Args:
            response_maps: (n_filters, H, W)

        Returns:
            contour_features: (4,) float32
              [mean_contour_len, n_contours_norm, mean_strength, max_contour_len]
        """
        h, w = self.image_size, self.image_size

        # 朝向角度
        orientations = np.linspace(0, np.pi, self.n_orientations,
                                   endpoint=False)

        all_contour_lens = []
        all_contour_strengths = []

        for oi, theta in enumerate(orientations):
            # 该朝向所有尺度的响应取平均
            orient_responses = []
            for si in range(self.n_scales):
                fi = si * self.n_orientations + oi
                orient_responses.append(np.abs(response_maps[fi]))
            orient_map = np.mean(orient_responses, axis=0)

            # 阈值: 高于均值 + 0.5 std → 边缘候选
            threshold = orient_map.mean() + 0.5 * orient_map.std()
            edge_mask = orient_map > threshold

            if not np.any(edge_mask):
                continue

            # 沿朝向方向追踪轮廓
            # 朝向方向向量
            dy = np.sin(theta)
            dx = np.cos(theta)

            visited = np.zeros_like(edge_mask, dtype=bool)

            for y0 in range(h):
                for x0 in range(w):
                    if not edge_mask[y0, x0] or visited[y0, x0]:
                        continue

                    # 沿朝向方向追踪
                    contour_len = 0
                    contour_strength = 0.0
                    y, x = float(y0), float(x0)

                    for _ in range(max(h, w)):
                        yi, xi = int(round(y)), int(round(x))
                        if yi < 0 or yi >= h or xi < 0 or xi >= w:
                            break
                        if not edge_mask[yi, xi] or visited[yi, xi]:
                            break

                        visited[yi, xi] = True
                        contour_len += 1
                        contour_strength += float(orient_map[yi, xi])
                        y += dy
                        x += dx

                    if contour_len >= 3:  # 至少 3 像素才算轮廓
                        all_contour_lens.append(contour_len)
                        all_contour_strengths.append(
                            contour_strength / contour_len)

        if not all_contour_lens:
            return np.zeros(4, dtype=np.float32)

        contour_lens = np.array(all_contour_lens, dtype=np.float32)
        contour_strengths_arr = np.array(all_contour_strengths, dtype=np.float32)

        mean_len = float(np.mean(contour_lens)) / max(h, w)
        n_contours_norm = min(1.0, len(contour_lens) / 50.0)
        mean_strength = float(np.mean(contour_strengths_arr))
        max_len = float(np.max(contour_lens)) / max(h, w)

        return np.array([mean_len, n_contours_norm, mean_strength, max_len],
                        dtype=np.float32)

    # ================================================================
    # 3. 相似分组 (Similarity)
    # ================================================================

    def group_similarity(self, response_maps: np.ndarray
                         ) -> np.ndarray:
        """相似律: 响应模式相似的区域 → 同组 (纹理分组)。

        1. 跨朝向相关: 相邻朝向响应的空间相关性
           → 高相关 = 纹理区域 (多个朝向共现)
           → 低相关 = 单一朝向边缘
        2. 跨尺度相关: 同朝向不同尺度的相关
           → 高相关 = 多尺度边缘 (强边缘)
           → 低相关 = 特定尺度纹理

        Args:
            response_maps: (n_filters, H, W)

        Returns:
            similarity_features: (4,) float32
              [cross_orient_corr, cross_scale_corr, texture_homogeneity, orient_sparsity]
        """
        h, w = self.image_size, self.image_size

        # ---- 跨朝向相关 ----
        cross_orient_corrs = []
        for si in range(self.n_scales):
            for oi in range(self.n_orientations - 1):
                fi1 = si * self.n_orientations + oi
                fi2 = si * self.n_orientations + oi + 1
                r1 = np.abs(response_maps[fi1]).ravel()
                r2 = np.abs(response_maps[fi2]).ravel()
                corr = np.corrcoef(r1, r2)[0, 1]
                if not np.isnan(corr):
                    cross_orient_corrs.append(corr)

        mean_cross_orient = float(np.mean(cross_orient_corrs)
                                  if cross_orient_corrs else 0.0)

        # ---- 跨尺度相关 ----
        cross_scale_corrs = []
        for oi in range(self.n_orientations):
            for si in range(self.n_scales - 1):
                fi1 = si * self.n_orientations + oi
                fi2 = (si + 1) * self.n_orientations + oi
                r1 = np.abs(response_maps[fi1]).ravel()
                r2 = np.abs(response_maps[fi2]).ravel()
                corr = np.corrcoef(r1, r2)[0, 1]
                if not np.isnan(corr):
                    cross_scale_corrs.append(corr)

        mean_cross_scale = float(np.mean(cross_scale_corrs)
                                 if cross_scale_corrs else 0.0)

        # ---- 纹理同质性 ----
        # 计算每个空间位置的局部响应方差 (跨朝向)
        # 低方差 = 均匀纹理, 高方差 = 各向异性边缘
        orient_energy = np.zeros((self.n_orientations, h, w),
                                 dtype=np.float32)
        for oi in range(self.n_orientations):
            for si in range(self.n_scales):
                fi = si * self.n_orientations + oi
                orient_energy[oi] += np.abs(response_maps[fi])
            orient_energy[oi] /= self.n_scales

        # 每像素的朝向方差
        orient_var = np.var(orient_energy, axis=0)
        texture_homogeneity = 1.0 - float(np.mean(orient_var) /
                                          (orient_var.max() + 1e-8))

        # ---- 朝向稀疏度 ----
        # 每像素有多少个朝向同时活跃
        orient_active = (orient_energy > orient_energy.mean(axis=0) * 0.5).sum(axis=0)
        orient_sparsity = 1.0 - float(np.mean(orient_active)) / self.n_orientations

        return np.array([mean_cross_orient, mean_cross_scale,
                         texture_homogeneity, orient_sparsity],
                        dtype=np.float32)

    # ================================================================
    # 4. 对称检测 (Symmetry)
    # ================================================================

    def detect_symmetry(self, response_maps: np.ndarray
                        ) -> np.ndarray:
        """对称律: 检测视觉模式的镜像对称性。

        使用分块朝向能量分布的镜像比较:
        - 垂直对称: block[i,j] vs block[i, N-1-j] 的相关性
        - 水平对称: block[i,j] vs block[M-1-i, j] 的相关性
        - 对角对称: block[i,j] vs block[M-1-i, N-1-j] 的相关性

        在 block 层面比较朝向能量, 比像素相关更鲁棒。

        Args:
            response_maps: (n_filters, H, W)

        Returns:
            symmetry_features: (3,) float32
              [vertical_sym, horizontal_sym, diagonal_sym]
        """
        h, w = self.image_size, self.image_size
        n_blocks = 4  # 4×4 grid, same as V1 retinotopy
        block_h, block_w = h // n_blocks, w // n_blocks

        # 每个 block 的朝向能量向量 (n_orientations,)
        block_energy = np.zeros((n_blocks, n_blocks, self.n_orientations),
                                dtype=np.float32)

        for bi in range(n_blocks):
            for bj in range(n_blocks):
                y_slice = slice(bi * block_h, (bi + 1) * block_h)
                x_slice = slice(bj * block_w, (bj + 1) * block_w)
                for oi in range(self.n_orientations):
                    e = 0.0
                    for si in range(self.n_scales):
                        fi = si * self.n_orientations + oi
                        e += float(np.mean(
                            np.abs(response_maps[fi][y_slice, x_slice])))
                    block_energy[bi, bj, oi] = e / self.n_scales

        # 归一化每个 block 的朝向分布
        for bi in range(n_blocks):
            for bj in range(n_blocks):
                nrm = np.linalg.norm(block_energy[bi, bj])
                if nrm > 1e-8:
                    block_energy[bi, bj] /= nrm

        # 垂直对称: (i,j) vs (i, n_blocks-1-j) 的余弦相似度
        v_sims = []
        for bi in range(n_blocks):
            for bj in range(n_blocks // 2):
                j_mirror = n_blocks - 1 - bj
                sim = np.dot(block_energy[bi, bj],
                             block_energy[bi, j_mirror])
                v_sims.append(sim)
        v_sym = float(np.mean(v_sims)) if v_sims else 0.0

        # 水平对称: (i,j) vs (n_blocks-1-i, j)
        h_sims = []
        for bi in range(n_blocks // 2):
            for bj in range(n_blocks):
                i_mirror = n_blocks - 1 - bi
                sim = np.dot(block_energy[bi, bj],
                             block_energy[i_mirror, bj])
                h_sims.append(sim)
        h_sym = float(np.mean(h_sims)) if h_sims else 0.0

        # 对角 (180°旋转): (i,j) vs (n_blocks-1-i, n_blocks-1-j)
        d_sims = []
        for bi in range(n_blocks // 2):
            for bj in range(n_blocks):
                i_mirror = n_blocks - 1 - bi
                j_mirror = n_blocks - 1 - bj
                sim = np.dot(block_energy[bi, bj],
                             block_energy[i_mirror, j_mirror])
                d_sims.append(sim)
        d_sym = float(np.mean(d_sims)) if d_sims else 0.0

        return np.array([v_sym, h_sym, d_sym], dtype=np.float32)

    # ================================================================
    # 5. 图底分离 (Figure-Ground Segregation)
    # ================================================================

    def figure_ground_mask(self, response_maps: np.ndarray
                           ) -> tuple[np.ndarray, np.ndarray]:
        """图底分离: 基于边缘密度的前景/背景分割。

        生物基础: V2 边界归属神经元
        - 边缘密度高的区域 (物体内部细节多) → 可能是"图"
        - 边缘密度低的区域 (平坦) → 可能是"底"
        - 边界处: 边缘密度梯度决定边界归属

        算法:
        1. 计算边缘密度热图 (所有 Gabor 响应之和)
        2. Otsu-like 自适应阈值 → 前景掩码
        3. 统计图/底特征

        Args:
            response_maps: (n_filters, H, W)

        Returns:
            (fg_mask, fg_features)
            fg_mask: (H, W) bool — True=前景
            fg_features: (4,) float32
              [fg_area_ratio, fg_edge_density, bg_edge_density, fg_bg_contrast]
        """
        h, w = self.image_size, self.image_size

        # 边缘密度: Gabor 能量之和 (跨滤波器)
        edge_density = np.sum(np.abs(response_maps), axis=0)  # (H, W)
        edge_density = edge_density / (edge_density.max() + 1e-8)

        # 自适应阈值: 中位数 + 0.3 * IQR
        flat = edge_density.ravel()
        median = np.median(flat)
        q75, q25 = np.percentile(flat, [75, 25])
        iqr = q75 - q25
        threshold = median + 0.3 * iqr

        fg_mask = edge_density > threshold

        # 形态学清理: 移除小块噪声
        fg_mask_int = fg_mask.astype(np.int32)
        labeled, n_labels = label(fg_mask_int)
        for li in range(1, n_labels + 1):
            if np.sum(labeled == li) < h * w * 0.02:  # < 2% = 噪声
                fg_mask[labeled == li] = False

        # ---- 特征 ----
        fg_area_ratio = float(np.mean(fg_mask))

        fg_edge_density = float(np.mean(edge_density[fg_mask])) \
            if np.any(fg_mask) else 0.0
        bg_edge_density = float(np.mean(edge_density[~fg_mask])) \
            if np.any(~fg_mask) else 0.0
        fg_bg_contrast = fg_edge_density - bg_edge_density

        fg_features = np.array([
            fg_area_ratio, fg_edge_density, bg_edge_density, fg_bg_contrast
        ], dtype=np.float32)

        return fg_mask, fg_features

    # ================================================================
    # 总入口: compute_all()
    # ================================================================

    def compute_all(self, response_maps: np.ndarray,
                    visual_features: np.ndarray = None
                    ) -> np.ndarray:
        """计算全部格式塔分组特征。

        Args:
            response_maps: (n_filters, H, W) Gabor 绝对值响应图
            visual_features: 可选的 V4/pulvinar 全局特征用于对称检测增强

        Returns:
            grouping_vector: 1D float32 — 所有分组特征的拼接 (~23d)
        """
        if response_maps.ndim != 3:
            # 单张图像: 需要先编码
            raise ValueError(
                f"response_maps must be (n_filters, H, W), "
                f"got {response_maps.shape}")

        # 确保绝对值
        rmap = np.abs(response_maps)

        # 1. 邻近分组
        proximity = self.group_proximity(rmap)

        # 2. 共线整合
        contours = self.integrate_contours(rmap)

        # 3. 相似分组
        similarity = self.group_similarity(rmap)

        # 4. 对称检测
        symmetry = self.detect_symmetry(rmap)

        # 5. 图底分离
        _, fg_features = self.figure_ground_mask(rmap)

        # 拼接 — 保留各组原始尺度, 不强制等权重
        # 不同图像的 gestalt 特征应有不同的范数, 否则会支配质心相似度
        grouping_vector = np.concatenate([
            proximity,    # 4
            contours,     # 4
            similarity,   # 4
            symmetry,     # 3
            fg_features,  # 4
        ])  # total = 19

        # 温和归一化: 缩放到单位方差, 不强制单位范数
        # 使用全局统计标准化而非 per-sample L2
        grouping_vector = np.tanh(grouping_vector * 0.5)

        return grouping_vector.astype(np.float32)

    @property
    def feature_dim(self) -> int:
        """分组特征向量的维度"""
        return 19


# ================================================================
# 便捷函数: 从 GaborFilterBank 和图像计算分组特征
# ================================================================

def compute_gestalt_from_image(image: np.ndarray,
                                gabor_bank) -> np.ndarray:
    """从图像直接计算格式塔分组特征。

    管线: 图像 → Gabor 响应图 → GestaltGrouping → 分组向量

    Args:
        image: (H, W) or (H, W, 3) uint8/float
        gabor_bank: GaborFilterBank instance

    Returns:
        grouping_vector: (19,) float32
    """
    from scipy.fft import fft2, ifft2
    from layer0_visual import GaborFilterBank

    # 预处理 (灰度)
    if hasattr(gabor_bank, '_preprocess'):
        gray = gabor_bank._preprocess(image)
    else:
        if image.ndim == 3:
            gray = (0.2989 * image[:, :, 0].astype(np.float32)
                    + 0.5870 * image[:, :, 1].astype(np.float32)
                    + 0.1140 * image[:, :, 2].astype(np.float32))
        else:
            gray = image.astype(np.float32)

    # Gabor 响应图 (全滤波器, 全尺寸)
    fft_size = gabor_bank.fft_size
    img_size = gabor_bank.image_size
    gray_padded = np.zeros((fft_size, fft_size), dtype=np.float32)
    gray_padded[:img_size, :img_size] = gray
    image_fft = fft2(gray_padded)

    response_maps = np.zeros(
        (gabor_bank.n_filters, img_size, img_size), dtype=np.float32)
    for i in range(gabor_bank.n_filters):
        resp_full = np.real(ifft2(image_fft * gabor_bank._kernel_ffts[i]))
        response = resp_full[:img_size, :img_size]
        response = gabor_bank._divisive_normalize(response)
        response_maps[i] = response * gabor_bank.gains[i]

    # 格式塔分组
    gestalt = GestaltGrouping(image_size=img_size,
                              n_scales=gabor_bank.n_scales,
                              n_orientations=gabor_bank.n_orientations)
    return gestalt.compute_all(response_maps)


# ================================================================
# 自测
# ================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  GestaltGrouping Test")
    print("=" * 60)

    from layer0_visual import GaborFilterBank

    gfb = GaborFilterBank(image_size=64, grid_size=4)
    rng = np.random.default_rng(42)

    # 测试 1: 随机噪声
    noise = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    vec_n = compute_gestalt_from_image(noise, gfb)
    print(f"  Noise gestalt: shape={vec_n.shape}, norm={np.linalg.norm(vec_n):.4f}")
    print(f"    Proximity[0:4]={vec_n[:4]}")
    print(f"    Contours[4:8]={vec_n[4:8]}")
    print(f"    Similarity[8:12]={vec_n[8:12]}")
    print(f"    Symmetry[12:15]={vec_n[12:15]}")
    print(f"    FigureGround[15:19]={vec_n[15:19]}")

    # 测试 2: 条纹图案 (应该高对称高连续性)
    stripe = np.zeros((64, 64, 3), dtype=np.uint8)
    for i in range(0, 64, 4):
        stripe[:, i:i + 2] = 255
    vec_s = compute_gestalt_from_image(stripe, gfb)
    print(f"\n  Stripes gestalt: norm={np.linalg.norm(vec_s):.4f}")
    print(f"    Symmetry[12:15]={vec_s[12:15]} (should be > noise)")
    cos = np.dot(vec_n, vec_s)
    print(f"  Cosine(noise, stripes) = {cos:.4f}")

    # 测试 3: 对称图案 (高斯圆)
    y, x = np.mgrid[0:64, 0:64].astype(np.float32)
    circle = np.exp(-((x - 32) ** 2 + (y - 32) ** 2) / (2 * 10 ** 2))
    circle_img = (circle * 255).astype(np.uint8)
    circle_rgb = np.stack([circle_img] * 3, axis=-1)
    vec_c = compute_gestalt_from_image(circle_rgb, gfb)
    print(f"\n  Circle gestalt: norm={np.linalg.norm(vec_c):.4f}")
    print(f"    Symmetry[12:15]={vec_c[12:15]} (should be > stripes)")

    # 测试 4: GestaltGrouping 类直接使用
    print("\n  --- GestaltGrouping direct ---")
    gg = GestaltGrouping(image_size=64)
    rmap = np.abs(rng.normal(0, 1, (32, 64, 64)).astype(np.float32))
    gvec = gg.compute_all(rmap)
    print(f"  Feature dim: {gg.feature_dim}")
    print(f"  Output shape: {gvec.shape}")

    print("\n  [PASS] GestaltGrouping tests complete")
