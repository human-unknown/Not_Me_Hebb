"""
attractors.py —— 吸引子发现与可视化 (M4)
自由能原理智能体

功能:
- pca_reduce(): PCA 降维到 2D
- cluster_features(): 基于距离的简单聚类
- track_attractors(): 聚类统计
- visualize_attractors(): 2 面板可视化仪表板
"""

import os
import time as _time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 颜色常量
C_BG = '#0a0e14'
C_CARD = '#131820'
C_BORDER = '#1e2a3a'
C_TEXT = '#c8ccd4'
C_DIM = '#7a8494'
CLUSTER_COLORS = ['#3b82f6', '#f59e0b', '#10b981', '#8b5cf6',
                  '#06b6d4', '#ef4444', '#f97316', '#84cc16',
                  '#ec4899', '#6366f1']

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# PCA 降维 (numpy SVD)
# ============================================================

def pca_reduce(features: np.ndarray, n_components: int = 2) -> np.ndarray:
    """PCA 降维 — 使用 numpy SVD

    Args:
        features: (n_samples, n_features) 特征矩阵（建议已 Z-score 归一化）
        n_components: 目标维度

    Returns:
        embeddings: (n_samples, n_components) 降维坐标
    """
    # 中心化
    X = features - features.mean(axis=0, keepdims=True)

    # SVD 分解
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    # 取前 n_components 个主成分
    components = Vt[:n_components]
    embeddings = X @ components.T

    # 归一化到 [-1, 1]
    emb_max = np.abs(embeddings).max(axis=0, keepdims=True) + 1e-8
    embeddings = embeddings / emb_max

    return embeddings


# ============================================================
# 聚类
# ============================================================

def cluster_features(embeddings: np.ndarray, eps: float = 0.12,
                     min_size: int = 3) -> np.ndarray:
    """基于距离的简单聚类

    算法:
    1. 计算 pairwise 欧氏距离矩阵
    2. 距离 < eps 的点互相连接
    3. 连通分量 = 聚类
    4. 小于 min_size 的聚类标记为噪声 (-1)

    Args:
        embeddings: (n, 2) 或 (n, d) 坐标
        eps: 聚类距离阈值
        min_size: 最小聚类大小

    Returns:
        labels: (n,) 聚类标签，-1 = 噪声
    """
    n = len(embeddings)
    if n == 0:
        return np.array([], dtype=int)

    # pairwise 距离矩阵
    dist = np.zeros((n, n))
    for i in range(n):
        diff = embeddings[i] - embeddings
        dist[i] = np.sqrt(np.sum(diff ** 2, axis=1))

    # 邻接矩阵
    adj = dist < eps
    np.fill_diagonal(adj, False)

    # 连通分量 (BFS)
    labels = np.full(n, -1, dtype=int)
    cluster_id = 0
    visited = set()

    for start in range(n):
        if start in visited:
            continue
        # BFS
        queue = [start]
        component = []
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            neighbors = np.where(adj[node])[0]
            for nb in neighbors:
                if nb not in visited:
                    queue.append(nb)

        if len(component) >= min_size:
            for node in component:
                labels[node] = cluster_id
            cluster_id += 1

    return labels


def track_attractors(embeddings: np.ndarray, labels: np.ndarray) -> dict:
    """聚类统计

    Returns:
        {n_clusters, labels, centers, sizes}
    """
    unique = sorted(set(labels) - {-1})
    centers = []
    sizes = []
    for cid in unique:
        mask = labels == cid
        centers.append(embeddings[mask].mean(axis=0))
        sizes.append(int(mask.sum()))

    return {
        'n_clusters': len(unique),
        'labels': labels,
        'centers': [c.tolist() for c in centers],
        'sizes': sizes,
    }


# ============================================================
# 可视化
# ============================================================

def visualize_attractors(embeddings: np.ndarray,
                         labels: np.ndarray,
                         thetas: list,
                         features: np.ndarray,
                         output_path: str = None,
                         dpi: int = 150) -> str:
    """2 面板吸引子可视化

    Panel 1: PCA 散点图（聚类着色 + 中心标注）
    Panel 2: 参数-行为相关性热力图

    Args:
        embeddings: (n, 2) PCA 坐标
        labels: (n,) 聚类标签
        thetas: Theta 列表
        features: (n, f) 特征矩阵
        output_path: 输出路径（自动生成）

    Returns:
        output_path
    """
    if output_path is None:
        os.makedirs('dashboards', exist_ok=True)
        ts = _time.strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join('dashboards',
                                   f'm4_attractors_{ts}.png')

    plt.rcParams.update({
        'figure.facecolor': C_BG, 'axes.facecolor': C_CARD,
        'axes.edgecolor': C_BORDER, 'axes.labelcolor': C_TEXT,
        'text.color': C_TEXT, 'xtick.color': C_DIM, 'ytick.color': C_DIM,
        'grid.color': C_BORDER, 'legend.facecolor': C_CARD,
        'legend.edgecolor': C_BORDER, 'legend.labelcolor': C_TEXT,
    })

    fig = plt.figure(figsize=(20, 10), dpi=dpi)
    gs = GridSpec(1, 2, figure=fig, hspace=0.3, wspace=0.30,
                  left=0.06, right=0.98, top=0.92, bottom=0.08)

    n_clusters = len(set(labels) - {-1})
    n_samples = len(embeddings)

    fig.suptitle(f'M4 Attractor Landscape — {n_samples} Theta Configs, '
                 f'{n_clusters} Clusters Found',
                 fontsize=16, fontweight='bold', color='#e8ecf2', y=0.97)

    # ---- Panel 1: PCA Scatter ----
    ax1 = fig.add_subplot(gs[0, 0])
    for cid in sorted(set(labels)):
        mask = labels == cid
        color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)] if cid >= 0 else C_DIM
        label = f'Cluster {cid}' if cid >= 0 else 'Noise'
        size = 80 if cid >= 0 else 30
        alpha = 0.8 if cid >= 0 else 0.3
        ax1.scatter(embeddings[mask, 0], embeddings[mask, 1],
                   c=color, s=size, alpha=alpha, label=label,
                   edgecolors='none')

    # 聚类中心
    for cid in sorted(set(labels) - {-1}):
        mask = labels == cid
        center = embeddings[mask].mean(axis=0)
        ax1.scatter(center[0], center[1], c='white', s=200, marker='X',
                   edgecolors=CLUSTER_COLORS[cid % len(CLUSTER_COLORS)],
                   linewidth=2, zorder=10)

    ax1.set_title('Panel 1: PCA Behavior Space', fontsize=13,
                 fontweight='bold', color='#e8ecf2', loc='left')
    ax1.set_xlabel('PC 1'); ax1.set_ylabel('PC 2')
    ax1.legend(fontsize=7, loc='upper right', ncol=2)
    ax1.grid(True, alpha=0.2, linestyle=':')

    # ---- Panel 2: Parameter-Behavior Heatmap ----
    ax2 = fig.add_subplot(gs[0, 1])

    # 选取 top 8 参数 (按与行为的方差关联)
    param_keys = ['sigma_z', 'sigma_x', 'w_body', 'w_social',
                  'w_cognitive', 'gamma', 'exploration_bonus', 'temperature']
    from features import FEATURE_NAMES
    feat_labels = [n[:12] for n in FEATURE_NAMES]

    # 构建相关性矩阵 (8 params × 13 features)
    n_p = len(param_keys)
    n_f = len(feat_labels)
    corr_mat = np.zeros((n_p, n_f))
    param_vals = np.array([[getattr(t, k, 0) for k in param_keys]
                           for t in thetas])

    for i in range(n_p):
        for j in range(n_f):
            if np.std(param_vals[:, i]) > 1e-8 and np.std(features[:, j]) > 1e-8:
                corr = np.corrcoef(param_vals[:, i], features[:, j])[0, 1]
                corr_mat[i, j] = corr

    im = ax2.imshow(corr_mat, aspect='auto', cmap='RdBu_r',
                   vmin=-1, vmax=1, interpolation='nearest')
    ax2.set_yticks(range(n_p))
    ax2.set_yticklabels(param_keys, fontsize=7, fontfamily='monospace')
    ax2.set_xticks(range(n_f))
    ax2.set_xticklabels(feat_labels, fontsize=6, rotation=45, ha='right')
    ax2.set_title('Panel 2: Parameter-Behavior Correlation', fontsize=13,
                 fontweight='bold', color='#e8ecf2', loc='left')
    ax2.tick_params(colors=C_DIM, labelsize=6)

    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.02)
    cbar.set_label('Pearson r', fontsize=7, color=C_DIM)
    cbar.ax.tick_params(colors=C_DIM, labelsize=6)

    # 保存
    fig.savefig(output_path, dpi=dpi, facecolor=C_BG, edgecolor='none',
                bbox_inches='tight')
    plt.close(fig)

    return output_path


# ============================================================
# 命令行测试
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  M4 Attractor Test")
    print("=" * 60)

    rng = np.random.default_rng(42)
    n = 30

    # 生成假的 2 聚类数据
    emb = np.vstack([
        rng.normal(-0.5, 0.2, (n // 2, 2)),
        rng.normal(0.5, 0.2, (n // 2, 2)),
    ])

    labels = cluster_features(emb, eps=0.5, min_size=3)
    att = track_attractors(emb, labels)

    print(f"  Clusters found: {att['n_clusters']}")
    print(f"  Sizes: {att['sizes']}")

    # 假特征和 thetas
    feats = np.random.randn(n, 13)
    from data_types import Theta
    thetas = [Theta() for _ in range(n)]

    path = visualize_attractors(emb, labels, thetas, feats)
    print(f"  Dashboard saved to: {path}")
