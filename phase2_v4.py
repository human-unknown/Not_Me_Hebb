"""
phase2_v4.py —— Phase 2 V4: Cluster of Clusters (Hierarchical Hebb Learning)
自由能原理智能体

V4 = "哪些 V1+V2 视觉模式同时出现？"

生物类比:
  V1:     朝向边缘 (4x4 网格)  → Gabor 简单细胞
  V2:     粗位置 + 角点 (2x2)  → 复杂细胞
  V4:     形状片段 + 曲率      → 多个 V1/V2 模式的共现

V4 不直接看像素。V4 的输入是「V1+V2 集群激活模式」:
  - 对每张图像，计算它与所有 V1+V2 集群的余弦相似度
  - 取 Top-K 最匹配的 V1+V2 集群
  - 构建一个 208 维的激活向量 (每个维度 = 一个 V1+V2 集群的相似度)
  - V4 ClusterNetwork 学习哪些 V1+V2 模式经常共现

这构建了真正的视觉层级: V1/V2(特征) → V4(形状) → IT(物体)

用法:
  python phase2_v4.py --n 10000
  python phase2_v4.py --n 2000 --top-k 8
"""

import os
import sys
import argparse
import time
import numpy as np
from collections import defaultdict

from data_types import D, Theta
from layer0_model import ClusterNetwork, sleep_cycle, _masked_cosine, _auto_mask
from layer3_meta import create_default_theta

# ================================================================
# 配置
# ================================================================

VISION_CHANNELS = {
    'v1v2_192': {'v1_start': 64, 'v1_width': 128, 'v2_start': 192, 'v2_width': 64},
    'v1v2_128': {'v1_start': 64, 'v1_width': 96,  'v2_start': 160, 'v2_width': 32},
}

V4_TOP_K = 10           # V4 输入: Top-K V1V2 集群
V4_SPARSITY = 0.10      # V4 输入稀疏度 (多少比例的 V1V2 集群被激活)
V4_THRESHOLD = 0.35     # V4 集群匹配阈值
V4_LEARN_RATE = 0.02    # V4 Hebb 学习率
V4_DECAY_RATE = 0.003


def build_v4_pattern(v1v2_net: ClusterNetwork, s_v1v2: np.ndarray,
                     top_k: int = V4_TOP_K,
                     sparsity: float = V4_SPARSITY) -> np.ndarray:
    """为一张图像构建 V4 输入模式。

    对每个 V1+V2 集群计算 cosine 相似度，保留 Top-K 激活，
    构建 (D,) 维 V4 感知向量 (放在 s[0:n_v1v2_clusters])。

    Args:
        v1v2_net: 训练好的 V1+V2 ClusterNetwork
        s_v1v2: V1+V2 感知向量 (D,) 用于 recall
        top_k: 保留多少个最匹配的 V1V2 集群
        sparsity: V4 向量中非零元素比例的上限

    Returns:
        (D,) V4 感知向量
    """
    n_clusters = v1v2_net.n_clusters
    if n_clusters == 0:
        return np.zeros(D, dtype=np.float32)

    h = np.tanh(s_v1v2 + 1e-8)
    mask = _auto_mask(s_v1v2)

    # 计算与所有 V1V2 集群的相似度
    sims = np.zeros(n_clusters, dtype=np.float32)
    for i, c in enumerate(v1v2_net.clusters):
        sims[i] = _masked_cosine(h, c.centroid, mask)

    # Top-K 激活
    top_indices = np.argsort(sims)[-top_k:]
    top_sims = sims[top_indices]

    # 构建 V4 感知向量: 激活模式放在 s[0:n_clusters]
    v4_s = np.zeros(D, dtype=np.float32)
    for idx, sim in zip(top_indices, top_sims):
        if sim > 0:  # 只保留正相似度
            v4_s[idx] = max(0.1, sim)  # 至少 0.1，确保非零

    # 额外: 也加入全局形状统计 (可选, 增强 V4)
    # 将 V1V2 的 vision 部分加权平均作为 global shape descriptor
    # 放在 s[256:320] (meta 通道)
    if top_k >= 3:
        weights = np.clip(top_sims[:3], 0, None)
        weights = weights / (weights.sum() + 1e-8)
        global_shape = np.zeros(64, dtype=np.float32)
        for idx, w in zip(top_indices[:3], weights):
            global_shape += w * v1v2_net.clusters[idx].centroid[64:128]
        v4_s[256:320] = global_shape

    return v4_s


def evaluate_v4(v4_net: ClusterNetwork, v1v2_net: ClusterNetwork,
                venv, vision_start: int, vision_end: int,
                top_k: int = V4_TOP_K) -> dict:
    """评估 V4 集群纯度。

    对每张图像:
    1. 构建 V1V2 感知向量
    2. 构建 V4 感知向量
    3. V4 recall → 最佳匹配 V4 集群
    4. 记录匹配
    """
    cluster_hits = defaultdict(list)  # v4_cluster_id -> [(idx, label, sim)]

    for idx in range(venv.n_images):
        # V1V2 sensory
        vis = venv.get_sensory(idx, include_v2=True)
        s_v1v2 = np.zeros(D, dtype=np.float32)
        end = min(vision_start + len(vis), D)
        vis_slice = vis[:end - vision_start]
        s_v1v2[vision_start:vision_start + len(vis_slice)] = vis_slice

        # V4 sensory
        s_v4 = build_v4_pattern(v1v2_net, s_v1v2, top_k=top_k)

        # V4 recall (只读)
        h = np.tanh(s_v4 + 1e-8)
        mask = _auto_mask(s_v4)
        best_sim, best_c = -1.0, None
        for c in v4_net.clusters:
            sim = _masked_cosine(h, c.centroid, mask)
            if sim > best_sim:
                best_sim, best_c = sim, c

        if best_c is not None:
            label = int(venv.labels[idx])
            cluster_hits[id(best_c)].append((idx, label, best_sim))

    # 计算纯度
    purities = {}
    for cid, hits in cluster_hits.items():
        if len(hits) < 3:
            continue
        class_counts = defaultdict(int)
        for _, label, _ in hits:
            class_counts[label] += 1
        total = len(hits)
        max_class = max(class_counts, key=class_counts.get)
        purities[cid] = {
            'total_hits': total,
            'purity': class_counts[max_class] / total,
            'top_class': max_class,
            'top_class_name': venv.label_names[max_class],
        }

    avg_purity = float(np.mean([d['purity'] for d in purities.values()])) if purities else 0.0
    hits_per_cluster = [d['total_hits'] for d in purities.values()]

    return {
        'n_clusters': v4_net.n_clusters,
        'n_active_clusters': len(purities),
        'avg_purity': avg_purity,
        'purities': purities,
        'mean_hits': np.mean(hits_per_cluster) if hits_per_cluster else 0,
        'total_images_assigned': sum(hits_per_cluster),
    }


def run_v4_experiment(n_images: int = 10000,
                      channel_config: str = 'v1v2_128',
                      v1v2_threshold: float = 0.50,
                      v1v2_lr: float = 0.02,
                      v4_top_k: int = V4_TOP_K,
                      v4_threshold: float = V4_THRESHOLD,
                      v4_lr: float = V4_LEARN_RATE,
                      seed: int = 42):
    """Phase 2 V4: 层级 Hebb 学习实验。

    管线:
    1. 加载 V1+V2 编码
    2. 训练 V1V2 ClusterNetwork
    3. 对每张图像构建 V4 激活模式
    4. 训练 V4 ClusterNetwork
    5. 比较 V1V2 vs V4 纯度
    """
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    ch = VISION_CHANNELS[channel_config]
    vision_start = ch['v1_start']
    vision_end = ch['v2_start'] + ch['v2_width']
    vision_width = vision_end - vision_start
    use_v2 = ch['v2_width'] > 0
    desc = f"V1({ch['v1_width']}d)+V2({ch['v2_width']}d)"

    print("=" * 64)
    print("  Phase 2 V4: Hierarchical Hebb Learning")
    print("=" * 64)
    print(f"  V1+V2: {desc}, {n_images} images")
    print(f"  V4: top-K={v4_top_k}, threshold={v4_threshold}, lr={v4_lr}")
    print()

    # ---- Stage 1: Load + Train V1V2 ----
    print("[Stage 1/4] Loading V1+V2 encodings...")
    from visual_interface import VisualEnvironment
    venv = VisualEnvironment(
        dataset='cifar10', n_images=n_images,
        pca_components=ch['v1_width'],
        v2_components=max(1, ch['v2_width']),
        use_v2=use_v2,
    )
    print(f"  V1: {venv.encodings.shape}, V2: {venv.encodings_v2.shape if venv.encodings_v2 is not None else 'N/A'}")

    print(f"\n[Stage 2/4] Training V1+V2 ClusterNetwork...")
    theta_v1v2 = create_default_theta()
    theta_v1v2.cluster_threshold = v1v2_threshold
    theta_v1v2.learn_rate_l0 = v1v2_lr
    theta_v1v2.decay_rate = 0.003
    v1v2_net = ClusterNetwork(theta_v1v2, hash_offset=vision_start)

    order = rng.permutation(n_images)
    t0 = time.perf_counter()
    for step, idx in enumerate(order):
        vis = venv.get_sensory(idx, include_v2=use_v2)
        s = np.zeros(D, dtype=np.float32)
        end = min(vision_start + len(vis), D)
        s[vision_start:vision_start + len(vis[:end - vision_start])] = vis[:end - vision_start]
        v1v2_net.learn(s)

        if (step + 1) % 1000 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  V1+V2: {step+1}/{n_images} images "
                  f"({(step+1)/max(elapsed,0.001):.0f} img/s), "
                  f"{v1v2_net.n_clusters} clusters")

        if (step + 1) % 500 == 0:
            sleep_cycle(v1v2_net, theta_v1v2)

    print(f"  V1+V2 trained: {v1v2_net.n_clusters} clusters "
          f"in {time.perf_counter()-t0:.1f}s")

    # ---- Stage 3: Build V4 patterns + Train V4 ----
    print(f"\n[Stage 3/4] Building V4 activation patterns + training V4...")
    theta_v4 = create_default_theta()
    theta_v4.cluster_threshold = v4_threshold
    theta_v4.learn_rate_l0 = v4_lr
    theta_v4.decay_rate = V4_DECAY_RATE
    v4_net = ClusterNetwork(theta_v4, hash_offset=0)

    # V4 输入空间的维度 = V1V2 集群数量
    v4_input_dim = v1v2_net.n_clusters
    active_ratio_v4 = min(1.0, v4_top_k / max(v4_input_dim, 1))
    eff_threshold_v4 = v4_threshold * (0.4 + 0.6 * active_ratio_v4)
    print(f"  V4 input dim: {v4_input_dim} (V1V2 clusters)")
    print(f"  V4 active_ratio: {active_ratio_v4:.3f}, "
          f"eff_threshold: {eff_threshold_v4:.3f}")

    t0 = time.perf_counter()
    for step, idx in enumerate(order):
        # Build V1V2 sensory
        vis = venv.get_sensory(idx, include_v2=use_v2)
        s_v1v2 = np.zeros(D, dtype=np.float32)
        end = min(vision_start + len(vis), D)
        vis_slice = vis[:end - vision_start]
        s_v1v2[vision_start:vision_start + len(vis_slice)] = vis_slice

        # Build V4 pattern
        s_v4 = build_v4_pattern(v1v2_net, s_v1v2, top_k=v4_top_k)

        # Train V4
        v4_net.learn(s_v4)

        if (step + 1) % 1000 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  V4: {step+1}/{n_images} "
                  f"({(step+1)/max(elapsed,0.001):.0f} img/s), "
                  f"{v4_net.n_clusters} clusters")

        if (step + 1) % 500 == 0:
            sleep_cycle(v4_net, theta_v4)

    print(f"  V4 trained: {v4_net.n_clusters} clusters "
          f"in {time.perf_counter()-t0:.1f}s")

    # ---- Stage 4: Evaluate ----
    print(f"\n[Stage 4/4] Evaluating V1+V2 vs V4...")

    # V1V2 evaluation
    v1v2_hits = defaultdict(list)
    for idx in range(n_images):
        vis = venv.get_sensory(idx, include_v2=use_v2)
        s = np.zeros(D, dtype=np.float32)
        end = min(vision_start + len(vis), D)
        vis_slice = vis[:end - vision_start]
        s[vision_start:vision_start + len(vis_slice)] = vis_slice
        h = np.tanh(s + 1e-8)
        mask = _auto_mask(s)
        best_sim, best_c = -1.0, None
        for c in v1v2_net.clusters:
            sim = _masked_cosine(h, c.centroid, mask)
            if sim > best_sim:
                best_sim, best_c = sim, c
        if best_c is not None:
            v1v2_hits[id(best_c)].append((idx, int(venv.labels[idx]), best_sim))

    # V4 evaluation
    v4_eval = evaluate_v4(v4_net, v1v2_net, venv, vision_start, vision_end,
                          top_k=v4_top_k)

    # ---- Report ----
    print("\n" + "=" * 64)
    print("  V4 Results: V1+V2 vs V4 Comparison")
    print("=" * 64)

    # Compute V1V2 purity
    v1v2_purities = []
    for cid, hits in v1v2_hits.items():
        if len(hits) < 3:
            continue
        class_counts = defaultdict(int)
        for _, label, _ in hits:
            class_counts[label] += 1
        max_count = max(class_counts.values())
        v1v2_purities.append(max_count / len(hits))
    v1v2_avg_purity = float(np.mean(v1v2_purities)) if v1v2_purities else 0.0

    print(f"\n  [Metrics]")
    print(f"    {'':<20} {'V1+V2':<15} {'V4':<15} {'Change':<10}")
    print(f"    {'-'*60}")
    print(f"    {'Clusters':<20} {v1v2_net.n_clusters:<15} "
          f"{v4_net.n_clusters:<15}")
    print(f"    {'Avg purity':<20} {v1v2_avg_purity:<15.3f} "
          f"{v4_eval['avg_purity']:<15.3f} "
          f"{'+' if v4_eval['avg_purity'] > v1v2_avg_purity else ''}"
          f"{v4_eval['avg_purity'] - v1v2_avg_purity:+.3f}")
    print(f"    {'Mean hits/cluster':<20} "
          f"{np.mean([len(h) for h in v1v2_hits.values()]):<15.0f} "
          f"{v4_eval['mean_hits']:<15.0f}")

    # Top-10 V4 clusters
    sorted_v4 = sorted(v4_eval['purities'].items(),
                      key=lambda x: -x[1]['purity'])
    print(f"\n  [Top-10] V4 Clusters:")
    print(f"    {'Rank':<5} {'Class':<12} {'Hits':<6} {'Purity':<8}")
    print(f"    {'-'*35}")
    for rank, (cid, data) in enumerate(sorted_v4[:10]):
        print(f"    {rank+1:<5} {data['top_class_name']:<12} "
              f"{data['total_hits']:<6} {data['purity']:.3f}")

    # Acceptance check
    print(f"\n  {'='*48}")
    print(f"  ACCEPTANCE CHECK")
    print(f"  {'='*48}")

    checks = []
    c1 = v4_net.n_clusters >= 10
    checks.append(('V4 >=10 clusters', c1, v4_net.n_clusters))
    c2 = v4_eval['avg_purity'] > v1v2_avg_purity
    checks.append(('V4 purity > V1+V2 purity', c2,
                   f"{v4_eval['avg_purity']:.3f} > {v1v2_avg_purity:.3f}"))
    c3 = v4_eval['avg_purity'] > 0.30
    checks.append(('V4 purity > 0.30', c3, f"{v4_eval['avg_purity']:.3f}"))

    for desc, passed, value in checks:
        status = '[PASS]' if passed else '[FAIL]'
        print(f"    {status} {desc}: {value}")

    all_pass = all(c[1] for c in checks)
    if all_pass:
        print(f"\n  *** ALL CHECKS PASSED ***")
    else:
        print(f"\n  {sum(1 for _,p,_ in checks if p)}/{len(checks)} checks passed")

    return {
        'v1v2_net': v1v2_net,
        'v4_net': v4_net,
        'v1v2_purity': v1v2_avg_purity,
        'v4_purity': v4_eval['avg_purity'],
        'all_checks_passed': all_pass,
    }


# ================================================================
# CLI
# ================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Phase 2 V4: Hierarchical Hebb Learning')
    parser.add_argument('--n', type=int, default=10000)
    parser.add_argument('--channel', type=str, default='v1v2_128')
    parser.add_argument('--v1v2-threshold', type=float, default=0.50)
    parser.add_argument('--v1v2-lr', type=float, default=0.02)
    parser.add_argument('--top-k', type=int, default=V4_TOP_K)
    parser.add_argument('--v4-threshold', type=float, default=V4_THRESHOLD)
    parser.add_argument('--v4-lr', type=float, default=V4_LEARN_RATE)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    results = run_v4_experiment(
        n_images=args.n,
        channel_config=args.channel,
        v1v2_threshold=args.v1v2_threshold,
        v1v2_lr=args.v1v2_lr,
        v4_top_k=args.top_k,
        v4_threshold=args.v4_threshold,
        v4_lr=args.v4_lr,
        seed=args.seed,
    )
