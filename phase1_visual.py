"""
phase1_visual.py —— Phase 1.5+2: V1 + V2 Visual Hierarchy
自由能原理智能体

Phase 1.5: Wider V1 features (128 PCA dims from 4x4 grid Gabor)
Phase 2: V2 features (64 PCA dims from 2x2 grid + cross-orientation interaction)

Combined pipeline:
  Image → Gabor V1 (4x4 grid, 1024d) → PCA → 128d V1 features
  Image → Gabor V2 (2x2 grid, 276d)  → PCA →  64d V2 features
  → s[64:192] = V1, s[192:256] = V2
  → ClusterNetwork with vision-aware hashing

V2 is biologically motivated:
  - 2x2 grid = larger receptive fields (32x32 vs 16x16 pixels)
  - Cross-orientation interaction = "corner detection"
  - Orientation contrast = "how oriented is this region"
  → More position-invariant, shape-sensitive representation

Usage:
  python phase1_visual.py                  # Default: 10k images, V1+V2
  python phase1_visual.py --n 2000         # Quick test
  python phase1_visual.py --v1-only         # V1 only (Phase 1.5 baseline)
  python phase1_visual.py --explore         # Interactive explorer
"""

import os
import sys
import argparse
import time
import numpy as np
from collections import defaultdict

from data_types import D, Theta
from layer0_model import ClusterNetwork, sleep_cycle
from layer3_meta import create_default_theta

# ================================================================
# 通道配置
# ================================================================

# 视觉通道在 s 向量中的布局 (D=330)
VISION_CHANNELS = {
    # Phase 1.5 (V1 only, wider): V1 128d at s[64:192]
    'v1_128':  {'v1_start': 64, 'v1_width': 128, 'v2_start': 192, 'v2_width': 0,
                'v4_start': 192, 'v4_width': 0},
    # Phase 1.5+2 (V1+V2): V1 128d + V2 64d = 192d at s[64:256]
    'v1v2_192': {'v1_start': 64, 'v1_width': 128, 'v2_start': 192, 'v2_width': 64,
                 'v4_start': 256, 'v4_width': 0},
    # Phase 1.5+2 compact: V1 96d + V2 32d = 128d at s[64:192]
    'v1v2_128': {'v1_start': 64, 'v1_width': 96, 'v2_start': 160, 'v2_width': 32,
                 'v4_start': 192, 'v4_width': 0},
    # Phase 2 V4: V1 96d + V2 32d + V4 72d = 200d at s[64:264]
    'v1v2v4_200': {'v1_start': 64, 'v1_width': 96, 'v2_start': 160, 'v2_width': 32,
                   'v4_start': 192, 'v4_width': 72},
}

# 视觉 Hebb 聚类参数
VISUAL_THRESHOLD = 0.50     # 视觉匹配阈值
VISUAL_LEARN_RATE = 0.02    # 视觉学习率 (比 Phase 1 更低, 更稳定)
VISUAL_DECAY_RATE = 0.003   # 视觉衰减率


# ================================================================
# 视觉相似类别分组 (V1 层级限制)
# ================================================================

VISUAL_SUBGROUPS = {
    'ground_vehicle': [1, 9],      # automobile, truck
    'aircraft':       [0],         # airplane
    'watercraft':     [8],         # ship
    'hoofed':         [4, 7],      # deer, horse
    'pet':            [3, 5],      # cat, dog
    'small_animal':   [2, 6],      # bird, frog
}


def _group_label(label: int) -> int:
    for gid, members in VISUAL_SUBGROUPS.items():
        if label in members:
            return hash(gid) % 100
    return label


# ================================================================
# 集群评估
# ================================================================

class ClusterEvaluator:
    """只读评估: 使用 argmax 余弦相似度 (不经过阈值筛选)"""

    def __init__(self, net: ClusterNetwork, venv,
                 vision_start: int, vision_end: int):
        self.net = net
        self.venv = venv
        self.vision_start = vision_start
        self.vision_end = vision_end
        self.cluster_hits: dict[int, list] = defaultdict(list)
        self.weak_hits: list = []

    def _build_sensory(self, idx: int) -> np.ndarray:
        """构建感知向量: vision 特征填入指定通道，其余为零"""
        vis = self.venv.get_sensory(idx, include_v2=True)
        s = np.zeros(D, dtype=np.float32)
        end = min(self.vision_start + len(vis), D)
        vis_slice = vis[:end - self.vision_start]
        s[self.vision_start:self.vision_start + len(vis_slice)] = vis_slice
        return s

    def _best_match(self, s: np.ndarray) -> tuple:
        """只读最佳匹配"""
        if not self.net.clusters:
            return None, -1.0
        from layer0_model import _masked_cosine, _auto_mask
        h = np.tanh(s + 1e-8)
        mask = _auto_mask(s)
        best_sim, best_c = -1.0, None
        for c in self.net.clusters:
            sim = _masked_cosine(h, c.centroid, mask)
            if sim > best_sim:
                best_sim, best_c = sim, c
        return best_c, best_sim

    def evaluate_image(self, idx: int) -> dict:
        s = self._build_sensory(idx)
        label = int(self.venv.labels[idx])
        best_c, best_sim = self._best_match(s)
        if best_c is not None:
            cid = id(best_c)
            self.cluster_hits[cid].append((idx, label, best_sim))
            return {'hit': True, 'cluster_id': cid, 'label': label,
                    'similarity': best_sim}
        return {'hit': False, 'label': label, 'similarity': 0.0}

    def cluster_purity(self, min_hits: int = 3) -> dict:
        results = {}
        for cid, hits in self.cluster_hits.items():
            if len(hits) < min_hits:
                continue
            class_counts = defaultdict(int)
            group_counts = defaultdict(int)
            for _, label, _ in hits:
                class_counts[label] += 1
                group_counts[_group_label(label)] += 1

            total = len(hits)
            max_class = max(class_counts, key=class_counts.get)
            max_count = class_counts[max_class]
            purity = max_count / total

            max_group = max(group_counts, key=group_counts.get)
            relaxed_purity = group_counts[max_group] / total

            entropy = sum(-(c/total)*np.log(c/total) for c in class_counts.values() if c > 0)
            n_classes = len(class_counts)
            norm_entropy = entropy / max(np.log(n_classes), 1e-8)

            results[cid] = {
                'total_hits': total, 'top_class': max_class,
                'top_class_name': self.venv.label_names[max_class],
                'top_count': max_count, 'purity': purity,
                'relaxed_purity': relaxed_purity,
                'norm_entropy': norm_entropy,
                'n_classes_present': n_classes,
            }
        return results

    def coverage_report(self, n_samples: int = 500) -> dict:
        n = min(n_samples, self.venv.n_images)
        indices = np.random.choice(self.venv.n_images, n, replace=False)
        hits, avg_sims = 0, []
        per_class = defaultdict(lambda: {'hits': 0, 'total': 0})
        for idx in indices:
            label = int(self.venv.labels[idx])
            s = self._build_sensory(idx)
            best_c, best_sim = self._best_match(s)
            per_class[label]['total'] += 1
            if best_c is not None:
                hits += 1
                per_class[label]['hits'] += 1
                avg_sims.append(best_sim)

        coverage = hits / n
        return {
            'n_samples': n, 'coverage': coverage,
            'avg_similarity': float(np.mean(avg_sims)) if avg_sims else 0.0,
            'per_class': {self.venv.label_names[l]: s['hits'] / max(s['total'], 1)
                         for l, s in per_class.items()},
        }


def confusion_report(net: ClusterNetwork, venv,
                     vision_start: int, vision_end: int,
                     n_per_class: int = 50) -> dict:
    """只读混淆矩阵: 集群分配类别 vs 真实类别"""
    n_classes = venv.n_classes

    # 分配每个集群到最多的类别
    cluster_class_hits = defaultdict(lambda: defaultdict(int))
    cluster_total = defaultdict(int)

    for idx in range(venv.n_images):
        vis = venv.get_sensory(idx, include_v2=True)
        s = np.zeros(D, dtype=np.float32)
        end = min(vision_start + len(vis), D)
        vis_slice = vis[:end - vision_start]
        s[vision_start:vision_start + len(vis_slice)] = vis_slice
        label = int(venv.labels[idx])

        from layer0_model import _masked_cosine, _auto_mask
        h = np.tanh(s + 1e-8)
        mask = _auto_mask(s)
        best_sim, best_cid = -1.0, None
        for c in net.clusters:
            sim = _masked_cosine(h, c.centroid, mask)
            if sim > best_sim:
                best_sim, best_cid = sim, id(c)

        if best_cid is not None:
            cluster_class_hits[best_cid][label] += 1
            cluster_total[best_cid] += 1

    cluster_assigned = {
        cid: max(counts, key=counts.get)
        for cid, counts in cluster_class_hits.items()
        if cluster_total[cid] > 0
    }

    # 计算混淆矩阵
    matrix = np.zeros((n_classes, n_classes), dtype=np.float32)
    counts = np.zeros(n_classes, dtype=np.int32)

    for c in range(n_classes):
        indices = np.where(venv.labels == c)[0]
        if len(indices) == 0:
            continue
        sample = np.random.choice(indices, min(n_per_class, len(indices)),
                                  replace=False)
        for idx in sample:
            vis = venv.get_sensory(idx, include_v2=True)
            s = np.zeros(D, dtype=np.float32)
            end = min(vision_start + len(vis), D)
            vis_slice = vis[:end - vision_start]
            s[vision_start:vision_start + len(vis_slice)] = vis_slice

            h = np.tanh(s + 1e-8)
            mask = (np.abs(s) > 1e-6)
            best_sim, best_cid = -1.0, None
            for cl in net.clusters:
                sim = _masked_cosine(h, cl.centroid, mask)
                if sim > best_sim:
                    best_sim, best_cid = sim, id(cl)

            counts[c] += 1
            if best_cid is not None and best_cid in cluster_assigned:
                matrix[c, cluster_assigned[best_cid]] += 1

    for c in range(n_classes):
        if counts[c] > 0:
            matrix[c] /= counts[c]

    result = {}
    for c, name in enumerate(venv.label_names):
        result[name] = {
            venv.label_names[o]: float(matrix[c, o])
            for o in range(n_classes)
        }
        result[name]['_diagonal'] = float(matrix[c, c])

    return {
        'per_class': result,
        'avg_diagonal': float(np.mean([matrix[c, c] for c in range(n_classes)])),
    }


# ================================================================
# 主实验
# ================================================================

def run_phase15(n_images: int = 10000,
                dataset: str = 'cifar10',
                channel_config: str = 'v1v2_192',
                cluster_threshold: float = VISUAL_THRESHOLD,
                learn_rate: float = VISUAL_LEARN_RATE,
                decay_rate: float = VISUAL_DECAY_RATE,
                sleep_interval: int = 500,
                eval_interval: int = 1000,
                seed: int = 42):
    """Phase 1.5+2: V1 + V2 视觉层级实验。

    Args:
        n_images: CIFAR-10 图像数量
        channel_config: 通道配置 ('v1_128', 'v1v2_192', 'v1v2_128')
        cluster_threshold: 集群匹配阈值
        learn_rate: Hebb 学习率
        decay_rate: 集群衰减率
        sleep_interval: 睡眠巩固间隔
        eval_interval: 评估间隔
        seed: 随机种子
    """
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    ch = VISION_CHANNELS[channel_config]
    vision_start = ch['v1_start']
    vision_end = max(ch['v2_start'] + ch['v2_width'],
                     ch.get('v4_start', 0) + ch.get('v4_width', 0))
    vision_width = vision_end - vision_start
    use_v2 = ch['v2_width'] > 0
    use_v4 = ch.get('v4_width', 0) > 0
    description = f"V1({ch['v1_width']}d)"
    if use_v2:
        description += f" + V2({ch['v2_width']}d)"
    if use_v4:
        description += f" + V4({ch['v4_width']}d)"

    print("=" * 64)
    print(f"  Phase 1.5+2: V1 + V2 Visual Hierarchy")
    print("=" * 64)
    print(f"  Config: {n_images} images, {description}")
    print(f"  Vision channel: s[{vision_start}:{vision_end}] = {vision_width}d")
    print(f"  threshold={cluster_threshold}, lr={learn_rate}, "
          f"decay={decay_rate}")
    print()

    # ---- 1. 加载视觉环境 ----
    print("[1/5] Loading VisualEnvironment...")
    t0 = time.perf_counter()
    from visual_interface import VisualEnvironment

    v1_components = ch['v1_width']
    v2_components = ch['v2_width']
    venv = VisualEnvironment(
        dataset=dataset, n_images=n_images,
        pca_components=v1_components,
        v2_components=max(1, v2_components),
        use_v2=use_v2,
        use_v4=use_v4,
    )
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s")
    print(f"  V1 encodings: {venv.encodings.shape}")
    if use_v2 and venv.encodings_v2 is not None:
        print(f"  V2 encodings: {venv.encodings_v2.shape}")

    # ---- 2. 评估特征质量 ----
    print(f"\n[2/5] Feature quality assessment...")

    # Compute combined encodings for class separation
    encs = venv.encodings
    if use_v2 and venv.encodings_v2 is not None:
        encs = np.concatenate([venv.encodings, venv.encodings_v2], axis=1)

    # 快速类间分离度
    class_centroids = {}
    for i, name in enumerate(venv.label_names):
        mask = venv.labels == i
        if mask.sum() > 0:
            class_centroids[name] = encs[mask].mean(axis=0)

    margins = {}
    for name, centroid in class_centroids.items():
        mask = venv.labels == venv.label_names.index(name)
        class_encs = encs[mask]
        intra = float(np.mean(np.dot(class_encs, centroid) / (
            np.linalg.norm(class_encs, axis=1) * np.linalg.norm(centroid) + 1e-8)))
        inter_sims = []
        for other_name, other_centroid in class_centroids.items():
            if other_name != name:
                inter_sims.append(float(np.dot(centroid, other_centroid) / (
                    np.linalg.norm(centroid) * np.linalg.norm(other_centroid) + 1e-8)))
        margins[name] = intra - float(np.mean(inter_sims))

    avg_margin = float(np.mean(list(margins.values())))
    print(f"  Class separation margins (V1+V2 combined):")
    for name in sorted(margins, key=lambda x: -margins[x]):
        bar = '#' * int(max(0, margins[name] * 30)) + '-' * int(max(0, 30 - max(0, margins[name] * 30)))
        print(f"    {name:<12s}: {margins[name]:+.3f} {bar}")
    print(f"    {'-'*36}")
    print(f"    Avg margin: {avg_margin:+.3f}")

    # ---- 3. 创建 ClusterNetwork ----
    print(f"\n[3/5] Creating ClusterNetwork (vision-tuned)...")
    theta = create_default_theta()
    theta.cluster_threshold = cluster_threshold
    theta.learn_rate_l0 = learn_rate
    theta.decay_rate = decay_rate

    active_ratio = vision_width / D
    eff_threshold = cluster_threshold * (0.4 + 0.6 * active_ratio)
    net = ClusterNetwork(theta, hash_offset=vision_start)

    print(f"  vision_width={vision_width}, active_ratio={active_ratio:.3f}")
    print(f"  effective_threshold={eff_threshold:.3f}")
    print(f"  hash_offset={vision_start}")
    print(f"  max clusters K=256")

    # ---- 4. 喂入图像 ----
    print(f"\n[4/5] Feeding {n_images} images...")
    order = rng.permutation(n_images)
    history = []
    t_start = time.perf_counter()

    for i, idx in enumerate(order):
        vis = venv.get_sensory(idx, include_v2=use_v2)
        s = np.zeros(D, dtype=np.float32)
        end = min(vision_start + len(vis), D)
        s[vision_start:vision_start + len(vis[:end - vision_start])] = vis[:end - vision_start]
        net.learn(s)

        if (i + 1) % eval_interval == 0:
            elapsed = time.perf_counter() - t_start
            ips = (i + 1) / max(elapsed, 0.001)
            evaluator = ClusterEvaluator(net, venv, vision_start, vision_end)
            eval_indices = rng.choice(n_images, min(500, n_images), replace=False)
            for eidx in eval_indices:
                evaluator.evaluate_image(eidx)
            purity_data = evaluator.cluster_purity(min_hits=2)
            purities = [d['purity'] for d in purity_data.values()]
            avg_purity = float(np.mean(purities)) if purities else 0.0
            n_stable = sum(1 for c in net.clusters if c.activation > 0.05)
            top_act = max((c.activation for c in net.clusters), default=0.0)

            history.append({
                'step': i + 1, 'n_clusters': net.n_clusters,
                'n_stable': n_stable, 'avg_purity': avg_purity,
                'top_activation': top_act, 'elapsed_s': elapsed,
                'images_per_sec': ips,
            })

            pct = (i + 1) / n_images * 100
            filled = int(30 * (i + 1) / n_images)
            bar = '#' * filled + '-' * (30 - filled)
            print(f"  [{bar}] {pct:5.1f}% | step={i+1:5d} | "
                  f"clusters={net.n_clusters:3d} stable={n_stable:3d} | "
                  f"purity={avg_purity:.3f} | {ips:.0f} img/s")

        if (i + 1) % sleep_interval == 0:
            n_before = net.n_clusters
            n_removed = sleep_cycle(net, theta)
            if n_removed > 0:
                print(f"  [Sleep] {n_before} -> {net.n_clusters} "
                      f"clusters (removed {n_removed})")

    total_time = time.perf_counter() - t_start
    print(f"\n  Feeding complete: {n_images} images in {total_time:.1f}s "
          f"({n_images / total_time:.0f} img/s)")

    # ---- 5. 最终评估 ----
    print(f"\n[5/5] Final evaluation...")
    evaluator = ClusterEvaluator(net, venv, vision_start, vision_end)

    print(f"  Evaluating all {n_images} images (read-only argmax match)...")
    for idx in range(n_images):
        evaluator.evaluate_image(idx)

    purity_data = evaluator.cluster_purity(min_hits=3)
    purities = [d['purity'] for d in purity_data.values()]
    relaxed_purities = [d['relaxed_purity'] for d in purity_data.values()]
    avg_purity = float(np.mean(purities)) if purities else 0.0
    avg_relaxed = float(np.mean(relaxed_purities)) if relaxed_purities else 0.0

    stability = {
        'n_stable': sum(1 for c in net.clusters if c.activation > 0.05),
        'n_clusters': net.n_clusters,
    }

    coverage = evaluator.coverage_report(n_samples=1000)
    confusion = confusion_report(net, venv, vision_start, vision_end,
                                 n_per_class=50)

    # ---- 报告 ----
    print("\n" + "=" * 64)
    print(f"  Phase 1.5+2 Results ({description})")
    print("=" * 64)

    print(f"\n  [Stats] Cluster Statistics:")
    print(f"    Feature margin:        {avg_margin:+.3f}")
    print(f"    Total clusters:        {net.n_clusters}")
    print(f"    Stable (act>0.05):     {stability['n_stable']}")
    print(f"    Avg purity (strict):   {avg_purity:.3f}")
    print(f"    Avg purity (relaxed):  {avg_relaxed:.3f}")
    print(f"    Coverage:              {coverage['coverage']:.1%}")
    print(f"    Avg match similarity:  {coverage.get('avg_similarity', 0):.3f}")

    # Top-15
    sorted_pure = sorted(purity_data.items(),
                        key=lambda x: x[1]['purity'], reverse=True)
    print(f"\n  [Top-15] Purest Clusters:")
    print(f"    {'Rank':<5} {'Class':<12} {'Hits':<6} {'Purity':<8} "
          f"{'Relaxed':<8} {'Top3 Classes'}")
    print(f"    {'-'*65}")
    for rank, (cid, data) in enumerate(sorted_pure[:15]):
        class_counts = defaultdict(int)
        for _, label, _ in evaluator.cluster_hits[cid]:
            class_counts[label] += 1
        top3 = sorted(class_counts.items(), key=lambda x: -x[1])[:3]
        top3_str = '+'.join(venv.label_names[c][:4] for c, _ in top3)
        print(f"    {rank+1:<5} {data['top_class_name']:<12} "
              f"{data['total_hits']:<6} {data['purity']:.3f}    "
              f"{data['relaxed_purity']:.3f}    {top3_str}")

    # 混淆矩阵
    print(f"\n  [Recall] Per-class diagonal:")
    diag = [(name, confusion['per_class'][name]['_diagonal'])
            for name in venv.label_names]
    diag.sort(key=lambda x: -x[1])
    for name, d in diag:
        bar = '#' * int(d * 20) + '-' * (20 - int(d * 20))
        print(f"    {name:<12s}: {d:.3f} {bar}")
    print(f"    {'-'*36}")
    print(f"    Avg diagonal: {confusion['avg_diagonal']:.3f}")

    # Per-class coverage
    print(f"\n  [Coverage] Per-class:")
    for name in venv.label_names:
        cr = coverage['per_class'].get(name, 0.0)
        bar = '#' * int(cr * 20) + '-' * (20 - int(cr * 20))
        print(f"    {name:<12s}: {cr:.2f} {bar}")

    # 集群大小分布
    cluster_sizes = [d['total_hits'] for d in purity_data.values()]
    if cluster_sizes:
        print(f"\n  [Size] Distribution:")
        print(f"    Mean={np.mean(cluster_sizes):.0f} "
              f"Median={np.median(cluster_sizes):.0f} "
              f"Max={max(cluster_sizes)} Min={min(cluster_sizes)}")

    # 验收检查
    print(f"\n  {'='*48}")
    print(f"  ACCEPTANCE CHECK")
    print(f"  {'='*48}")

    checks = []
    c1 = stability['n_stable'] >= 20
    checks.append(('>=20 stable clusters', c1, stability['n_stable']))
    c2 = avg_purity > 0.25
    checks.append(('Strict purity > 0.25', c2, f"{avg_purity:.3f}"))
    c3 = avg_relaxed > 0.40
    checks.append(('Relaxed purity > 0.40', c3, f"{avg_relaxed:.3f}"))
    c4 = coverage['coverage'] > 0.50
    checks.append(('Coverage > 50%', c4, f"{coverage['coverage']:.1%}"))
    c5 = confusion['avg_diagonal'] > 0.20
    checks.append(('Confusion diag > 0.20', c5,
                   f"{confusion['avg_diagonal']:.3f}"))

    for desc, passed, value in checks:
        status = '[PASS]' if passed else '[FAIL]'
        print(f"    {status} {desc}: {value}")

    all_pass = all(c[1] for c in checks)
    if all_pass:
        print(f"\n  *** ALL CHECKS PASSED ***")
    else:
        n_pass = sum(1 for _, p, _ in checks if p)
        print(f"\n  {n_pass}/{len(checks)} checks passed")

    return {
        'config': {
            'channel_config': channel_config,
            'description': description,
            'vision_width': vision_width,
            'cluster_threshold': cluster_threshold,
            'learn_rate': learn_rate,
        },
        'history': history,
        'avg_purity': avg_purity,
        'avg_relaxed': avg_relaxed,
        'avg_margin': avg_margin,
        'confusion': confusion,
        'coverage': coverage,
        'total_time_s': total_time,
        'all_checks_passed': all_pass,
        'net': net,
        'venv': venv,
    }


# ================================================================
# CLI
# ================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Phase 1.5+2: V1+V2 Visual Hierarchy')
    parser.add_argument('--n', type=int, default=10000)
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'imagenette'])
    parser.add_argument('--channel', type=str, default='v1v2_192',
                       choices=['v1_128', 'v1v2_192', 'v1v2_128',
                                'v1v2v4_200'])
    parser.add_argument('--threshold', type=float, default=VISUAL_THRESHOLD)
    parser.add_argument('--lr', type=float, default=VISUAL_LEARN_RATE)
    parser.add_argument('--decay', type=float, default=VISUAL_DECAY_RATE)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--v1-only', action='store_true',
                       help='V1 only (128d, no V2)')
    args = parser.parse_args()

    ch = 'v1_128' if args.v1_only else args.channel

    results = run_phase15(
        n_images=args.n,
        dataset=args.dataset,
        channel_config=ch,
        cluster_threshold=args.threshold,
        learn_rate=args.lr,
        decay_rate=args.decay,
        seed=args.seed,
    )
