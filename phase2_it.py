"""
phase2_it.py — Phase 2 IT: Hierarchical Hebb Learning with IT Layer
自由能原理智能体

V1→V2→V4→IT hierarchical visual processing with sequential and joint
training strategies. IT (inferotemporal cortex) pools across all lower
feature detectors to form object-level representations.

Architecture:
  V1: 4×4 grid Gabor edges (128d PCA) → position-sensitive
  V2: 2×2 grid + cross-orientation (64d PCA) → position-invariant
  V4: 1×1 global + curvature (64d PCA) → shape detection
  IT: ALL features + cluster summary (320d) → object identity

Each layer has its own ClusterNetwork. Higher layers see:
  - Raw Gabor features from ALL lower levels (geometric backbone)
  - Cluster Activation Summary (64d) — which lower clusters fire

Key improvements over phase2_v4 (failed "Cluster of Clusters"):
  - IT input uses FEATURE VECTORS, not cluster IDs → geometric structure
  - Cluster summary is a supplement, not the main input
  - Hierarchical training: freeze lower layers for stability

Usage:
  python phase2_it.py --n 2000 --mode both     # Compare seq vs joint
  python phase2_it.py --n 2000 --mode seq       # Sequential only
  python phase2_it.py --n 2000 --mode joint     # Joint only
"""

import os
import sys
import argparse
import time
import numpy as np
from collections import defaultdict

from data_types import D, Theta, Cluster
from layer0_model import ClusterNetwork, sleep_cycle, _masked_cosine, _auto_mask
from layer3_meta import create_default_theta


# ================================================================
# Configuration — Shared Sensory Layout (D=330)
# ================================================================

V1_WIDTH  = 128   # V1 Gabor PCA features
V2_WIDTH  = 64    # V2 Gabor PCA features
V4_WIDTH  = 64    # V4 Gabor PCA features (may be PCA'd from 72d)
CS_WIDTH  = 64    # Cluster Activation Summary

V1_START, V1_END = 10, 10 + V1_WIDTH          # s[10:138]
V2_START, V2_END = 138, 138 + V2_WIDTH         # s[138:202]
V4_START, V4_END = 202, 202 + V4_WIDTH         # s[202:266]
CS_START, CS_END  = 266, 266 + CS_WIDTH        # s[266:330]

# Per-layer Hebb parameters (higher layers = lower threshold = more convergence)
LAYER_PARAMS = {
    'v1': {'threshold': 0.50, 'lr': 0.02, 'decay': 0.003, 'hash_offset': V1_START},
    'v2': {'threshold': 0.45, 'lr': 0.02, 'decay': 0.003, 'hash_offset': V2_START},
    'v4': {'threshold': 0.40, 'lr': 0.02, 'decay': 0.003, 'hash_offset': V4_START},
    'it': {'threshold': 0.35, 'lr': 0.02, 'decay': 0.003, 'hash_offset': V1_START},
}

# Training
SLEEP_INTERVAL = 500
TOP_K_CLUSTER = 5    # top-K similarities per layer in cluster summary

# Visual subgroup definitions (for relaxed purity)
VISUAL_SUBGROUPS = {
    'ground_vehicle': [1, 9],      # automobile, truck
    'aircraft':       [0],         # airplane
    'watercraft':     [8],         # ship
    'hoofed':         [4, 7],      # deer, horse
    'pet':            [3, 5],      # cat, dog
    'small_animal':   [2, 6],      # bird, frog
}

# ImageNette visual subgroups (10 classes → 5 groups)
IMAGENETTE_SUBGROUPS = {
    'vehicle':    [6, 7],               # garbage_truck, gas_pump
    'animal':     [0, 1],               # tench, springer
    'device':     [2, 3],               # cassette_player, chain_saw
    'structure':  [4],                  # church
    'instrument': [5],                  # french_horn
    'sport':      [8, 9],               # golf_ball, parachute
}


def _group_label(label: int, n_classes: int) -> int:
    """Map label to visual subgroup for relaxed purity."""
    if n_classes <= 10:  # CIFAR-10
        for gid, members in VISUAL_SUBGROUPS.items():
            if label in members:
                return hash(gid) % 100
    else:
        for gid, members in IMAGENETTE_SUBGROUPS.items():
            if label in members:
                return hash(gid) % 100
    return label


# ================================================================
# Cluster Activation Computation
# ================================================================

def compute_cluster_sims(net: ClusterNetwork, features: np.ndarray,
                          start: int, end: int) -> np.ndarray:
    """Compute cosine similarity of features to all clusters in net.

    Builds a sensory vector with features placed at s[start:end],
    then computes masked cosine to each cluster's centroid.

    Args:
        net: Trained ClusterNetwork (frozen or not)
        features: Feature vector for this layer
        start, end: Where features go in sensory s[]

    Returns:
        (n_clusters,) array of cosine similarities
    """
    n = net.n_clusters
    if n == 0:
        return np.zeros(0, dtype=np.float32)

    s = np.zeros(D, dtype=np.float32)
    flen = min(len(features), end - start)
    s[start:start + flen] = features[:flen]
    h = np.tanh(s + 1e-8)
    mask = _auto_mask(s)

    sims = np.zeros(n, dtype=np.float32)
    for i, c in enumerate(net.clusters):
        sims[i] = _masked_cosine(h, c.centroid, mask)

    return sims


def build_cluster_summary(nets: dict, features_dict: dict,
                           stage: int) -> np.ndarray:
    """Build 64d cluster activation summary for higher layers.

    Encodes which lower-layer clusters are active:
      - Top-5 similarities per lower layer
      - Mean, max, std statistics per lower layer
      - 8 values per layer × up to 3 layers = 24 values → padded to 64d

    Args:
        nets: {'v1': net, 'v2': net, 'v4': net} — lower may be None
        features_dict: {'v1': vec, 'v2': vec, 'v4': vec}
        stage: 0=V1 training, 1=V2 training, 2=V4 training, 3=IT training

    Returns:
        (64,) float32, L2-normalized
    """
    summary = np.zeros(CS_WIDTH, dtype=np.float32)
    pos = 0

    layer_order = []
    if stage >= 1: layer_order.append(('v1', V1_START, V1_END))
    if stage >= 2: layer_order.append(('v2', V2_START, V2_END))
    if stage >= 3: layer_order.append(('v4', V4_START, V4_END))

    for layer_name, start, end in layer_order:
        net = nets.get(layer_name)
        feats = features_dict.get(layer_name)
        if net is None or net.n_clusters == 0 or feats is None:
            pos += 8
            continue

        sims = compute_cluster_sims(net, feats, start, end)
        if len(sims) == 0:
            pos += 8
            continue

        sorted_sims = np.sort(sims)[::-1]
        n_top = min(TOP_K_CLUSTER, len(sorted_sims))

        if pos + 8 <= CS_WIDTH:
            summary[pos:pos + n_top] = sorted_sims[:n_top]
            pos += TOP_K_CLUSTER
            summary[pos:pos + 3] = [
                float(np.mean(sims)), float(np.max(sims)), float(np.std(sims))]
            pos += 3
        else:
            break  # shouldn't happen with 3 layers × 8 values

    # L2 normalize
    norm = np.linalg.norm(summary)
    if norm > 1e-8:
        summary /= norm
    return summary.astype(np.float32)


# ================================================================
# Sensory Vector Construction
# ================================================================

def build_layer_sensory(features_dict: dict, nets: dict,
                         stage: int) -> np.ndarray:
    """Build D=330 sensory vector for training/evaluating a specific layer.

    Shared layout:
      s[10:138]  = V1 Gabor PCA features (128d)
      s[138:202] = V2 Gabor PCA features (64d)
      s[202:266] = V4 Gabor PCA features (64d)
      s[266:330] = Cluster Activation Summary (64d)

    The features present depend on the stage:
      Stage 0 (V1): only V1 features active, rest zero
      Stage 1 (V2): V1+V2 features + V1 cluster summary
      Stage 2 (V4): V1+V2+V4 features + V1+V2 cluster summary
      Stage 3 (IT): ALL features + V1+V2+V4 cluster summary

    Args:
        features_dict: {'v1': vec, 'v2': vec, 'v4': vec}
        nets: {'v1': net, 'v2': net, 'v4': net} — may have None entries
        stage: 0=V1, 1=V2, 2=V4, 3=IT

    Returns:
        (D,) float32 sensory vector
    """
    s = np.zeros(D, dtype=np.float32)

    # V1 features always present
    v1_feat = features_dict.get('v1')
    if v1_feat is not None:
        flen = min(len(v1_feat), V1_WIDTH)
        s[V1_START:V1_START + flen] = v1_feat[:flen]

    # V2 features present for stage >= 1
    if stage >= 1:
        v2_feat = features_dict.get('v2')
        if v2_feat is not None:
            flen = min(len(v2_feat), V2_WIDTH)
            s[V2_START:V2_START + flen] = v2_feat[:flen]

    # V4 features present for stage >= 2
    if stage >= 2:
        v4_feat = features_dict.get('v4')
        if v4_feat is not None:
            flen = min(len(v4_feat), V4_WIDTH)
            s[V4_START:V4_START + flen] = v4_feat[:flen]

    # Cluster summary present for stage >= 1
    if stage >= 1:
        cs = build_cluster_summary(nets, features_dict, stage)
        s[CS_START:CS_START + len(cs)] = cs

    return s


# ================================================================
# Network Factory
# ================================================================

def _create_net(layer_name: str) -> ClusterNetwork:
    """Create a ClusterNetwork tuned for a specific layer."""
    params = LAYER_PARAMS[layer_name]
    theta = create_default_theta()
    theta.cluster_threshold = params['threshold']
    theta.learn_rate_l0 = params['lr']
    theta.decay_rate = params['decay']
    return ClusterNetwork(theta, hash_offset=params['hash_offset'])


# ================================================================
# Read-only Evaluation
# ================================================================

def evaluate_layer(net: ClusterNetwork, venv,
                    nets: dict, stage: int,
                    n_images: int = None) -> dict:
    """Read-only evaluation of a single layer's clustering.

    For each image:
      1. Build sensory vector appropriate for this stage
      2. Find best-matching cluster (argmax cosine, no threshold)
      3. Record match

    Args:
        net: ClusterNetwork to evaluate
        venv: VisualEnvironment
        nets: All lower networks (for building cluster summary)
        stage: Which layer this is (0=V1, 1=V2, 2=V4, 3=IT)
        n_images: How many images to evaluate (None = all)

    Returns:
        {'purities': {...}, 'avg_purity': float, 'avg_relaxed': float,
         'confusion': {...}, 'coverage': float, 'n_clusters': int, ...}
    """
    if n_images is None:
        n_images = venv.n_images

    cluster_hits = defaultdict(list)  # cid -> [(idx, label, sim)]

    for idx in range(n_images):
        # Get features
        v1_feat = venv.encodings[idx]
        v2_feat = (venv.encodings_v2[idx].copy()
                   if venv.encodings_v2 is not None else None)
        v4_feat = (venv.encodings_v4[idx].copy()
                   if venv.encodings_v4 is not None else None)

        features_dict = {'v1': v1_feat, 'v2': v2_feat, 'v4': v4_feat}
        s = build_layer_sensory(features_dict, nets, stage)

        # Best match (read-only)
        if net.n_clusters == 0:
            continue
        h = np.tanh(s + 1e-8)
        mask = _auto_mask(s)
        best_sim, best_c = -1.0, None
        for c in net.clusters:
            sim = _masked_cosine(h, c.centroid, mask)
            if sim > best_sim:
                best_sim, best_c = sim, c

        if best_c is not None:
            label = int(venv.labels[idx])
            cluster_hits[id(best_c)].append((idx, label, best_sim))

    # Compute purities
    n_classes = venv.n_classes
    purities = {}
    for cid, hits in cluster_hits.items():
        if len(hits) < 3:
            continue
        class_counts = defaultdict(int)
        group_counts = defaultdict(int)
        for _, label, _ in hits:
            class_counts[label] += 1
            group_counts[_group_label(label, n_classes)] += 1

        total = len(hits)
        max_class = max(class_counts, key=class_counts.get)
        max_group = max(group_counts, key=group_counts.get)

        purities[cid] = {
            'total_hits': total,
            'purity': class_counts[max_class] / total,
            'relaxed_purity': group_counts[max_group] / total,
            'top_class': max_class,
            'top_class_name': venv.label_names[max_class],
            'top_count': class_counts[max_class],
            'n_classes_present': len(class_counts),
        }

    avg_purity = float(np.mean([d['purity'] for d in purities.values()])) \
        if purities else 0.0
    avg_relaxed = float(np.mean([d['relaxed_purity'] for d in purities.values()])) \
        if purities else 0.0

    # Coverage
    n_eval = min(1000, n_images)
    indices = np.random.choice(n_images, n_eval, replace=False)
    n_hit = 0
    avg_sim = 0.0
    for idx in indices:
        v1_feat = venv.encodings[idx]
        v2_feat = (venv.encodings_v2[idx].copy()
                   if venv.encodings_v2 is not None else None)
        v4_feat = (venv.encodings_v4[idx].copy()
                   if venv.encodings_v4 is not None else None)
        features_dict = {'v1': v1_feat, 'v2': v2_feat, 'v4': v4_feat}
        s = build_layer_sensory(features_dict, nets, stage)

        if net.n_clusters > 0:
            h = np.tanh(s + 1e-8)
            mask = _auto_mask(s)
            best_sim = max(_masked_cosine(h, c.centroid, mask)
                          for c in net.clusters)
            n_hit += 1
            avg_sim += best_sim
    coverage = n_hit / n_eval
    avg_sim = avg_sim / n_hit if n_hit > 0 else 0.0

    # Confusion matrix (per-class recall)
    confusion = defaultdict(lambda: defaultdict(float))
    n_per_class = min(50, n_images // n_classes)
    class_counts = np.zeros(n_classes, dtype=np.int32)
    cluster_assignment = {}
    # Assign each cluster to its top class
    for cid, data in purities.items():
        if data['total_hits'] >= 3:
            cluster_assignment[cid] = data['top_class']

    for c in range(n_classes):
        c_indices = np.where(venv.labels == c)[0]
        if len(c_indices) == 0:
            continue
        sample = np.random.choice(c_indices,
                                   min(n_per_class, len(c_indices)),
                                   replace=False)
        for idx in sample:
            v1_feat = venv.encodings[idx]
            v2_feat = (venv.encodings_v2[idx].copy()
                       if venv.encodings_v2 is not None else None)
            v4_feat = (venv.encodings_v4[idx].copy()
                       if venv.encodings_v4 is not None else None)
            features_dict = {'v1': v1_feat, 'v2': v2_feat, 'v4': v4_feat}
            s = build_layer_sensory(features_dict, nets, stage)

            class_counts[c] += 1
            if net.n_clusters > 0:
                h = np.tanh(s + 1e-8)
                mask = _auto_mask(s)
                best_sim, best_cid = -1.0, None
                for cl in net.clusters:
                    sim = _masked_cosine(h, cl.centroid, mask)
                    if sim > best_sim:
                        best_sim, best_cid = sim, id(cl)
                if best_cid is not None and best_cid in cluster_assignment:
                    confusion[c][cluster_assignment[best_cid]] += 1.0

    # Normalize confusion rows
    for c in range(n_classes):
        if class_counts[c] > 0:
            for pred_c in confusion[c]:
                confusion[c][pred_c] /= class_counts[c]

    avg_diagonal = float(np.mean([confusion[c].get(c, 0.0)
                                   for c in range(n_classes)]))

    # Top clusters
    sorted_pure = sorted(purities.items(),
                          key=lambda x: -x[1]['purity'])
    top_clusters = []
    for rank, (cid, data) in enumerate(sorted_pure[:10]):
        top_clusters.append({
            'rank': rank + 1,
            'class': data['top_class_name'],
            'hits': data['total_hits'],
            'purity': data['purity'],
            'relaxed_purity': data['relaxed_purity'],
        })

    # Cluster size distribution
    cluster_sizes = [d['total_hits'] for d in purities.values()]

    return {
        'n_clusters': net.n_clusters,
        'n_active': len(purities),
        'avg_purity': avg_purity,
        'avg_relaxed': avg_relaxed,
        'coverage': coverage,
        'avg_similarity': avg_sim,
        'avg_diagonal': avg_diagonal,
        'top_clusters': top_clusters,
        'cluster_sizes': cluster_sizes,
        'mean_hits': float(np.mean(cluster_sizes)) if cluster_sizes else 0,
        'median_hits': float(np.median(cluster_sizes)) if cluster_sizes else 0,
        'cluster_hits': dict(cluster_hits),
        'purities': purities,
        'confusion': dict(confusion),
    }


# ================================================================
# Sequential Training (Hierarchical)
# ================================================================

def train_sequential(venv, n_images: int = 2000,
                      seed: int = 42) -> dict:
    """4-stage hierarchical training: V1→freeze→V2→freeze→V4→freeze→IT.

    Each stage trains one layer to convergence on its feature subspace,
    then freezes it. Higher layers see lower-layer cluster activation
    summaries as part of their input.
    """
    rng = np.random.default_rng(seed)
    nets = {}
    history = []

    # Image order (consistent across stages)
    order = rng.permutation(n_images)

    # ---- Stage 1: V1 ----
    print("\n" + "=" * 64)
    print("  Stage 1/4: Training V1 (128d Gabor edges)")
    print("=" * 64)
    print(f"  threshold={LAYER_PARAMS['v1']['threshold']}, "
          f"lr={LAYER_PARAMS['v1']['lr']}")

    v1_net = _create_net('v1')
    nets['v1'] = v1_net
    t0 = time.perf_counter()

    for step, idx in enumerate(order):
        v1_feat = venv.encodings[idx]
        features_dict = {'v1': v1_feat, 'v2': None, 'v4': None}
        s = build_layer_sensory(features_dict, nets, stage=0)
        v1_net.learn(s)

        if (step + 1) % 1000 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  V1: {step+1}/{n_images} "
                  f"({(step+1)/max(elapsed,0.001):.0f} img/s), "
                  f"{v1_net.n_clusters} clusters")

        if (step + 1) % SLEEP_INTERVAL == 0:
            n_removed = sleep_cycle(v1_net, v1_net.theta)
            if n_removed > 0:
                print(f"  [Sleep] V1: removed {n_removed}, "
                      f"{v1_net.n_clusters} remain")

    elapsed = time.perf_counter() - t0
    print(f"  V1 trained: {v1_net.n_clusters} clusters in {elapsed:.1f}s")
    history.append({'stage': 'V1', 'n_clusters': v1_net.n_clusters,
                    'time_s': elapsed})

    # ---- Stage 2: V2 ----
    print(f"\n{'='*64}")
    print(f"  Stage 2/4: Training V2 (64d + V1 cluster context)")
    print(f"{'='*64}")
    print(f"  threshold={LAYER_PARAMS['v2']['threshold']}, "
          f"lr={LAYER_PARAMS['v2']['lr']}")

    v2_net = _create_net('v2')
    nets['v2'] = v2_net
    t0 = time.perf_counter()

    for step, idx in enumerate(order):
        v1_feat = venv.encodings[idx]
        v2_feat = venv.encodings_v2[idx]
        features_dict = {'v1': v1_feat, 'v2': v2_feat, 'v4': None}
        s = build_layer_sensory(features_dict, nets, stage=1)
        v2_net.learn(s)

        if (step + 1) % 1000 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  V2: {step+1}/{n_images} "
                  f"({(step+1)/max(elapsed,0.001):.0f} img/s), "
                  f"{v2_net.n_clusters} clusters")

        if (step + 1) % SLEEP_INTERVAL == 0:
            n_removed = sleep_cycle(v2_net, v2_net.theta)
            if n_removed > 0:
                print(f"  [Sleep] V2: removed {n_removed}, "
                      f"{v2_net.n_clusters} remain")

    elapsed = time.perf_counter() - t0
    print(f"  V2 trained: {v2_net.n_clusters} clusters in {elapsed:.1f}s")
    history.append({'stage': 'V2', 'n_clusters': v2_net.n_clusters,
                    'time_s': elapsed})

    # ---- Stage 3: V4 ----
    print(f"\n{'='*64}")
    print(f"  Stage 3/4: Training V4 (64d + V1+V2 cluster context)")
    print(f"{'='*64}")
    print(f"  threshold={LAYER_PARAMS['v4']['threshold']}, "
          f"lr={LAYER_PARAMS['v4']['lr']}")

    v4_net = _create_net('v4')
    nets['v4'] = v4_net
    t0 = time.perf_counter()

    for step, idx in enumerate(order):
        v1_feat = venv.encodings[idx]
        v2_feat = venv.encodings_v2[idx]
        v4_feat = venv.encodings_v4[idx]
        features_dict = {'v1': v1_feat, 'v2': v2_feat, 'v4': v4_feat}
        s = build_layer_sensory(features_dict, nets, stage=2)
        v4_net.learn(s)

        if (step + 1) % 1000 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  V4: {step+1}/{n_images} "
                  f"({(step+1)/max(elapsed,0.001):.0f} img/s), "
                  f"{v4_net.n_clusters} clusters")

        if (step + 1) % SLEEP_INTERVAL == 0:
            n_removed = sleep_cycle(v4_net, v4_net.theta)
            if n_removed > 0:
                print(f"  [Sleep] V4: removed {n_removed}, "
                      f"{v4_net.n_clusters} remain")

    elapsed = time.perf_counter() - t0
    print(f"  V4 trained: {v4_net.n_clusters} clusters in {elapsed:.1f}s")
    history.append({'stage': 'V4', 'n_clusters': v4_net.n_clusters,
                    'time_s': elapsed})

    # ---- Stage 4: IT ----
    print(f"\n{'='*64}")
    print(f"  Stage 4/4: Training IT (ALL features + ALL cluster context)")
    print(f"{'='*64}")
    print(f"  threshold={LAYER_PARAMS['it']['threshold']}, "
          f"lr={LAYER_PARAMS['it']['lr']}")

    it_net = _create_net('it')
    nets['it'] = it_net
    t0 = time.perf_counter()

    for step, idx in enumerate(order):
        v1_feat = venv.encodings[idx]
        v2_feat = venv.encodings_v2[idx]
        v4_feat = venv.encodings_v4[idx]
        features_dict = {'v1': v1_feat, 'v2': v2_feat, 'v4': v4_feat}
        s = build_layer_sensory(features_dict, nets, stage=3)
        it_net.learn(s)

        if (step + 1) % 1000 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  IT: {step+1}/{n_images} "
                  f"({(step+1)/max(elapsed,0.001):.0f} img/s), "
                  f"{it_net.n_clusters} clusters")

        if (step + 1) % SLEEP_INTERVAL == 0:
            n_removed = sleep_cycle(it_net, it_net.theta)
            if n_removed > 0:
                print(f"  [Sleep] IT: removed {n_removed}, "
                      f"{it_net.n_clusters} remain")

    elapsed = time.perf_counter() - t0
    print(f"  IT trained: {it_net.n_clusters} clusters in {elapsed:.1f}s")
    history.append({'stage': 'IT', 'n_clusters': it_net.n_clusters,
                    'time_s': elapsed})

    return {'nets': nets, 'history': history, 'mode': 'sequential'}


# ================================================================
# Joint Training (Simultaneous)
# ================================================================

def train_joint(venv, n_images: int = 2000, seed: int = 42) -> dict:
    """Simultaneous training of all 4 layers.

    Each image updates V1, V2, V4, and IT simultaneously.
    IT sees the CURRENT (evolving) state of lower-layer clusters.
    """
    rng = np.random.default_rng(seed)
    history = []

    print("\n" + "=" * 64)
    print("  Joint Training: All 4 layers simultaneously")
    print("=" * 64)

    # Create all networks
    v1_net = _create_net('v1')
    v2_net = _create_net('v2')
    v4_net = _create_net('v4')
    it_net = _create_net('it')
    nets = {'v1': v1_net, 'v2': v2_net, 'v4': v4_net, 'it': it_net}

    for name, net in nets.items():
        p = LAYER_PARAMS[name]
        print(f"  {name}: threshold={p['threshold']}, lr={p['lr']}")

    order = rng.permutation(n_images)
    t0 = time.perf_counter()

    for step, idx in enumerate(order):
        v1_feat = venv.encodings[idx]
        v2_feat = (venv.encodings_v2[idx].copy()
                   if venv.encodings_v2 is not None else None)
        v4_feat = (venv.encodings_v4[idx].copy()
                   if venv.encodings_v4 is not None else None)

        # V1: stage 0 (only V1 features)
        s_v1 = build_layer_sensory(
            {'v1': v1_feat, 'v2': None, 'v4': None}, nets, stage=0)
        v1_net.learn(s_v1)

        # V2: stage 1 (V1+V2 features + V1 cluster summary)
        s_v2 = build_layer_sensory(
            {'v1': v1_feat, 'v2': v2_feat, 'v4': None}, nets, stage=1)
        v2_net.learn(s_v2)

        # V4: stage 2 (V1+V2+V4 features + V1+V2 cluster summary)
        s_v4 = build_layer_sensory(
            {'v1': v1_feat, 'v2': v2_feat, 'v4': v4_feat}, nets, stage=2)
        v4_net.learn(s_v4)

        # IT: stage 3 (ALL features + ALL cluster summaries)
        s_it = build_layer_sensory(
            {'v1': v1_feat, 'v2': v2_feat, 'v4': v4_feat}, nets, stage=3)
        it_net.learn(s_it)

        if (step + 1) % 1000 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  Step {step+1}/{n_images} "
                  f"({(step+1)/max(elapsed,0.001):.0f} img/s) | "
                  f"V1:{v1_net.n_clusters} V2:{v2_net.n_clusters} "
                  f"V4:{v4_net.n_clusters} IT:{it_net.n_clusters}")

        if (step + 1) % SLEEP_INTERVAL == 0:
            for name, net in nets.items():
                n_removed = sleep_cycle(net, net.theta)
                if n_removed > 0:
                    print(f"  [Sleep] {name}: removed {n_removed}, "
                          f"{net.n_clusters} remain")

    elapsed = time.perf_counter() - t0
    print(f"\n  Joint training complete in {elapsed:.1f}s")
    for name, net in nets.items():
        print(f"  {name}: {net.n_clusters} clusters")
        history.append({'stage': name.upper(), 'n_clusters': net.n_clusters,
                        'time_s': elapsed / 4})  # rough per-layer time

    return {'nets': nets, 'history': history, 'mode': 'joint'}


# ================================================================
# Per-Class Recall Comparison
# ================================================================

def per_class_recall(eval_results: dict, venv, nets: dict,
                      stage: int, n_per_class: int = 30) -> dict:
    """Compute per-class recall for a specific layer.

    For each class, sample images and check if the layer's best cluster
    is assigned to the correct class.
    """
    net = eval_results.get('_net')
    if net is None:
        return {}

    n_classes = venv.n_classes
    cluster_hits = eval_results.get('cluster_hits', {})

    # Assign each cluster to its top class
    cluster_to_class = {}
    for cid, hits in cluster_hits.items():
        class_counts = defaultdict(int)
        for _, label, _ in hits:
            class_counts[label] += 1
        cluster_to_class[cid] = max(class_counts, key=class_counts.get)

    recalls = {}
    for c in range(n_classes):
        c_indices = np.where(venv.labels == c)[0]
        if len(c_indices) == 0:
            recalls[venv.label_names[c]] = 0.0
            continue

        sample = np.random.choice(c_indices,
                                   min(n_per_class, len(c_indices)),
                                   replace=False)
        correct = 0
        for idx in sample:
            v1_feat = venv.encodings[idx]
            v2_feat = (venv.encodings_v2[idx].copy()
                       if venv.encodings_v2 is not None else None)
            v4_feat = (venv.encodings_v4[idx].copy()
                       if venv.encodings_v4 is not None else None)
            features_dict = {'v1': v1_feat, 'v2': v2_feat, 'v4': v4_feat}
            s = build_layer_sensory(features_dict, nets, stage)

            if net.n_clusters > 0:
                h = np.tanh(s + 1e-8)
                mask = _auto_mask(s)
                best_sim, best_cid = -1.0, None
                for cl in net.clusters:
                    sim = _masked_cosine(h, cl.centroid, mask)
                    if sim > best_sim:
                        best_sim, best_cid = sim, id(cl)
                if (best_cid is not None and
                    best_cid in cluster_to_class and
                    cluster_to_class[best_cid] == c):
                    correct += 1

        recalls[venv.label_names[c]] = correct / len(sample)

    return recalls


# ================================================================
# Main Experiment
# ================================================================

def run_it_experiment(n_images: int = 2000,
                       dataset: str = 'imagenette',
                       mode: str = 'both',
                       seed: int = 42):
    """Phase 2 IT: Hierarchical Hebb Learning + IT Layer.

    Args:
        n_images: Number of images to use
        dataset: 'cifar10' or 'imagenette'
        mode: 'seq', 'joint', or 'both'
        seed: Random seed
    """
    np.random.seed(seed)

    print("=" * 64)
    print("  Phase 2 IT: Hierarchical Hebb Learning + IT Layer")
    print("=" * 64)
    print(f"  Dataset: {dataset}, {n_images} images")
    print(f"  Mode: {mode}")
    print(f"  Layout: V1({V1_WIDTH}d) + V2({V2_WIDTH}d) + "
          f"V4({V4_WIDTH}d) + CS({CS_WIDTH}d)")
    print()

    # ---- Load VisualEnvironment ----
    print("[1/3] Loading VisualEnvironment...")
    t0 = time.perf_counter()
    from visual_interface import VisualEnvironment

    venv = VisualEnvironment(
        dataset=dataset, n_images=n_images,
        pca_components=V1_WIDTH,
        v2_components=V2_WIDTH,
        v4_components=V4_WIDTH,  # PCA V4 to 64d
        use_v2=True, use_v4=True,
    )
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s")
    print(f"  V1: {venv.encodings.shape}")
    if venv.encodings_v2 is not None:
        print(f"  V2: {venv.encodings_v2.shape}")
    if venv.encodings_v4 is not None:
        print(f"  V4: {venv.encodings_v4.shape}")

    results = {}
    n_images_actual = venv.n_images

    # ---- Sequential Training ----
    if mode in ('seq', 'both'):
        print(f"\n[2a/3] Sequential Training...")
        seq_result = train_sequential(venv, n_images_actual, seed=seed)
        nets_seq = seq_result['nets']

        print(f"\n[3a/3] Evaluating Sequential...")
        seq_eval = {}
        for layer_name, stage in [('v1', 0), ('v2', 1), ('v4', 2), ('it', 3)]:
            net = nets_seq[layer_name]
            print(f"  Evaluating {layer_name.upper()} (stage {stage})...")
            # Build a nets dict with only lower layers (IT evaluation needs all)
            eval_nets = {k: nets_seq[k] for k in nets_seq
                        if list(nets_seq.keys()).index(k) <
                           list(nets_seq.keys()).index(layer_name)}
            eval_result = evaluate_layer(net, venv, nets_seq, stage,
                                          n_images_actual)
            eval_result['_net'] = net  # attach for recall computation
            seq_eval[layer_name] = eval_result
            print(f"    {layer_name.upper()}: {eval_result['n_clusters']} clusters, "
                  f"purity={eval_result['avg_purity']:.3f}, "
                  f"relaxed={eval_result['avg_relaxed']:.3f}, "
                  f"diag={eval_result['avg_diagonal']:.3f}")

        results['sequential'] = {
            'training': seq_result,
            'evaluation': seq_eval,
        }

    # ---- Joint Training ----
    if mode in ('joint', 'both'):
        print(f"\n[2b/3] Joint Training...")
        joint_result = train_joint(venv, n_images_actual, seed=seed)
        nets_joint = joint_result['nets']

        print(f"\n[3b/3] Evaluating Joint...")
        joint_eval = {}
        for layer_name, stage in [('v1', 0), ('v2', 1), ('v4', 2), ('it', 3)]:
            net = nets_joint[layer_name]
            print(f"  Evaluating {layer_name.upper()} (stage {stage})...")
            eval_result = evaluate_layer(net, venv, nets_joint, stage,
                                          n_images_actual)
            eval_result['_net'] = net
            joint_eval[layer_name] = eval_result
            print(f"    {layer_name.upper()}: {eval_result['n_clusters']} clusters, "
                  f"purity={eval_result['avg_purity']:.3f}, "
                  f"relaxed={eval_result['avg_relaxed']:.3f}, "
                  f"diag={eval_result['avg_diagonal']:.3f}")

        results['joint'] = {
            'training': joint_result,
            'evaluation': joint_eval,
        }

    # ---- Report ----
    print("\n" + "=" * 72)
    print("  Phase 2 IT Results")
    print("=" * 72)

    # Comparison table
    if mode == 'both':
        print(f"\n  [Comparison] Sequential vs Joint Training:")
        print(f"  {'Layer':<6} {'Seq Purity':<12} {'Seq Relax':<12} "
              f"{'Seq Diag':<10} {'Joint Purity':<12} {'Joint Relax':<12} "
              f"{'Joint Diag':<10}")
        print(f"  {'-'*72}")
        for layer_name in ['v1', 'v2', 'v4', 'it']:
            s = results['sequential']['evaluation'][layer_name]
            j = results['joint']['evaluation'][layer_name]
            print(f"  {layer_name.upper():<6} {s['avg_purity']:<12.3f} "
                  f"{s['avg_relaxed']:<12.3f} {s['avg_diagonal']:<10.3f} "
                  f"{j['avg_purity']:<12.3f} {j['avg_relaxed']:<12.3f} "
                  f"{j['avg_diagonal']:<10.3f}")

        # Determine winner
        print(f"\n  [Analysis] Per-layer comparison:")
        for layer_name in ['v1', 'v2', 'v4', 'it']:
            s = results['sequential']['evaluation'][layer_name]
            j = results['joint']['evaluation'][layer_name]
            diff = s['avg_purity'] - j['avg_purity']
            winner = 'SEQ' if diff > 0 else ('JOINT' if diff < 0 else 'TIE')
            print(f"    {layer_name.upper()}: delta_purity={diff:+.3f} → {winner}")

        # IT baseline comparison
        print(f"\n  [IT vs Baseline] ")
        it_seq = results['sequential']['evaluation']['it']
        # Baseline = V1+V2+V4 combined features (use v4 eval as proxy)
        v4_seq = results['sequential']['evaluation']['v4']
        print(f"    IT purity:     {it_seq['avg_purity']:.3f}")
        print(f"    V4 purity:     {v4_seq['avg_purity']:.3f}")
        print(f"    IT - V4:       {it_seq['avg_purity'] - v4_seq['avg_purity']:+.3f}")
        print(f"    IT diagonal:   {it_seq['avg_diagonal']:.3f}")
        print(f"    V4 diagonal:   {v4_seq['avg_diagonal']:.3f}")

    # Report the best mode's detailed results
    best_mode = 'sequential' if mode == 'seq' else (
        'joint' if mode == 'joint' else 'sequential')
    best_eval = results[best_mode]['evaluation']

    # Top clusters for each layer
    for layer_name in ['v1', 'v2', 'v4', 'it']:
        ev = best_eval[layer_name]
        print(f"\n  [Top-5] {layer_name.upper()} Clusters ({best_mode}):")
        print(f"    {'Rank':<5} {'Class':<16} {'Hits':<6} {'Purity':<8} {'Relaxed':<8}")
        print(f"    {'-'*48}")
        for tc in ev['top_clusters'][:5]:
            print(f"    {tc['rank']:<5} {tc['class']:<16} "
                  f"{tc['hits']:<6} {tc['purity']:.3f}    {tc['relaxed_purity']:.3f}")

    # Cluster counts
    print(f"\n  [Cluster Counts] ({best_mode}):")
    for layer_name in ['v1', 'v2', 'v4', 'it']:
        ev = best_eval[layer_name]
        print(f"    {layer_name.upper()}: {ev['n_clusters']} total, "
              f"{ev['n_active']} active, "
              f"mean_hits={ev['mean_hits']:.0f}, "
              f"median_hits={ev['median_hits']:.0f}")

    # ---- Acceptance Check ----
    print(f"\n  {'='*48}")
    print(f"  ACCEPTANCE CHECK")
    print(f"  {'='*48}")

    it_ev = best_eval['it']
    checks = []

    c1 = it_ev['n_active'] >= 10
    checks.append(('IT >=10 active clusters', c1, it_ev['n_active']))

    c2 = it_ev['avg_purity'] > 0.25
    checks.append(('IT purity > 0.25', c2, f"{it_ev['avg_purity']:.3f}"))

    c3 = it_ev['coverage'] > 0.50
    checks.append(('IT coverage > 50%', c3, f"{it_ev['coverage']:.1%}"))

    c4 = it_ev['avg_diagonal'] > 0.20
    checks.append(('IT diag > 0.20', c4, f"{it_ev['avg_diagonal']:.3f}"))

    # IT > V4 check
    v4_ev = best_eval['v4']
    it_v4_diff = it_ev['avg_purity'] - v4_ev['avg_purity']
    c5 = it_v4_diff >= -0.02  # Allow small regression
    checks.append(('IT purity >= V4 purity (within 0.02)',
                   c5, f"{it_v4_diff:+.3f}"))

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
            'dataset': dataset,
            'n_images': n_images_actual,
            'mode': mode,
            'v1_width': V1_WIDTH,
            'v2_width': V2_WIDTH,
            'v4_width': V4_WIDTH,
            'cs_width': CS_WIDTH,
        },
        'results': results,
        'checks_passed': all_pass,
    }


# ================================================================
# CLI
# ================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Phase 2 IT: Hierarchical Hebb Learning + IT Layer')
    parser.add_argument('--n', type=int, default=2000,
                       help='Number of images (default: 2000)')
    parser.add_argument('--dataset', type=str, default='imagenette',
                       choices=['cifar10', 'imagenette'],
                       help='Dataset (default: imagenette)')
    parser.add_argument('--mode', type=str, default='both',
                       choices=['seq', 'joint', 'both'],
                       help='Training mode (default: both)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    args = parser.parse_args()

    run_it_experiment(
        n_images=args.n,
        dataset=args.dataset,
        mode=args.mode,
        seed=args.seed,
    )
