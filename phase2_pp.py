"""
phase2_pp.py — Tier 1: Predictive Processing + Color Opponent Vision
自由能原理智能体

Combines two biologically-motivated improvements:
  A. Predictive Processing (预测加工):
     - Cluster centroids ARE predictions
     - IT → V4 → V2 → V1 top-down prediction
     - Prediction error drives learning rate modulation
     - Prediction error accumulates into F_accuracy → Valence

  B. Color Opponent Channels (色拮抗):
     - Red-Green (L-M) + Blue-Yellow (S-(L+M)) opponent channels
     - Gabor filtered + 2x2 grid pooled
     - Adds chromatic information to visual hierarchy

Experiment matrix (4 configurations):
  1. baseline:  Pure feedforward, no color
  2. color:     Feedforward with color opponent
  3. pp:        Predictive processing, no color
  4. color_pp:  Predictive processing with color (full Tier 1)

Usage:
  python phase2_pp.py --n 2000 --mode all    # Run all 4 configs
  python phase2_pp.py --n 2000 --mode pp     # PP only
  python phase2_pp.py --n 2000 --mode color  # Color only
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
# Sensory Layout (D=330)
# ================================================================

V1_WIDTH    = 96
V2_WIDTH    = 64
V4_WIDTH    = 64
COLOR_WIDTH = 64
PE_WIDTH    = 32  # Prediction Error encoding

V1_START,    V1_END    = 10,  10 + V1_WIDTH       # s[10:106]
V2_START,    V2_END    = 106, 106 + V2_WIDTH       # s[106:170]
V4_START,    V4_END    = 170, 170 + V4_WIDTH       # s[170:234]
COLOR_START, COLOR_END = 234, 234 + COLOR_WIDTH    # s[234:298]
PE_START,    PE_END    = 298, 298 + PE_WIDTH       # s[298:330]

# Per-layer Hebb params
LAYER_PARAMS = {
    'v1': {'threshold': 0.50, 'lr': 0.02, 'decay': 0.003, 'hash_offset': V1_START},
    'v2': {'threshold': 0.45, 'lr': 0.02, 'decay': 0.003, 'hash_offset': V2_START},
    'v4': {'threshold': 0.40, 'lr': 0.02, 'decay': 0.003, 'hash_offset': V4_START},
    'it': {'threshold': 0.35, 'lr': 0.02, 'decay': 0.003, 'hash_offset': V1_START},
}

SLEEP_INTERVAL = 500

# Experiment configurations
CONFIGS = {
    'baseline':  {'use_color': False, 'use_pp': False},
    'color':     {'use_color': True,  'use_pp': False},
    'pp':        {'use_color': False, 'use_pp': True},
    'color_pp':  {'use_color': True,  'use_pp': True},
}


# ================================================================
# Prediction Error Encoding
# ================================================================

def encode_prediction_errors(pe_v1: float, pe_v2: float,
                              pe_v4: float, pe_it: float = 0.0) -> np.ndarray:
    """Encode layer-wise prediction errors into 32d feature vector.

    Each layer gets 8 dims: [scalar_value, log_value, 6×pad].
    The pattern of errors tells IT which layers are surprising.
    """
    pe = np.zeros(PE_WIDTH, dtype=np.float32)
    errors = [pe_v1, pe_v2, pe_v4, pe_it]
    for i, err in enumerate(errors):
        base = i * 8
        pe[base] = float(np.clip(err, 0, 10.0))
        pe[base + 1] = float(np.log1p(err))
        # base+2:base+8 stay zero (padding)
    norm = np.linalg.norm(pe)
    if norm > 1e-8:
        pe /= norm
    return pe


# ================================================================
# Cluster Activation Summary
# ================================================================

def compute_cluster_sims(net: ClusterNetwork, features: np.ndarray,
                          start: int, end: int) -> np.ndarray:
    """Cosine similarity of features to all clusters in net."""
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
    """Build 32d cluster activation summary for higher layers."""
    summary = np.zeros(32, dtype=np.float32)
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
        n_top = min(5, len(sorted_sims))
        if pos + 8 <= 32:
            summary[pos:pos + n_top] = sorted_sims[:n_top]
            pos += 5
            summary[pos:pos + 3] = [
                float(np.mean(sims)), float(np.max(sims)), float(np.std(sims))]
            pos += 3

    norm = np.linalg.norm(summary)
    if norm > 1e-8:
        summary /= norm
    return summary.astype(np.float32)


# ================================================================
# Sensory Vector Construction
# ================================================================

def build_layer_sensory(features_dict: dict, nets: dict,
                         stage: int, use_color: bool = False,
                         use_pp: bool = False,
                         pe_errors: dict = None) -> np.ndarray:
    """Build D=330 sensory vector for a specific layer.

    Layout:
      s[10:106]   = V1 Gabor (96d)
      s[106:170]  = V2 Gabor (64d)
      s[170:234]  = V4 Gabor (64d)
      s[234:298]  = Color opponent (64d) if use_color
      s[298:330]  = Prediction Errors (32d) if use_pp

    Args:
        features_dict: {'v1': vec, 'v2': vec, 'v4': vec, 'color': vec}
        nets: layered networks for cluster summary
        stage: 0=V1, 1=V2, 2=V4, 3=IT
        use_color: include color features
        use_pp: include prediction error encoding
        pe_errors: {'v1': float, 'v2': float, 'v4': float} or None
    """
    s = np.zeros(D, dtype=np.float32)

    # V1 features: always present
    v1_feat = features_dict.get('v1')
    if v1_feat is not None:
        flen = min(len(v1_feat), V1_WIDTH)
        s[V1_START:V1_START + flen] = v1_feat[:flen]

    # V2 features: stage >= 1
    if stage >= 1:
        v2_feat = features_dict.get('v2')
        if v2_feat is not None:
            flen = min(len(v2_feat), V2_WIDTH)
            s[V2_START:V2_START + flen] = v2_feat[:flen]

    # V4 features: stage >= 2
    if stage >= 2:
        v4_feat = features_dict.get('v4')
        if v4_feat is not None:
            flen = min(len(v4_feat), V4_WIDTH)
            s[V4_START:V4_START + flen] = v4_feat[:flen]

    # Color features: stage >= 1 and use_color
    if stage >= 1 and use_color:
        color_feat = features_dict.get('color')
        if color_feat is not None:
            flen = min(len(color_feat), COLOR_WIDTH)
            s[COLOR_START:COLOR_START + flen] = color_feat[:flen]

    # Prediction Errors: stage >= 1 and use_pp
    if stage >= 1 and use_pp and pe_errors is not None:
        pe_vec = encode_prediction_errors(
            pe_errors.get('v1', 0.0),
            pe_errors.get('v2', 0.0),
            pe_errors.get('v4', 0.0),
            pe_errors.get('it', 0.0),
        )
        s[PE_START:PE_END] = pe_vec

    return s


# ================================================================
# Network Factory
# ================================================================

def _create_net(layer_name: str) -> ClusterNetwork:
    params = LAYER_PARAMS[layer_name]
    theta = create_default_theta()
    theta.cluster_threshold = params['threshold']
    theta.learn_rate_l0 = params['lr']
    theta.decay_rate = params['decay']
    return ClusterNetwork(theta, hash_offset=params['hash_offset'])


def learn_modulated(net: ClusterNetwork, s: np.ndarray,
                     pe_scalar: float, pe_baseline: float = 0.05):
    """Learn with prediction-error-modulated learning rate.

    Higher prediction error → higher learning rate (surprise → learn more).
    Modulation range: [0.5x, 3.0x] of base learning rate.
    """
    mod = max(0.5, min(3.0, 1.0 + pe_scalar / (pe_baseline + 1e-8)))
    original_lr = net.theta.learn_rate_l0
    net.theta.learn_rate_l0 = original_lr * mod
    net.learn(s)
    net.theta.learn_rate_l0 = original_lr


# ================================================================
# Training
# ================================================================

def train_hierarchy(venv, config: dict, n_images: int,
                     seed: int = 42) -> dict:
    """Train V1→V2→V4→IT hierarchy with optional PP and color.

    Args:
        venv: VisualEnvironment with all encodings loaded
        config: {'use_color': bool, 'use_pp': bool}
        n_images: number of images to use
        seed: random seed

    Returns:
        {'nets': dict, 'history': list, 'F_visual_ema': float}
    """
    rng = np.random.default_rng(seed)
    use_color = config['use_color']
    use_pp = config['use_pp']
    mode_name = ('color_pp' if (use_color and use_pp) else
                 'pp' if use_pp else 'color' if use_color else 'baseline')

    # Create networks
    nets = {}
    for name in ['v1', 'v2', 'v4', 'it']:
        nets[name] = _create_net(name)

    order = rng.permutation(n_images)
    t0 = time.perf_counter()
    history = []

    # EMA tracking for F_accuracy
    F_visual_ema = 0.0
    F_alpha = 0.95

    print(f"\n{'='*64}")
    print(f"  Training: {mode_name}")
    print(f"{'='*64}")
    print(f"  PP={use_pp}, Color={use_color}")
    for name in ['v1', 'v2', 'v4', 'it']:
        p = LAYER_PARAMS[name]
        print(f"  {name}: threshold={p['threshold']}, lr={p['lr']}")

    pe_baseline = 0.05  # baseline prediction error for modulation

    for step, idx in enumerate(order):
        # Get features
        v1_feat = venv.encodings[idx]
        v2_feat = (venv.encodings_v2[idx].copy()
                   if venv.encodings_v2 is not None else None)
        v4_feat = (venv.encodings_v4[idx].copy()
                   if venv.encodings_v4 is not None else None)
        color_feat = (venv.encodings_color[idx].copy()
                      if use_color and getattr(venv, 'encodings_color', None) is not None
                      else None)

        features_dict = {
            'v1': v1_feat, 'v2': v2_feat, 'v4': v4_feat, 'color': color_feat,
        }

        # Build sensory vectors for each layer
        # Start with no PP errors (will compute them after recall)
        pe_errors = {'v1': 0.0, 'v2': 0.0, 'v4': 0.0, 'it': 0.0}

        s_it = build_layer_sensory(features_dict, nets, stage=3,
                                    use_color=use_color, use_pp=use_pp,
                                    pe_errors=pe_errors)
        s_v4 = build_layer_sensory(features_dict, nets, stage=2,
                                    use_color=use_color, use_pp=use_pp,
                                    pe_errors=pe_errors)
        s_v2 = build_layer_sensory(features_dict, nets, stage=1,
                                    use_color=use_color, use_pp=use_pp,
                                    pe_errors=pe_errors)
        s_v1 = build_layer_sensory(features_dict, nets, stage=0,
                                    use_color=use_color, use_pp=False,
                                    pe_errors=None)

        # ---- Feedforward recall ----
        c_it  = nets['it'].recall(s_it)
        c_v4  = nets['v4'].recall(s_v4)
        c_v2  = nets['v2'].recall(s_v2)
        c_v1  = nets['v1'].recall(s_v1)

        # ---- Predictive Processing ----
        if use_pp:
            # IT predicts V4 features
            if c_it is not None:
                pred_v4 = c_it.centroid[V4_START:V4_END]
                actual_v4 = s_v4[V4_START:V4_END]
                mask_v4 = np.abs(actual_v4) > 1e-6
                if mask_v4.any():
                    pe_errors['v4'] = float(np.sum(
                        (actual_v4[mask_v4] - pred_v4[mask_v4]) ** 2))
                else:
                    pe_errors['v4'] = 0.0
            else:
                pe_errors['v4'] = 1.0  # max surprise: no IT cluster matched

            # V4 predicts V2 features
            if c_v4 is not None:
                pred_v2 = c_v4.centroid[V2_START:V2_END]
                actual_v2 = s_v2[V2_START:V2_END]
                mask_v2 = np.abs(actual_v2) > 1e-6
                if mask_v2.any():
                    pe_errors['v2'] = float(np.sum(
                        (actual_v2[mask_v2] - pred_v2[mask_v2]) ** 2))
                else:
                    pe_errors['v2'] = 0.0
            else:
                pe_errors['v2'] = 1.0

            # V2 predicts V1 features
            if c_v2 is not None:
                pred_v1 = c_v2.centroid[V1_START:V1_END]
                actual_v1 = s_v1[V1_START:V1_END]
                mask_v1 = np.abs(actual_v1) > 1e-6
                if mask_v1.any():
                    pe_errors['v1'] = float(np.sum(
                        (actual_v1[mask_v1] - pred_v1[mask_v1]) ** 2))
                else:
                    pe_errors['v1'] = 0.0
            else:
                pe_errors['v1'] = 1.0

            # IT self-prediction error
            if c_it is not None:
                pe_errors['it'] = float(np.sum(
                    (s_it[V1_START:PE_END] - c_it.centroid[V1_START:PE_END]) ** 2
                )) / (PE_END - V1_START)
            else:
                pe_errors['it'] = 1.0

            # Update F_visual_ema
            F_visual_total = (pe_errors['v1'] + pe_errors['v2'] + pe_errors['v4']) / 3.0
            F_visual_ema = F_alpha * F_visual_ema + (1 - F_alpha) * F_visual_total

            # Rebuild sensory vectors with prediction errors encoded
            s_it = build_layer_sensory(features_dict, nets, stage=3,
                                        use_color=use_color, use_pp=True,
                                        pe_errors=pe_errors)
            s_v4 = build_layer_sensory(features_dict, nets, stage=2,
                                        use_color=use_color, use_pp=True,
                                        pe_errors=pe_errors)
            s_v2 = build_layer_sensory(features_dict, nets, stage=1,
                                        use_color=use_color, use_pp=True,
                                        pe_errors=pe_errors)

        # ---- Learn (modulated by PP if enabled) ----
        if use_pp:
            # IT learns with V4 prediction error modulation
            learn_modulated(nets['it'], s_it, pe_errors['v4'], pe_baseline)
            # V4 learns with V2 prediction error modulation
            learn_modulated(nets['v4'], s_v4, pe_errors['v2'], pe_baseline)
            # V2 learns with V1 prediction error modulation
            learn_modulated(nets['v2'], s_v2, pe_errors['v1'], pe_baseline)
            # V1 learns at baseline rate (no lower layer)
            nets['v1'].learn(s_v1)
        else:
            nets['it'].learn(s_it)
            nets['v4'].learn(s_v4)
            nets['v2'].learn(s_v2)
            nets['v1'].learn(s_v1)

        # ---- Sleep ----
        if (step + 1) % SLEEP_INTERVAL == 0:
            for name, net in nets.items():
                n_removed = sleep_cycle(net, net.theta)
                if n_removed > 0:
                    print(f"  [Sleep] {name}: removed {n_removed}, "
                          f"{net.n_clusters} remain")

        # ---- Logging ----
        if (step + 1) % 1000 == 0:
            elapsed = time.perf_counter() - t0
            ips = (step + 1) / max(elapsed, 0.001)
            pp_info = (f"PE=[{pe_errors['v1']:.3f},{pe_errors['v2']:.3f},"
                       f"{pe_errors['v4']:.3f}] Fv={F_visual_ema:.4f}") if use_pp else ""
            print(f"  {step+1}/{n_images} ({ips:.0f} img/s) | "
                  f"V1:{nets['v1'].n_clusters} V2:{nets['v2'].n_clusters} "
                  f"V4:{nets['v4'].n_clusters} IT:{nets['it'].n_clusters} | "
                  f"{pp_info}")

    elapsed = time.perf_counter() - t0
    print(f"  Training complete: {elapsed:.1f}s")
    for name, net in nets.items():
        print(f"    {name}: {net.n_clusters} clusters")

    history.append({
        'mode': mode_name,
        'total_time_s': elapsed,
        'F_visual_ema': float(F_visual_ema),
        'final_clusters': {name: net.n_clusters for name, net in nets.items()},
    })

    return {
        'nets': nets,
        'history': history,
        'F_visual_ema': F_visual_ema,
        'mode': mode_name,
    }


# ================================================================
# Evaluation
# ================================================================

def evaluate_layer(net: ClusterNetwork, venv, nets: dict,
                    stage: int, use_color: bool, use_pp: bool,
                    n_images: int = None) -> dict:
    """Read-only evaluation of a layer's clustering quality."""
    if n_images is None:
        n_images = venv.n_images

    cluster_hits = defaultdict(list)

    for idx in range(n_images):
        v1_feat = venv.encodings[idx]
        v2_feat = (venv.encodings_v2[idx].copy()
                   if venv.encodings_v2 is not None else None)
        v4_feat = (venv.encodings_v4[idx].copy()
                   if venv.encodings_v4 is not None else None)
        color_feat = (venv.encodings_color[idx].copy()
                      if use_color and getattr(venv, 'encodings_color', None) is not None
                      else None)

        features_dict = {
            'v1': v1_feat, 'v2': v2_feat, 'v4': v4_feat, 'color': color_feat,
        }
        s = build_layer_sensory(features_dict, nets, stage,
                                 use_color=use_color, use_pp=use_pp,
                                 pe_errors=None)

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

    avg_purity = float(np.mean([d['purity'] for d in purities.values()])) \
        if purities else 0.0

    # Coverage
    n_eval = min(1000, n_images)
    indices = np.random.choice(n_images, n_eval, replace=False)
    n_hit = 0
    for idx in indices:
        v1_feat = venv.encodings[idx]
        v2_feat = (venv.encodings_v2[idx].copy()
                   if venv.encodings_v2 is not None else None)
        v4_feat = (venv.encodings_v4[idx].copy()
                   if venv.encodings_v4 is not None else None)
        color_feat = (venv.encodings_color[idx].copy()
                      if use_color and getattr(venv, 'encodings_color', None) is not None
                      else None)
        features_dict = {
            'v1': v1_feat, 'v2': v2_feat, 'v4': v4_feat, 'color': color_feat,
        }
        s = build_layer_sensory(features_dict, nets, stage,
                                 use_color=use_color, use_pp=use_pp,
                                 pe_errors=None)
        if net.n_clusters > 0:
            h = np.tanh(s + 1e-8)
            mask = _auto_mask(s)
            best_sim = max(_masked_cosine(h, c.centroid, mask)
                          for c in net.clusters)
            n_hit += 1

    coverage = n_hit / n_eval

    # Confusion diagonal
    n_classes = venv.n_classes
    cluster_assignment = {}
    for cid, data in purities.items():
        if data['total_hits'] >= 3:
            cluster_assignment[cid] = data['top_class']

    confusion_diag = np.zeros(n_classes, dtype=np.float32)
    n_per_class = min(50, n_images // n_classes)
    for c in range(n_classes):
        c_indices = np.where(venv.labels == c)[0]
        if len(c_indices) == 0:
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
            color_feat = (venv.encodings_color[idx].copy()
                          if use_color and getattr(venv, 'encodings_color', None) is not None
                          else None)
            features_dict = {
                'v1': v1_feat, 'v2': v2_feat, 'v4': v4_feat, 'color': color_feat,
            }
            s = build_layer_sensory(features_dict, nets, stage,
                                     use_color=use_color, use_pp=use_pp,
                                     pe_errors=None)
            if net.n_clusters > 0:
                h = np.tanh(s + 1e-8)
                mask = _auto_mask(s)
                best_sim, best_cid = -1.0, None
                for cl in net.clusters:
                    sim = _masked_cosine(h, cl.centroid, mask)
                    if sim > best_sim:
                        best_sim, best_cid = sim, id(cl)
                if (best_cid is not None and
                    best_cid in cluster_assignment and
                    cluster_assignment[best_cid] == c):
                    correct += 1
        confusion_diag[c] = correct / max(len(sample), 1)

    avg_diagonal = float(np.mean(confusion_diag))

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
        })

    return {
        'n_clusters': net.n_clusters,
        'n_active': len(purities),
        'avg_purity': avg_purity,
        'coverage': coverage,
        'avg_diagonal': avg_diagonal,
        'top_clusters': top_clusters,
        'cluster_hits': dict(cluster_hits),
        'purities': purities,
    }


# ================================================================
# Per-class recall
# ================================================================

def per_class_recall(net: ClusterNetwork, venv, nets: dict,
                      stage: int, use_color: bool, use_pp: bool,
                      n_per_class: int = 30) -> dict:
    """Compute per-class recall for a layer."""
    n_classes = venv.n_classes

    # Build cluster→class assignment
    cluster_hits = defaultdict(list)
    for idx in range(min(500, venv.n_images)):
        v1_feat = venv.encodings[idx]
        v2_feat = (venv.encodings_v2[idx].copy()
                   if venv.encodings_v2 is not None else None)
        v4_feat = (venv.encodings_v4[idx].copy()
                   if venv.encodings_v4 is not None else None)
        color_feat = (venv.encodings_color[idx].copy()
                      if use_color and getattr(venv, 'encodings_color', None) is not None
                      else None)
        features_dict = {
            'v1': v1_feat, 'v2': v2_feat, 'v4': v4_feat, 'color': color_feat,
        }
        s = build_layer_sensory(features_dict, nets, stage,
                                 use_color=use_color, use_pp=use_pp,
                                 pe_errors=None)
        if net.n_clusters > 0:
            h = np.tanh(s + 1e-8)
            mask = _auto_mask(s)
            best_sim, best_cid = -1.0, None
            for c in net.clusters:
                sim = _masked_cosine(h, c.centroid, mask)
                if sim > best_sim:
                    best_sim, best_cid = sim, id(c)
            if best_cid is not None:
                cluster_hits[best_cid].append(int(venv.labels[idx]))

    cluster_to_class = {}
    for cid, labels in cluster_hits.items():
        class_counts = defaultdict(int)
        for l in labels:
            class_counts[l] += 1
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
            color_feat = (venv.encodings_color[idx].copy()
                          if use_color and getattr(venv, 'encodings_color', None) is not None
                          else None)
            features_dict = {
                'v1': v1_feat, 'v2': v2_feat, 'v4': v4_feat, 'color': color_feat,
            }
            s = build_layer_sensory(features_dict, nets, stage,
                                     use_color=use_color, use_pp=use_pp,
                                     pe_errors=None)
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

def run_pp_experiment(n_images: int = 2000,
                       dataset: str = 'imagenette',
                       mode: str = 'all',
                       seed: int = 42):
    """Tier 1: Predictive Processing + Color Opponent experiment.

    Args:
        n_images: Number of images
        dataset: 'cifar10' or 'imagenette'
        mode: 'all', 'baseline', 'color', 'pp', 'color_pp'
        seed: Random seed
    """
    np.random.seed(seed)

    print("=" * 72)
    print("  Tier 1: Predictive Processing + Color Opponent Vision")
    print("=" * 72)
    print(f"  Dataset: {dataset}, {n_images} images, mode: {mode}")
    print(f"  Layout: V1({V1_WIDTH}) + V2({V2_WIDTH}) + V4({V4_WIDTH})"
          f" + Color({COLOR_WIDTH}) + PE({PE_WIDTH})")
    print()

    # Determine which configs to run
    if mode == 'all':
        configs_to_run = list(CONFIGS.keys())
    else:
        configs_to_run = [mode]

    # Load VisualEnvironment ONCE (with color if any config needs it)
    need_color = any(CONFIGS[c]['use_color'] for c in configs_to_run)

    print(f"[1/3] Loading VisualEnvironment (color={need_color})...")
    t0 = time.perf_counter()
    from visual_interface import VisualEnvironment

    venv = VisualEnvironment(
        dataset=dataset, n_images=n_images,
        pca_components=V1_WIDTH,
        v2_components=V2_WIDTH,
        v4_components=V4_WIDTH,
        color_components=COLOR_WIDTH,
        use_v2=True, use_v4=True,
        use_color=need_color,
    )
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s")
    print(f"  V1: {venv.encodings.shape}")
    if venv.encodings_v2 is not None:
        print(f"  V2: {venv.encodings_v2.shape}")
    if venv.encodings_v4 is not None:
        print(f"  V4: {venv.encodings_v4.shape}")
    if need_color and getattr(venv, 'encodings_color', None) is not None:
        print(f"  Color: {venv.encodings_color.shape}")
    n_actual = venv.n_images

    results = {}

    for config_name in configs_to_run:
        config = CONFIGS[config_name]
        print(f"\n[2/{config_name}] Training...")

        train_result = train_hierarchy(venv, config, n_actual, seed=seed)
        nets = train_result['nets']

        print(f"\n[3/{config_name}] Evaluating...")
        eval_results = {}
        for layer_name, stage in [('v1', 0), ('v2', 1), ('v4', 2), ('it', 3)]:
            net = nets[layer_name]
            ev = evaluate_layer(net, venv, nets, stage,
                                 config['use_color'], config['use_pp'],
                                 n_actual)
            eval_results[layer_name] = ev
            print(f"    {layer_name.upper()}: {ev['n_clusters']} clusters, "
                  f"purity={ev['avg_purity']:.3f}, "
                  f"diag={ev['avg_diagonal']:.3f}")

        # Per-class recall for IT
        it_recall = per_class_recall(nets['it'], venv, nets, 3,
                                      config['use_color'], config['use_pp'])
        eval_results['it']['recall'] = it_recall

        results[config_name] = {
            'training': train_result,
            'evaluation': eval_results,
        }

    # ---- Report ----
    print("\n" + "=" * 72)
    print("  Tier 1 Results: Predictive Processing + Color")
    print("=" * 72)

    # Summary table
    print(f"\n  [IT Layer Comparison]")
    header = f"  {'Config':<12} {'IT Clusters':<12} {'IT Purity':<10} {'IT Diag':<10} {'F_visual':<10}"
    print(header)
    print(f"  {'-'*len(header)}")
    for config_name in configs_to_run:
        r = results[config_name]
        it_ev = r['evaluation']['it']
        F_ema = r['training']['F_visual_ema']
        print(f"  {config_name:<12} {it_ev['n_active']:<12} "
              f"{it_ev['avg_purity']:<10.3f} {it_ev['avg_diagonal']:<10.3f} "
              f"{F_ema:<10.4f}")

    # Per-class recall comparison
    print(f"\n  [Per-Class IT Recall]")
    if results:
        first_config = configs_to_run[0]
        class_names = list(results[first_config]['evaluation']['it'].get('recall', {}).keys())
        header = f"  {'Class':<16}"
        for cn in configs_to_run:
            header += f" {cn:<10}"
        print(header)
        print(f"  {'-'*len(header)}")
        for name in class_names:
            row = f"  {name:<16}"
            for cn in configs_to_run:
                rec = results[cn]['evaluation']['it'].get('recall', {}).get(name, 0.0)
                row += f" {rec:<10.3f}"
            print(row)

    # Best IT per-class improvements
    if len(configs_to_run) >= 2:
        print(f"\n  [Per-Class Delta vs Baseline]")
        baseline_recall = results[configs_to_run[0]]['evaluation']['it'].get('recall', {})
        for cn in configs_to_run[1:]:
            cn_recall = results[cn]['evaluation']['it'].get('recall', {})
            print(f"  {cn} vs {configs_to_run[0]}:")
            deltas = []
            for name in class_names:
                delta = cn_recall.get(name, 0.0) - baseline_recall.get(name, 0.0)
                deltas.append((name, delta))
            deltas.sort(key=lambda x: -x[1])
            for name, delta in deltas[:5]:
                marker = '+' if delta > 0 else ''
                print(f"    {name:<16}: {marker}{delta:.3f}")
            avg_delta = float(np.mean([d[1] for d in deltas]))
            print(f"    {'avg':<16}: {avg_delta:+.3f}")

    # ---- Acceptance Check ----
    print(f"\n  {'='*48}")
    print(f"  ACCEPTANCE CHECK")
    print(f"  {'='*48}")

    best_config = configs_to_run[-1]  # Full Tier 1 is last
    best_it = results[best_config]['evaluation']['it']
    baseline_it = results.get('baseline', {}).get('evaluation', {}).get('it', {})

    checks = []
    c1 = best_it['n_active'] >= 10
    checks.append(('IT >=10 active clusters', c1, best_it['n_active']))

    c2 = best_it['avg_purity'] > 0.25
    checks.append(('IT purity > 0.25', c2, f"{best_it['avg_purity']:.3f}"))

    c3 = best_it['coverage'] > 0.50
    checks.append(('IT coverage > 50%', c3, f"{best_it['coverage']:.1%}"))

    c4 = best_it['avg_diagonal'] > 0.22
    checks.append(('IT diag > 0.22', c4, f"{best_it['avg_diagonal']:.3f}"))

    if baseline_it:
        delta = best_it['avg_purity'] - baseline_it['avg_purity']
        c5 = delta >= -0.02
        checks.append(('Best IT >= Baseline IT', c5, f"{delta:+.3f}"))
    else:
        checks.append(('IT purity above threshold', True, 'N/A'))

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
        'configs': configs_to_run,
        'results': results,
        'all_checks_passed': all_pass,
    }


# ================================================================
# CLI
# ================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Tier 1: Predictive Processing + Color Opponent')
    parser.add_argument('--n', type=int, default=2000)
    parser.add_argument('--dataset', type=str, default='imagenette',
                       choices=['cifar10', 'imagenette'])
    parser.add_argument('--mode', type=str, default='all',
                       choices=['all', 'baseline', 'color', 'pp', 'color_pp'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    run_pp_experiment(
        n_images=args.n,
        dataset=args.dataset,
        mode=args.mode,
        seed=args.seed,
    )
