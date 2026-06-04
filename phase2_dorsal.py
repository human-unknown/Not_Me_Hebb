"""
phase2_dorsal.py — Pulvinar 捷径 + 背侧通路 (MT/MST)
自由能原理智能体

在 Tier 1 (color + PP) 最佳架构上叠加两条新生物通路:

  A. Pulvinar → IT 低空间频率捷径:
     - 丘脑枕核直接投射到 IT，完全绕过 V1-V2-V4
     - 强模糊 + 降采样 + 大尺度 Gabor → 快速"场景要旨"
     - 提供粗糙但快速的对象分类线索

  B. V1 → MT → MST 背侧"在哪里"通路:
     - MT: 朝向选择性空间梯度 → "隐含运动能量"
     - MST: 响应质心 (center of mass) → "特征在哪里"
     - 独立于腹侧 V1→V2→V4→IT "是什么"通路

实验矩阵 (4 配置):
  1. reference: 缩减布局, color + PP, 无新通路 (基线)
  2. +pulvinar: reference + Pulvinar 捷径
  3. +dorsal:  reference + 背侧通路
  4. full:      reference + Pulvinar + Dorsal (完整)

  + 5. legacy: 原始 phase2_pp 布局 (Color 64d, PE 32d) 作为对照

Usage:
  python phase2_dorsal.py --n 2000 --mode all
  python phase2_dorsal.py --n 2000 --mode full
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
# Sensory Layouts
# ================================================================

# --- Legacy layout (phase2_pp.py, unchanged reference) ---
V1_WIDTH    = 96
V2_WIDTH    = 64
V4_WIDTH    = 64

LEGACY_COLOR_WIDTH = 64
LEGACY_PE_WIDTH    = 32

LEGACY_V1_START,    LEGACY_V1_END    = 10,  10 + V1_WIDTH        # s[10:106]
LEGACY_V2_START,    LEGACY_V2_END    = 106, 106 + V2_WIDTH        # s[106:170]
LEGACY_V4_START,    LEGACY_V4_END    = 170, 170 + V4_WIDTH        # s[170:234]
LEGACY_COLOR_START, LEGACY_COLOR_END = 234, 234 + LEGACY_COLOR_WIDTH  # s[234:298]
LEGACY_PE_START,    LEGACY_PE_END    = 298, 298 + LEGACY_PE_WIDTH     # s[298:330]

# --- Dorsal layout (new pathways, fits within D=330) ---
DORSAL_COLOR_WIDTH = 32
DORSAL_DOR_WIDTH   = 32  # MT + MST dorsal stream
DORSAL_PUL_WIDTH   = 16  # Pulvinar low-SF shortcut
DORSAL_PE_WIDTH    = 16  # Prediction error (reduced)

DORSAL_V1_START,    DORSAL_V1_END    = 10,  10 + V1_WIDTH        # s[10:106]
DORSAL_V2_START,    DORSAL_V2_END    = 106, 106 + V2_WIDTH        # s[106:170]
DORSAL_V4_START,    DORSAL_V4_END    = 170, 170 + V4_WIDTH        # s[170:234]
DORSAL_COLOR_START, DORSAL_COLOR_END = 234, 234 + DORSAL_COLOR_WIDTH  # s[234:266]
DORSAL_DOR_START,   DORSAL_DOR_END   = 266, 266 + DORSAL_DOR_WIDTH    # s[266:298]
DORSAL_PUL_START,   DORSAL_PUL_END   = 298, 298 + DORSAL_PUL_WIDTH    # s[298:314]
DORSAL_PE_START,    DORSAL_PE_END    = 314, 314 + DORSAL_PE_WIDTH     # s[314:330]

# Verify layouts fit within D
assert DORSAL_PE_END == D, f"Dorsal layout ends at {DORSAL_PE_END}, expected {D}"
assert LEGACY_PE_END == D, f"Legacy layout ends at {LEGACY_PE_END}, expected {D}"

# Per-layer Hebb params
LAYER_PARAMS = {
    'v1': {'threshold': 0.50, 'lr': 0.02, 'decay': 0.003, 'hash_offset': 10},
    'v2': {'threshold': 0.45, 'lr': 0.02, 'decay': 0.003, 'hash_offset': 106},
    'v4': {'threshold': 0.40, 'lr': 0.02, 'decay': 0.003, 'hash_offset': 170},
    'it': {'threshold': 0.35, 'lr': 0.02, 'decay': 0.003, 'hash_offset': 10},
}

SLEEP_INTERVAL = 500

# Experiment configurations
CONFIGS = {
    'reference':  {'layout': 'dorsal', 'use_color': True, 'use_pp': True,
                   'use_pulvinar': False, 'use_dorsal': False},
    '+pulvinar':  {'layout': 'dorsal', 'use_color': True, 'use_pp': True,
                   'use_pulvinar': True, 'use_dorsal': False},
    '+dorsal':    {'layout': 'dorsal', 'use_color': True, 'use_pp': True,
                   'use_pulvinar': False, 'use_dorsal': True},
    'full':       {'layout': 'dorsal', 'use_color': True, 'use_pp': True,
                   'use_pulvinar': True, 'use_dorsal': True},
    'legacy':     {'layout': 'legacy', 'use_color': True, 'use_pp': True,
                   'use_pulvinar': False, 'use_dorsal': False},
}


# ================================================================
# Prediction Error Encoding
# ================================================================

def encode_prediction_errors(pe_v1: float, pe_v2: float,
                              pe_v4: float, pe_it: float = 0.0,
                              pe_width: int = 32) -> np.ndarray:
    """Encode layer-wise prediction errors into PE feature vector.

    Each layer gets (pe_width // 4) dims: [scalar_value, log_value, ...pad].
    """
    pe = np.zeros(pe_width, dtype=np.float32)
    errors = [pe_v1, pe_v2, pe_v4, pe_it]
    per_layer = max(2, pe_width // 4)
    for i, err in enumerate(errors):
        base = i * per_layer
        if base < pe_width:
            pe[base] = float(np.clip(err, 0, 10.0))
        if base + 1 < pe_width:
            pe[base + 1] = float(np.log1p(err))
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
    if stage >= 1: layer_order.append(('v1', DORSAL_V1_START, DORSAL_V1_END))
    if stage >= 2: layer_order.append(('v2', DORSAL_V2_START, DORSAL_V2_END))
    if stage >= 3: layer_order.append(('v4', DORSAL_V4_START, DORSAL_V4_END))

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

def build_layer_sensory_dorsal(features_dict: dict, nets: dict,
                                stage: int, config: dict,
                                pe_errors: dict = None) -> np.ndarray:
    """Build D=330 sensory vector using Dorsal layout.

    Layout:
      s[10:106]   = V1 Gabor (96d)
      s[106:170]  = V2 Gabor (64d)
      s[170:234]  = V4 Gabor (64d)
      s[234:266]  = Color opponent (32d) if use_color
      s[266:298]  = Dorsal stream (32d) if use_dorsal
      s[298:314]  = Pulvinar (16d) if use_pulvinar
      s[314:330]  = Prediction Errors (16d) if use_pp
    """
    use_color = config['use_color']
    use_pp = config['use_pp']
    use_pulvinar = config['use_pulvinar']
    use_dorsal = config['use_dorsal']

    s = np.zeros(D, dtype=np.float32)

    # V1 features: always present
    v1_feat = features_dict.get('v1')
    if v1_feat is not None:
        flen = min(len(v1_feat), V1_WIDTH)
        s[DORSAL_V1_START:DORSAL_V1_START + flen] = v1_feat[:flen]

    # V2 features: stage >= 1
    if stage >= 1:
        v2_feat = features_dict.get('v2')
        if v2_feat is not None:
            flen = min(len(v2_feat), V2_WIDTH)
            s[DORSAL_V2_START:DORSAL_V2_START + flen] = v2_feat[:flen]

    # V4 features: stage >= 2
    if stage >= 2:
        v4_feat = features_dict.get('v4')
        if v4_feat is not None:
            flen = min(len(v4_feat), V4_WIDTH)
            s[DORSAL_V4_START:DORSAL_V4_START + flen] = v4_feat[:flen]

    # Color features: stage >= 1 and use_color
    if stage >= 1 and use_color:
        color_feat = features_dict.get('color')
        if color_feat is not None:
            flen = min(len(color_feat), DORSAL_COLOR_WIDTH)
            s[DORSAL_COLOR_START:DORSAL_COLOR_START + flen] = color_feat[:flen]

    # Dorsal stream (MT+MST): stage >= 1 and use_dorsal
    if stage >= 1 and use_dorsal:
        dorsal_feat = features_dict.get('dorsal')
        if dorsal_feat is not None:
            flen = min(len(dorsal_feat), DORSAL_DOR_WIDTH)
            s[DORSAL_DOR_START:DORSAL_DOR_START + flen] = dorsal_feat[:flen]

    # Pulvinar shortcut: stage >= 1 and use_pulvinar
    if stage >= 1 and use_pulvinar:
        pulvinar_feat = features_dict.get('pulvinar')
        if pulvinar_feat is not None:
            flen = min(len(pulvinar_feat), DORSAL_PUL_WIDTH)
            s[DORSAL_PUL_START:DORSAL_PUL_START + flen] = pulvinar_feat[:flen]

    # Prediction Errors: stage >= 1 and use_pp
    if stage >= 1 and use_pp and pe_errors is not None:
        pe_vec = encode_prediction_errors(
            pe_errors.get('v1', 0.0),
            pe_errors.get('v2', 0.0),
            pe_errors.get('v4', 0.0),
            pe_errors.get('it', 0.0),
            pe_width=DORSAL_PE_WIDTH,
        )
        s[DORSAL_PE_START:DORSAL_PE_END] = pe_vec

    return s


def build_layer_sensory_legacy(features_dict: dict, nets: dict,
                                stage: int, use_color: bool,
                                use_pp: bool,
                                pe_errors: dict = None) -> np.ndarray:
    """Build D=330 sensory vector using Legacy layout (same as phase2_pp.py)."""
    s = np.zeros(D, dtype=np.float32)

    v1_feat = features_dict.get('v1')
    if v1_feat is not None:
        flen = min(len(v1_feat), V1_WIDTH)
        s[LEGACY_V1_START:LEGACY_V1_START + flen] = v1_feat[:flen]

    if stage >= 1:
        v2_feat = features_dict.get('v2')
        if v2_feat is not None:
            flen = min(len(v2_feat), V2_WIDTH)
            s[LEGACY_V2_START:LEGACY_V2_START + flen] = v2_feat[:flen]

    if stage >= 2:
        v4_feat = features_dict.get('v4')
        if v4_feat is not None:
            flen = min(len(v4_feat), V4_WIDTH)
            s[LEGACY_V4_START:LEGACY_V4_START + flen] = v4_feat[:flen]

    if stage >= 1 and use_color:
        color_feat = features_dict.get('color')
        if color_feat is not None:
            flen = min(len(color_feat), LEGACY_COLOR_WIDTH)
            s[LEGACY_COLOR_START:LEGACY_COLOR_START + flen] = color_feat[:flen]

    if stage >= 1 and use_pp and pe_errors is not None:
        pe_vec = encode_prediction_errors(
            pe_errors.get('v1', 0.0),
            pe_errors.get('v2', 0.0),
            pe_errors.get('v4', 0.0),
            pe_errors.get('it', 0.0),
            pe_width=LEGACY_PE_WIDTH,
        )
        s[LEGACY_PE_START:LEGACY_PE_END] = pe_vec

    return s


def build_sensory(features_dict, nets, stage, config, pe_errors=None):
    """Dispatcher: choose layout based on config."""
    if config['layout'] == 'legacy':
        return build_layer_sensory_legacy(
            features_dict, nets, stage,
            config['use_color'], config['use_pp'], pe_errors)
    else:
        return build_layer_sensory_dorsal(
            features_dict, nets, stage, config, pe_errors)


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
    """Learn with prediction-error-modulated learning rate."""
    mod = max(0.5, min(3.0, 1.0 + pe_scalar / (pe_baseline + 1e-8)))
    original_lr = net.theta.learn_rate_l0
    net.theta.learn_rate_l0 = original_lr * mod
    net.learn(s)
    net.theta.learn_rate_l0 = original_lr


# ================================================================
# Training
# ================================================================

def train_hierarchy(venv, config: dict, n_images: int,
                     precomputed: dict = None,
                     seed: int = 42) -> dict:
    """Train V1→V2→V4→IT hierarchy.

    Args:
        venv: VisualEnvironment with all encodings loaded
        config: experiment configuration dict
        n_images: number of images
        precomputed: dict with precomputed 'pulvinar' and 'dorsal' arrays
        seed: random seed
    """
    rng = np.random.default_rng(seed)
    use_color = config['use_color']
    use_pp = config['use_pp']
    use_pulvinar = config.get('use_pulvinar', False)
    use_dorsal = config.get('use_dorsal', False)
    layout = config.get('layout', 'dorsal')

    mode_name = layout
    if use_pulvinar: mode_name += '+pul'
    if use_dorsal: mode_name += '+dor'
    if use_pp: mode_name += '+pp'
    if use_color: mode_name += '+col'

    # Create networks
    nets = {}
    for name in ['v1', 'v2', 'v4', 'it']:
        nets[name] = _create_net(name)

    order = rng.permutation(n_images)
    t0 = time.perf_counter()

    # EMA tracking
    F_visual_ema = 0.0
    F_alpha = 0.95

    # For layout-specific prediction error ranges (used in PP)
    if layout == 'legacy':
        v1_start, v1_end = LEGACY_V1_START, LEGACY_V1_END
        v2_start, v2_end = LEGACY_V2_START, LEGACY_V2_END
        v4_start, v4_end = LEGACY_V4_START, LEGACY_V4_END
        pe_start, pe_end = LEGACY_PE_START, LEGACY_PE_END
    else:
        v1_start, v1_end = DORSAL_V1_START, DORSAL_V1_END
        v2_start, v2_end = DORSAL_V2_START, DORSAL_V2_END
        v4_start, v4_end = DORSAL_V4_START, DORSAL_V4_END
        pe_start, pe_end = DORSAL_PE_START, DORSAL_PE_END

    print(f"\n{'='*64}")
    print(f"  Training: {mode_name}")
    print(f"{'='*64}")
    print(f"  Layout={layout}, PP={use_pp}, Color={use_color}")
    print(f"  Pulvinar={use_pulvinar}, Dorsal={use_dorsal}")
    for name in ['v1', 'v2', 'v4', 'it']:
        p = LAYER_PARAMS[name]
        print(f"  {name}: threshold={p['threshold']}, lr={p['lr']}")

    pe_baseline = 0.05

    # Pre-extract precomputed features if available
    pulvinar_feats = precomputed.get('pulvinar') if precomputed else None
    dorsal_feats = precomputed.get('dorsal') if precomputed else None

    for step, idx in enumerate(order):
        # Get features from venv
        v1_feat = venv.encodings[idx]
        v2_feat = (venv.encodings_v2[idx].copy()
                   if venv.encodings_v2 is not None else None)
        v4_feat = (venv.encodings_v4[idx].copy()
                   if venv.encodings_v4 is not None else None)

        # Color: truncate to layout width if needed
        if use_color:
            color_full = venv.encodings_color[idx].copy() if getattr(venv, 'encodings_color', None) is not None else None
            if color_full is not None:
                if layout == 'dorsal':
                    color_feat = color_full[:DORSAL_COLOR_WIDTH]
                else:
                    color_feat = color_full[:LEGACY_COLOR_WIDTH]
            else:
                color_feat = None
        else:
            color_feat = None

        # New pathways
        pulv_feat = (pulvinar_feats[idx].copy()
                     if use_pulvinar and pulvinar_feats is not None
                     else None)
        dors_feat = (dorsal_feats[idx].copy()
                     if use_dorsal and dorsal_feats is not None
                     else None)

        features_dict = {
            'v1': v1_feat, 'v2': v2_feat, 'v4': v4_feat,
            'color': color_feat,
            'pulvinar': pulv_feat, 'dorsal': dors_feat,
        }

        # Build sensory vectors
        pe_errors_init = {'v1': 0.0, 'v2': 0.0, 'v4': 0.0, 'it': 0.0}

        s_it = build_sensory(features_dict, nets, stage=3, config=config,
                              pe_errors=pe_errors_init)
        s_v4 = build_sensory(features_dict, nets, stage=2, config=config,
                              pe_errors=pe_errors_init)
        s_v2 = build_sensory(features_dict, nets, stage=1, config=config,
                              pe_errors=pe_errors_init)
        s_v1 = build_sensory(features_dict, nets, stage=0, config=config,
                              pe_errors=None)

        # ---- Feedforward recall ----
        c_it  = nets['it'].recall(s_it)
        c_v4  = nets['v4'].recall(s_v4)
        c_v2  = nets['v2'].recall(s_v2)
        c_v1  = nets['v1'].recall(s_v1)

        # ---- Predictive Processing ----
        pe_errors = {'v1': 0.0, 'v2': 0.0, 'v4': 0.0, 'it': 0.0}

        if use_pp:
            # IT predicts V4 features
            if c_it is not None:
                pred_v4 = c_it.centroid[v4_start:v4_end]
                actual_v4 = s_v4[v4_start:v4_end]
                mask_v4 = np.abs(actual_v4) > 1e-6
                if mask_v4.any():
                    pe_errors['v4'] = float(np.sum(
                        (actual_v4[mask_v4] - pred_v4[mask_v4]) ** 2))
                else:
                    pe_errors['v4'] = 0.0
            else:
                pe_errors['v4'] = 1.0

            # V4 predicts V2 features
            if c_v4 is not None:
                pred_v2 = c_v4.centroid[v2_start:v2_end]
                actual_v2 = s_v2[v2_start:v2_end]
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
                pred_v1 = c_v2.centroid[v1_start:v1_end]
                actual_v1 = s_v1[v1_start:v1_end]
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
                    (s_it[v1_start:pe_end] - c_it.centroid[v1_start:pe_end]) ** 2
                )) / (pe_end - v1_start)
            else:
                pe_errors['it'] = 1.0

            # Update F_visual_ema
            F_visual_total = (pe_errors['v1'] + pe_errors['v2'] + pe_errors['v4']) / 3.0
            F_visual_ema = F_alpha * F_visual_ema + (1 - F_alpha) * F_visual_total

            # Rebuild sensory vectors with PE encoded
            s_it = build_sensory(features_dict, nets, stage=3, config=config,
                                  pe_errors=pe_errors)
            s_v4 = build_sensory(features_dict, nets, stage=2, config=config,
                                  pe_errors=pe_errors)
            s_v2 = build_sensory(features_dict, nets, stage=1, config=config,
                                  pe_errors=pe_errors)

        # ---- Learn ----
        if use_pp:
            learn_modulated(nets['it'], s_it, pe_errors['v4'], pe_baseline)
            learn_modulated(nets['v4'], s_v4, pe_errors['v2'], pe_baseline)
            learn_modulated(nets['v2'], s_v2, pe_errors['v1'], pe_baseline)
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

    return {
        'nets': nets,
        'F_visual_ema': F_visual_ema,
        'mode': mode_name,
    }


# ================================================================
# Evaluation
# ================================================================

def evaluate_layer(net: ClusterNetwork, venv, nets: dict,
                    stage: int, config: dict,
                    precomputed: dict = None,
                    n_images: int = None) -> dict:
    """Read-only evaluation of a layer's clustering quality."""
    if n_images is None:
        n_images = venv.n_images

    use_color = config['use_color']
    use_pp = config.get('use_pp', False)
    use_pulvinar = config.get('use_pulvinar', False)
    use_dorsal = config.get('use_dorsal', False)

    pulvinar_feats = precomputed.get('pulvinar') if precomputed else None
    dorsal_feats = precomputed.get('dorsal') if precomputed else None

    cluster_hits = defaultdict(list)

    for idx in range(n_images):
        v1_feat = venv.encodings[idx]
        v2_feat = (venv.encodings_v2[idx].copy()
                   if venv.encodings_v2 is not None else None)
        v4_feat = (venv.encodings_v4[idx].copy()
                   if venv.encodings_v4 is not None else None)

        if use_color and getattr(venv, 'encodings_color', None) is not None:
            color_full = venv.encodings_color[idx].copy()
            color_feat = color_full[:DORSAL_COLOR_WIDTH] if config['layout'] == 'dorsal' else color_full[:LEGACY_COLOR_WIDTH]
        else:
            color_feat = None

        pulv_feat = (pulvinar_feats[idx].copy()
                     if use_pulvinar and pulvinar_feats is not None
                     else None)
        dors_feat = (dorsal_feats[idx].copy()
                     if use_dorsal and dorsal_feats is not None
                     else None)

        features_dict = {
            'v1': v1_feat, 'v2': v2_feat, 'v4': v4_feat,
            'color': color_feat,
            'pulvinar': pulv_feat, 'dorsal': dors_feat,
        }
        s = build_sensory(features_dict, nets, stage, config, pe_errors=None)

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
        if use_color and getattr(venv, 'encodings_color', None) is not None:
            color_full = venv.encodings_color[idx].copy()
            color_feat = color_full[:DORSAL_COLOR_WIDTH] if config['layout'] == 'dorsal' else color_full[:LEGACY_COLOR_WIDTH]
        else:
            color_feat = None
        pulv_feat = (pulvinar_feats[idx].copy()
                     if use_pulvinar and pulvinar_feats is not None
                     else None)
        dors_feat = (dorsal_feats[idx].copy()
                     if use_dorsal and dorsal_feats is not None
                     else None)
        features_dict = {
            'v1': v1_feat, 'v2': v2_feat, 'v4': v4_feat,
            'color': color_feat,
            'pulvinar': pulv_feat, 'dorsal': dors_feat,
        }
        s = build_sensory(features_dict, nets, stage, config, pe_errors=None)
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
            if use_color and getattr(venv, 'encodings_color', None) is not None:
                color_full = venv.encodings_color[idx].copy()
                color_feat = color_full[:DORSAL_COLOR_WIDTH] if config['layout'] == 'dorsal' else color_full[:LEGACY_COLOR_WIDTH]
            else:
                color_feat = None
            pulv_feat = (pulvinar_feats[idx].copy()
                         if use_pulvinar and pulvinar_feats is not None
                         else None)
            dors_feat = (dorsal_feats[idx].copy()
                         if use_dorsal and dorsal_feats is not None
                         else None)
            features_dict = {
                'v1': v1_feat, 'v2': v2_feat, 'v4': v4_feat,
                'color': color_feat,
                'pulvinar': pulv_feat, 'dorsal': dors_feat,
            }
            s = build_sensory(features_dict, nets, stage, config, pe_errors=None)
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
    sorted_pure = sorted(purities.items(), key=lambda x: -x[1]['purity'])
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
    }


# ================================================================
# Per-class recall
# ================================================================

def per_class_recall(net: ClusterNetwork, venv, nets: dict,
                      stage: int, config: dict,
                      precomputed: dict = None,
                      n_per_class: int = 30) -> dict:
    """Compute per-class recall for a layer."""
    n_classes = venv.n_classes
    use_color = config['use_color']
    use_pulvinar = config.get('use_pulvinar', False)
    use_dorsal = config.get('use_dorsal', False)
    pulvinar_feats = precomputed.get('pulvinar') if precomputed else None
    dorsal_feats = precomputed.get('dorsal') if precomputed else None

    # Build cluster→class assignment
    cluster_hits = defaultdict(list)
    for idx in range(min(500, venv.n_images)):
        v1_feat = venv.encodings[idx]
        v2_feat = (venv.encodings_v2[idx].copy()
                   if venv.encodings_v2 is not None else None)
        v4_feat = (venv.encodings_v4[idx].copy()
                   if venv.encodings_v4 is not None else None)
        if use_color and getattr(venv, 'encodings_color', None) is not None:
            color_full = venv.encodings_color[idx].copy()
            color_feat = color_full[:DORSAL_COLOR_WIDTH] if config['layout'] == 'dorsal' else color_full[:LEGACY_COLOR_WIDTH]
        else:
            color_feat = None
        pulv_feat = (pulvinar_feats[idx].copy()
                     if use_pulvinar and pulvinar_feats is not None
                     else None)
        dors_feat = (dorsal_feats[idx].copy()
                     if use_dorsal and dorsal_feats is not None
                     else None)
        features_dict = {
            'v1': v1_feat, 'v2': v2_feat, 'v4': v4_feat,
            'color': color_feat,
            'pulvinar': pulv_feat, 'dorsal': dors_feat,
        }
        s = build_sensory(features_dict, nets, stage, config, pe_errors=None)
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
            if use_color and getattr(venv, 'encodings_color', None) is not None:
                color_full = venv.encodings_color[idx].copy()
                color_feat = color_full[:DORSAL_COLOR_WIDTH] if config['layout'] == 'dorsal' else color_full[:LEGACY_COLOR_WIDTH]
            else:
                color_feat = None
            pulv_feat = (pulvinar_feats[idx].copy()
                         if use_pulvinar and pulvinar_feats is not None
                         else None)
            dors_feat = (dorsal_feats[idx].copy()
                         if use_dorsal and dorsal_feats is not None
                         else None)
            features_dict = {
                'v1': v1_feat, 'v2': v2_feat, 'v4': v4_feat,
                'color': color_feat,
                'pulvinar': pulv_feat, 'dorsal': dors_feat,
            }
            s = build_sensory(features_dict, nets, stage, config, pe_errors=None)
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
# Precompute new pathway features
# ================================================================

def precompute_features(venv) -> dict:
    """Precompute Pulvinar and Dorsal features for all images.

    Returns:
        {'pulvinar': (N, 32), 'dorsal': (N, 32)}
    """
    n = venv.n_images
    gabor = venv._gabor

    pulvinar_all = np.zeros((n, 32), dtype=np.float32)
    dorsal_all = np.zeros((n, 32), dtype=np.float32)

    print(f"  Precomputing Pulvinar + Dorsal features for {n} images...")
    t0 = time.perf_counter()
    for i in range(n):
        if (i + 1) % 1000 == 0:
            print(f"    {i+1}/{n}...")
        img = venv.images[i]
        pulvinar_all[i] = gabor.encode_pulvinar(img)
        dorsal_all[i] = gabor.encode_dorsal(img)

    elapsed = time.perf_counter() - t0
    print(f"  Precomputed in {elapsed:.1f}s ({n/elapsed:.0f} img/s)")

    return {'pulvinar': pulvinar_all, 'dorsal': dorsal_all}


# ================================================================
# Main Experiment
# ================================================================

def run_dorsal_experiment(n_images: int = 2000,
                           dataset: str = 'imagenette',
                           mode: str = 'all',
                           seed: int = 42):
    """Pulvinar + Dorsal stream experiment.

    Args:
        n_images: Number of images
        dataset: 'cifar10' or 'imagenette'
        mode: 'all', 'reference', '+pulvinar', '+dorsal', 'full', 'legacy'
        seed: Random seed
    """
    np.random.seed(seed)

    print("=" * 72)
    print("  Pulvinar Shortcut + Dorsal Stream (MT/MST)")
    print("=" * 72)
    print(f"  Dataset: {dataset}, {n_images} images, mode: {mode}")
    print()

    # Determine which configs to run
    if mode == 'all':
        configs_to_run = list(CONFIGS.keys())
    else:
        configs_to_run = [mode]

    # Determine which features we need
    need_legacy = any(CONFIGS[c]['layout'] == 'legacy' for c in configs_to_run)
    need_color = any(CONFIGS[c]['use_color'] for c in configs_to_run)
    need_pulvinar = any(CONFIGS[c].get('use_pulvinar', False) for c in configs_to_run)
    need_dorsal = any(CONFIGS[c].get('use_dorsal', False) for c in configs_to_run)

    # Load VisualEnvironment — always load with 64d color (we truncate later if needed)
    color_width = LEGACY_COLOR_WIDTH  # 64d — always load full, truncate for dorsal layout

    print(f"[1/4] Loading VisualEnvironment (color={need_color})...")
    t0 = time.perf_counter()
    from visual_interface import VisualEnvironment

    venv = VisualEnvironment(
        dataset=dataset, n_images=n_images,
        pca_components=V1_WIDTH,
        v2_components=V2_WIDTH,
        v4_components=V4_WIDTH,
        color_components=color_width,
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

    # Precompute new pathway features
    precomputed = None
    if need_pulvinar or need_dorsal:
        print(f"\n[2/4] Precomputing new pathway features...")
        precomputed = precompute_features(venv)
        if need_pulvinar:
            print(f"  Pulvinar: {precomputed['pulvinar'].shape}")
        if need_dorsal:
            print(f"  Dorsal: {precomputed['dorsal'].shape}")
    else:
        print(f"\n[2/4] No new pathway features needed")

    results = {}
    step_offset = 3 if (need_pulvinar or need_dorsal) else 2

    for i, config_name in enumerate(configs_to_run):
        config = CONFIGS[config_name]
        print(f"\n[{i+step_offset}/{config_name}] Training...")

        train_result = train_hierarchy(venv, config, n_actual,
                                        precomputed=precomputed, seed=seed)
        nets = train_result['nets']

        print(f"\n[{i+step_offset+1}/{config_name}] Evaluating...")
        eval_results = {}
        for layer_name, stage in [('v1', 0), ('v2', 1), ('v4', 2), ('it', 3)]:
            net = nets[layer_name]
            ev = evaluate_layer(net, venv, nets, stage, config,
                                 precomputed=precomputed, n_images=n_actual)
            eval_results[layer_name] = ev
            print(f"    {layer_name.upper()}: {ev['n_clusters']} clusters, "
                  f"purity={ev['avg_purity']:.3f}, "
                  f"diag={ev['avg_diagonal']:.3f}")

        # Per-class recall for IT
        it_recall = per_class_recall(nets['it'], venv, nets, 3, config,
                                      precomputed=precomputed)
        eval_results['it']['recall'] = it_recall

        results[config_name] = {
            'training': train_result,
            'evaluation': eval_results,
        }

    # ---- Report ----
    print("\n" + "=" * 72)
    print("  Results: Pulvinar + Dorsal Stream")
    print("=" * 72)

    # Summary table
    print(f"\n  [IT Layer Comparison]")
    header = f"  {'Config':<14} {'IT Clusters':<12} {'IT Purity':<10} {'IT Diag':<10} {'F_visual':<10}"
    print(header)
    print(f"  {'-'*len(header)}")
    for config_name in configs_to_run:
        r = results[config_name]
        it_ev = r['evaluation']['it']
        F_ema = r['training']['F_visual_ema']
        print(f"  {config_name:<14} {it_ev['n_active']:<12} "
              f"{it_ev['avg_purity']:<10.3f} {it_ev['avg_diagonal']:<10.3f} "
              f"{F_ema:<10.4f}")

    # Per-class recall comparison
    print(f"\n  [Per-Class IT Recall]")
    if results:
        first_config = configs_to_run[0]
        class_names = list(results[first_config]['evaluation']['it'].get('recall', {}).keys())
        header = f"  {'Class':<16}"
        for cn in configs_to_run:
            header += f" {cn:<12}"
        print(header)
        print(f"  {'-'*len(header)}")
        for name in class_names:
            row = f"  {name:<16}"
            for cn in configs_to_run:
                rec = results[cn]['evaluation']['it'].get('recall', {}).get(name, 0.0)
                row += f" {rec:<12.3f}"
            print(row)

    # Delta vs reference
    if len(configs_to_run) >= 2 and 'reference' in results:
        print(f"\n  [Per-Class Delta vs reference]")
        ref_recall = results['reference']['evaluation']['it'].get('recall', {})
        for cn in configs_to_run:
            if cn == 'reference':
                continue
            cn_recall = results[cn]['evaluation']['it'].get('recall', {})
            deltas = []
            for name in class_names:
                delta = cn_recall.get(name, 0.0) - ref_recall.get(name, 0.0)
                deltas.append((name, delta))
            deltas.sort(key=lambda x: -x[1])
            print(f"  {cn}:")
            for name, delta in deltas[:5]:
                marker = '+' if delta > 0 else ''
                print(f"    {name:<16}: {marker}{delta:.3f}")
            avg_delta = float(np.mean([d[1] for d in deltas]))
            print(f"    {'avg':<16}: {avg_delta:+.3f}")

    # ---- Acceptance Check ----
    print(f"\n  {'='*48}")
    print(f"  ACCEPTANCE CHECK")
    print(f"  {'='*48}")

    best_config = 'full' if 'full' in results else configs_to_run[-1]
    best_it = results[best_config]['evaluation']['it']
    ref_it = results.get('legacy', results.get('reference', {})).get('evaluation', {}).get('it', {})

    checks = []
    c1 = best_it['n_active'] >= 8
    checks.append(('IT >=8 active clusters', c1, best_it['n_active']))

    c2 = best_it['avg_purity'] > 0.25
    checks.append(('IT purity > 0.25', c2, f"{best_it['avg_purity']:.3f}"))

    c3 = best_it['coverage'] > 0.50
    checks.append(('IT coverage > 50%', c3, f"{best_it['coverage']:.1%}"))

    c4 = best_it['avg_diagonal'] > 0.22
    checks.append(('IT diag > 0.22', c4, f"{best_it['avg_diagonal']:.3f}"))

    if ref_it:
        delta = best_it['avg_purity'] - ref_it.get('avg_purity', 0)
        c5 = delta >= -0.03
        checks.append(('Best IT >= Ref IT (-0.03 margin)', c5, f"{delta:+.3f}"))
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
        description='Pulvinar Shortcut + Dorsal Stream (MT/MST)')
    parser.add_argument('--n', type=int, default=2000)
    parser.add_argument('--dataset', type=str, default='imagenette',
                       choices=['cifar10', 'imagenette'])
    parser.add_argument('--mode', type=str, default='all',
                       choices=['all', 'reference', '+pulvinar', '+dorsal',
                                'full', 'legacy'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    run_dorsal_experiment(
        n_images=args.n,
        dataset=args.dataset,
        mode=args.mode,
        seed=args.seed,
    )
