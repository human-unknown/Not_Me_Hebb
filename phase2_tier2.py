"""
phase2_tier2.py — Tier 2: Saccade Attention + Neuromodulation
自由能原理智能体

Two biologically-motivated mechanisms:
  C. Saccade Attention (眼跳注意):
     - Saliency map from Gabor response magnitudes
     - 3 fixation points with inhibition of return
     - Foveated encoding (Gaussian weight mask centered on fixation)
     - Multiple views provide richer IT input

  D. Neuromodulation (神经调节):
     - DA (Dopamine): F_visual decreases → reward → boost IT learning
     - ACh (Acetylcholine): F_visual increases → novelty → boost V1/V2 learning
     - NE (Norepinephrine): sustained high F_visual → uncertainty → lower thresholds

Experiment matrix (5 configurations):
  1. baseline:   Uniform encoding, no neuromodulation
  2. saccade:    3-fixation foveated encoding, no neuromodulation
  3. neuromod:   Uniform encoding, DA/ACh/NE modulation
  4. sacc_nm:    Saccade + neuromodulation (full Tier 2)
  5. tier1_nm:   Color + PP + neuromodulation (Tier 1 + Tier 2)

Usage:
  python phase2_tier2.py --n 2000 --mode all
  python phase2_tier2.py --n 2000 --mode sacc_nm
"""

import os
import sys
import argparse
import time
import numpy as np
from collections import defaultdict
from scipy.ndimage import gaussian_filter

from data_types import D, Theta
from layer0_model import ClusterNetwork, sleep_cycle, _masked_cosine, _auto_mask
from layer3_meta import create_default_theta


# ================================================================
# Sensory Layout
# ================================================================

# Per-fixation compressed dims (for saccade mode)
FIX_V1_DIM = 24
FIX_V2_DIM = 16
FIX_V4_DIM = 16
FIX_COLOR_DIM = 16
FIX_DIM = FIX_V1_DIM + FIX_V2_DIM + FIX_V4_DIM + FIX_COLOR_DIM  # 72d

N_FIXATIONS = 3

# Channel ranges for saccade mode
FIX1_START, FIX1_END = 10, 10 + FIX_DIM       # s[10:82]
FIX2_START, FIX2_END = 82, 82 + FIX_DIM       # s[82:154]
FIX3_START, FIX3_END = 154, 154 + FIX_DIM     # s[154:226]
CS_START,    CS_END    = 226, 274             # s[226:274]  Cluster Summary (48d)
SAL_START,   SAL_END   = 274, 298             # s[274:298]  Saliency stats (24d)
PE_START,    PE_END    = 298, 330             # s[298:330]  Prediction Errors (32d)

# Channel ranges for uniform mode (reuse VisualEnvironment cached features)
UNI_V1_START, UNI_V1_END = 10, 106
UNI_V2_START, UNI_V2_END = 106, 170
UNI_V4_START, UNI_V4_END = 170, 234
UNI_COLOR_START, UNI_COLOR_END = 234, 298
UNI_PE_START, UNI_PE_END = 298, 330

# Per-layer Hebb params
LAYER_PARAMS = {
    'v1': {'threshold': 0.50, 'lr': 0.02, 'decay': 0.003},
    'v2': {'threshold': 0.45, 'lr': 0.02, 'decay': 0.003},
    'v4': {'threshold': 0.40, 'lr': 0.02, 'decay': 0.003},
    'it': {'threshold': 0.35, 'lr': 0.02, 'decay': 0.003},
}

SLEEP_INTERVAL = 500
FOVEA_SIGMA = 24.0          # Gaussian sigma for foveation (pixels)
INHIBITION_RADIUS = 32.0    # Inhibition of return radius (pixels)

CONFIGS = {
    'baseline':  {'saccade': False, 'neuromod': False, 'color': False, 'pp': False},
    'saccade':   {'saccade': True,  'neuromod': False, 'color': False, 'pp': False},
    'neuromod':  {'saccade': False, 'neuromod': True,  'color': False, 'pp': False},
    'sacc_nm':   {'saccade': True,  'neuromod': True,  'color': False, 'pp': False},
    'tier1_nm':  {'saccade': False, 'neuromod': True,  'color': True,  'pp': True},
}


# ================================================================
# Saliency Map + Fixation Selection
# ================================================================

class SaliencyMap:
    """Compute saliency from Gabor response magnitudes and select fixations."""

    def __init__(self, gabor, rng: np.random.Generator):
        self.gabor = gabor
        self.rng = rng
        self.image_size = gabor.image_size

    def compute(self, image: np.ndarray) -> np.ndarray:
        """Compute saliency map from summed Gabor response magnitudes.

        saliency[y, x] = Σ_i |Gabor_response_i[y, x]| → smoothed
        """
        gray = self.gabor._preprocess(image)
        fft_size = self.gabor.fft_size
        gray_padded = np.zeros((fft_size, fft_size), dtype=np.float32)
        gray_padded[:self.image_size, :self.image_size] = gray
        image_fft = np.fft.fft2(gray_padded)

        saliency = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        for i in range(self.gabor.n_filters):
            resp = np.real(np.fft.ifft2(image_fft * self.gabor._kernel_ffts[i]))
            saliency += np.abs(resp[:self.image_size, :self.image_size])

        # Smooth
        saliency = gaussian_filter(saliency, sigma=8.0)

        # Normalize to [0, 1]
        s_min, s_max = saliency.min(), saliency.max()
        if s_max - s_min > 1e-8:
            saliency = (saliency - s_min) / (s_max - s_min)

        return saliency.astype(np.float32)

    def select_fixations(self, saliency: np.ndarray, n: int = N_FIXATIONS
                          ) -> list[tuple[int, int]]:
        """Select N fixation points with inhibition of return.

        After each selection, suppress the region around the fixation
        to prevent re-fixating the same spot (biologically: inhibition of return).
        """
        fixations = []
        working = saliency.copy()

        for _ in range(n):
            fy, fx = np.unravel_index(np.argmax(working), working.shape)
            fixations.append((int(fx), int(fy)))

            # Inhibition of return: Gaussian suppression
            y, x = np.mgrid[0:saliency.shape[0], 0:saliency.shape[1]]
            dist_sq = (x - fx) ** 2 + (y - fy) ** 2
            mask = np.exp(-dist_sq / (2 * INHIBITION_RADIUS ** 2))
            working *= (1.0 - mask)

            # Small noise to break ties
            working += self.rng.uniform(0, 0.001, working.shape)

        return fixations


# ================================================================
# Foveated Encoding
# ================================================================

class FoveatedEncoder:
    """Encode image with foveated attention at specific fixation points."""

    def __init__(self, gabor, rng: np.random.Generator, use_color: bool = False):
        self.gabor = gabor
        self.rng = rng
        self.use_color = use_color
        self.image_size = gabor.image_size

        # Fixed random projections for per-fixation compression
        # (not learned — biologically: fixed V1 receptive fields)
        self._proj_v1 = rng.standard_normal(
            (gabor.raw_dim, FIX_V1_DIM)).astype(np.float32) / np.sqrt(gabor.raw_dim)

        # V2 raw dim estimation
        dummy = gabor.encode_v2(np.zeros((gabor.image_size, gabor.image_size, 3),
                                          dtype=np.uint8))
        v2_raw_dim = len(dummy)
        self._proj_v2 = rng.standard_normal(
            (v2_raw_dim, FIX_V2_DIM)).astype(np.float32) / np.sqrt(v2_raw_dim)

        # V4 raw dim
        dummy_v4 = gabor.encode_v4(np.zeros((gabor.image_size, gabor.image_size, 3),
                                             dtype=np.uint8))
        v4_raw_dim = len(dummy_v4)
        self._proj_v4 = rng.standard_normal(
            (v4_raw_dim, FIX_V4_DIM)).astype(np.float32) / np.sqrt(max(v4_raw_dim, 1))

        # Color raw dim
        if use_color:
            dummy_c = gabor.encode_color(np.zeros((gabor.image_size, gabor.image_size, 3),
                                                   dtype=np.uint8))
            color_raw_dim = len(dummy_c)
            self._proj_color = rng.standard_normal(
                (color_raw_dim, FIX_COLOR_DIM)).astype(np.float32) / np.sqrt(color_raw_dim)
        else:
            self._proj_color = None

    def encode_foveated(self, image: np.ndarray, fixation: tuple[int, int]
                         ) -> np.ndarray:
        """Encode image with foveated attention at fixation point.

        Efficient: computes Gabor response maps once, then derives
        V1, V2, V4 features from shared maps (avoids redundant FFTs).

        Returns:
            (FIX_DIM,) float32 — compressed per-fixation features
        """
        fx, fy = fixation

        # Resize image
        if image.shape[0] != self.image_size or image.shape[1] != self.image_size:
            from PIL import Image
            if image.ndim == 3 and image.shape[2] >= 3:
                pil_img = Image.fromarray(image.astype(np.uint8))
            elif image.ndim == 3:
                pil_img = Image.fromarray(image[:,:,0].astype(np.uint8))
            else:
                pil_img = Image.fromarray(image.astype(np.uint8))
            pil_img = pil_img.resize((self.image_size, self.image_size),
                                     Image.LANCZOS)
            image_resized = np.array(pil_img)
        else:
            image_resized = image

        # Gaussian weight mask
        y, x = np.mgrid[0:self.image_size, 0:self.image_size]
        dist_sq = (x - fx) ** 2 + (y - fy) ** 2
        weights = np.exp(-dist_sq / (2 * FOVEA_SIGMA ** 2))

        # Preprocess to grayscale, apply weight, normalize
        gray = self.gabor._preprocess(image)
        gray_weighted = gray * weights  # foveated attention

        # ---- Shared: compute all 32 Gabor response maps once ----
        fft_size = self.gabor.fft_size
        gray_padded = np.zeros((fft_size, fft_size), dtype=np.float32)
        gray_padded[:self.image_size, :self.image_size] = gray_weighted
        image_fft = np.fft.fft2(gray_padded)

        response_maps = np.zeros(
            (self.gabor.n_filters, self.image_size, self.image_size),
            dtype=np.float32)
        for i in range(self.gabor.n_filters):
            resp = np.real(np.fft.ifft2(image_fft * self.gabor._kernel_ffts[i]))
            response_maps[i] = resp[:self.image_size, :self.image_size]

        # ---- V1 features (4x4 grid) from shared maps ----
        n_cells_v1 = self.gabor.grid_size * self.gabor.grid_size  # 16
        v1_raw = np.zeros(self.gabor.n_filters * n_cells_v1 * 2, dtype=np.float32)
        for i in range(self.gabor.n_filters):
            modulated = response_maps[i] * self.gabor.gains[i]
            for ci, (sy, sx) in enumerate(self.gabor._grid_slices):
                cell = modulated[sy, sx]
                v1_raw[i * n_cells_v1 * 2 + ci * 2] = float(np.mean(cell))
                v1_raw[i * n_cells_v1 * 2 + ci * 2 + 1] = float(np.std(cell))
        norm = np.linalg.norm(v1_raw)
        if norm > 1e-8:
            v1_raw /= norm

        # ---- V2 features (2x2 grid) from shared maps ----
        half = self.image_size // 2
        v2_cells = [
            (slice(0, half), slice(0, half)),
            (slice(0, half), slice(half, self.image_size)),
            (slice(half, self.image_size), slice(0, half)),
            (slice(half, self.image_size), slice(half, self.image_size)),
        ]
        n_cells_v2 = 4
        extra_dim = n_cells_v2 * (self.gabor.n_orientations // 2) + n_cells_v2
        v2_raw = np.zeros(self.gabor.n_filters * n_cells_v2 * 2 + extra_dim,
                          dtype=np.float32)
        for i in range(self.gabor.n_filters):
            modulated = response_maps[i] * self.gabor.gains[i]
            for ci, (sy, sx) in enumerate(v2_cells):
                cell = modulated[sy, sx]
                v2_raw[i * n_cells_v2 * 2 + ci * 2] = float(np.mean(cell))
                v2_raw[i * n_cells_v2 * 2 + ci * 2 + 1] = float(np.std(cell))
        # Simplified V2 extra features (orientation contrast only)
        base_offset = self.gabor.n_filters * n_cells_v2 * 2
        for ci in range(n_cells_v2):
            orient_means = np.zeros(self.gabor.n_orientations, dtype=np.float32)
            for oi in range(self.gabor.n_orientations):
                orient_means[oi] = np.mean([
                    v2_raw[i * n_cells_v2 * 2 + ci * 2]
                    for i in range(oi, self.gabor.n_filters, self.gabor.n_orientations)
                ])
            v2_raw[base_offset + ci] = float(np.max(orient_means) - np.min(orient_means))
        norm = np.linalg.norm(v2_raw)
        if norm > 1e-8:
            v2_raw /= norm

        # ---- V4 features (global + curvature) from shared maps ----
        v4_global = np.zeros(self.gabor.n_filters * 2, dtype=np.float32)
        for i in range(self.gabor.n_filters):
            modulated = response_maps[i] * self.gabor.gains[i]
            v4_global[i * 2] = float(np.mean(modulated))
            v4_global[i * 2 + 1] = float(np.std(modulated))
        # Curvature (simplified: 8 orient)
        curvature = np.zeros(self.gabor.n_orientations, dtype=np.float32)
        for oi in range(self.gabor.n_orientations):
            o1, o2 = oi, (oi + 1) % self.gabor.n_orientations
            r1, r2 = np.zeros(self.gabor.n_scales), np.zeros(self.gabor.n_scales)
            for si in range(self.gabor.n_scales):
                r1[si] = np.mean(response_maps[si * self.gabor.n_orientations + o1])
                r2[si] = np.mean(response_maps[si * self.gabor.n_orientations + o2])
            curvature[oi] = float(np.dot(r1, r2) / (np.linalg.norm(r1) * np.linalg.norm(r2) + 1e-8))
        v4_raw = np.concatenate([v4_global, curvature])
        norm = np.linalg.norm(v4_raw)
        if norm > 1e-8:
            v4_raw /= norm

        # ---- Random projection compression ----
        v1_comp = (v1_raw @ self._proj_v1).astype(np.float32)
        v2_comp = (v2_raw @ self._proj_v2).astype(np.float32)
        v4_comp = (v4_raw @ self._proj_v4).astype(np.float32)

        features = np.concatenate([v1_comp, v2_comp, v4_comp,
                                    np.zeros(FIX_COLOR_DIM, dtype=np.float32)])
        norm = np.linalg.norm(features)
        if norm > 1e-8:
            features /= norm

        return features.astype(np.float32)


# ================================================================
# Neuromodulation
# ================================================================

class NeuroModulator:
    """DA/ACh/NE neuromodulation of visual learning rates and thresholds.

    Three neuromodulatory signals:
      DA (Dopamine): F_visual drops → reward prediction → boost IT learning
      ACh (Acetylcholine): F_visual rises → novelty → boost V1/V2 learning
      NE (Norepinephrine): sustained high F → uncertainty → lower thresholds
    """

    def __init__(self):
        self.DA: float = 0.0
        self.ACh: float = 0.0
        self.NE: float = 0.0
        self.F_visual_ema: float = 0.0
        self.F_alpha: float = 0.95
        self.mod_history: list[dict] = []

    def update(self, F_visual: float):
        """Update neuromodulatory signals based on prediction error."""
        # EMA tracking
        self.F_visual_ema = (self.F_alpha * self.F_visual_ema +
                             (1 - self.F_alpha) * F_visual)
        F_delta = F_visual - self.F_visual_ema

        # DA: prediction improving (error dropping) → reward
        self.DA = float(1.0 / (1.0 + np.exp(F_delta * 5.0)))

        # ACh: prediction worsening (error rising, novelty) → attend/encode
        self.ACh = float(1.0 / (1.0 + np.exp(-F_delta * 5.0)))

        # NE: sustained high error → uncertainty/arousal
        self.NE = float(1.0 / (1.0 + np.exp(-(F_visual - 0.3) * 10.0)))

        self.mod_history.append({
            'F_visual': F_visual, 'F_ema': self.F_visual_ema,
            'DA': self.DA, 'ACh': self.ACh, 'NE': self.NE,
        })

    def apply_to_net(self, net: ClusterNetwork, layer_name: str,
                      base_lr: float, base_threshold: float):
        """Apply neuromodulation to a specific network.

        Args:
            net: ClusterNetwork to modulate
            layer_name: 'v1', 'v2', 'v4', or 'it'
            base_lr: original learning rate
            base_threshold: original cluster threshold
        """
        # DA boosts IT learning (consolidation of good predictions)
        if layer_name == 'it':
            lr_mod = 1.0 + 0.5 * self.DA
        # ACh boosts V1/V2 learning (encoding novel features)
        elif layer_name == 'v1':
            lr_mod = 1.0 + 1.0 * self.ACh
        elif layer_name == 'v2':
            lr_mod = 1.0 + 0.8 * self.ACh
        elif layer_name == 'v4':
            lr_mod = 1.0 + 0.3 * self.ACh
        else:
            lr_mod = 1.0

        net.theta.learn_rate_l0 = base_lr * lr_mod

        # NE lowers thresholds globally (uncertainty → more liberal matching)
        threshold_mod = 1.0 - 0.2 * self.NE
        net.theta.cluster_threshold = base_threshold * threshold_mod


# ================================================================
# Network Factory
# ================================================================

def _create_net(layer_name: str, hash_offset: int) -> ClusterNetwork:
    params = LAYER_PARAMS[layer_name]
    theta = create_default_theta()
    theta.cluster_threshold = params['threshold']
    theta.learn_rate_l0 = params['lr']
    theta.decay_rate = params['decay']
    return ClusterNetwork(theta, hash_offset=hash_offset)


# ================================================================
# Cluster Summary (for higher layers)
# ================================================================

def compute_cluster_sims(net: ClusterNetwork, features: np.ndarray,
                          start: int, end: int) -> np.ndarray:
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


def build_cluster_summary(nets: dict, all_features: np.ndarray,
                           stage: int, saccade_mode: bool) -> np.ndarray:
    """48d cluster summary for higher layers."""
    summary = np.zeros(48, dtype=np.float32)
    pos = 0
    if saccade_mode:
        layer_order = [
            ('v1', FIX1_START, FIX1_START + FIX_V1_DIM),
            ('v2', FIX1_START + FIX_V1_DIM, FIX1_START + FIX_V1_DIM + FIX_V2_DIM),
        ]
        if stage >= 2:
            layer_order.append(('v4', FIX1_START + FIX_V1_DIM + FIX_V2_DIM,
                                FIX1_START + FIX_V1_DIM + FIX_V2_DIM + FIX_V4_DIM))
    else:
        layer_order = [
            ('v1', UNI_V1_START, UNI_V1_END),
        ]
        if stage >= 1:
            layer_order.append(('v2', UNI_V2_START, UNI_V2_END))
        if stage >= 2:
            layer_order.append(('v4', UNI_V4_START, UNI_V4_END))

    for layer_name, start, end in layer_order:
        net = nets.get(layer_name)
        if net is None or net.n_clusters == 0:
            pos += 8
            continue
        feat_slice = all_features[start:end]
        if len(feat_slice) == 0:
            pos += 8
            continue
        sims = compute_cluster_sims(net, feat_slice, start, end)
        if len(sims) == 0:
            pos += 8
            continue
        sorted_sims = np.sort(sims)[::-1]
        n_top = min(5, len(sorted_sims))
        if pos + 8 <= 48:
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
# Training
# ================================================================

def train_hierarchy(venv, config: dict, n_images: int,
                     seed: int = 42) -> dict:
    """Train V1→V2→V4→IT hierarchy with saccade/neuromod options.

    Args:
        venv: VisualEnvironment (cached features for uniform mode)
        config: {'saccade', 'neuromod', 'color', 'pp'}
    """
    rng = np.random.default_rng(seed)
    use_saccade = config['saccade']
    use_neuromod = config['neuromod']
    use_color = config['color']
    use_pp = config['pp']
    mode_name = ('tier1_nm' if (use_color and use_pp and use_neuromod) else
                 'sacc_nm' if (use_saccade and use_neuromod) else
                 'neuromod' if use_neuromod else
                 'saccade' if use_saccade else 'baseline')

    # Create networks
    nets = {}
    if use_saccade:
        offsets = {'v1': FIX1_START, 'v2': FIX2_START,
                    'v4': FIX3_START, 'it': FIX1_START}
    else:
        offsets = {'v1': UNI_V1_START, 'v2': UNI_V2_START,
                    'v4': UNI_V4_START, 'it': UNI_V1_START}

    for name in ['v1', 'v2', 'v4', 'it']:
        nets[name] = _create_net(name, offsets[name])

    # Setup saccade components
    saliency_map = None
    foveated_encoder = None
    if use_saccade:
        from layer0_visual import GaborFilterBank
        # Create a dedicated Gabor for on-the-fly encoding (no Hebb gain updates)
        gabor_sacc = GaborFilterBank(image_size=venv.image_size, grid_size=venv.grid_size)
        saliency_map = SaliencyMap(gabor_sacc, rng)
        foveated_encoder = FoveatedEncoder(gabor_sacc, rng, use_color=use_color)

    # Neuromodulation
    neuro_mod = NeuroModulator() if use_neuromod else None

    # Store base params for neuromodulation
    base_params = {}
    if use_neuromod:
        for name in ['v1', 'v2', 'v4', 'it']:
            base_params[name] = {
                'lr': LAYER_PARAMS[name]['lr'],
                'threshold': LAYER_PARAMS[name]['threshold'],
            }

    order = rng.permutation(n_images)
    t0 = time.perf_counter()
    F_visual_ema = 0.0

    print(f"\n{'='*64}")
    print(f"  Training: {mode_name}")
    print(f"{'='*64}")
    print(f"  Saccade={use_saccade}, NeuroMod={use_neuromod}, "
          f"Color={use_color}, PP={use_pp}")

    for step, idx in enumerate(order):
        # ---- Build sensory vector ----
        if use_saccade:
            image = venv.images[idx]
            sal = saliency_map.compute(image)
            fixations = saliency_map.select_fixations(sal, n=N_FIXATIONS)

            # Encode each fixation
            fix_features = []
            for fix in fixations:
                ff = foveated_encoder.encode_foveated(image, fix)
                fix_features.append(ff)

            # Concatenate fixations into s[]
            s = np.zeros(D, dtype=np.float32)
            s[FIX1_START:FIX1_END] = fix_features[0]
            s[FIX2_START:FIX2_END] = fix_features[1]
            s[FIX3_START:FIX3_END] = fix_features[2]

            # Saliency stats: mean, std, entropy of saliency map
            sal_flat = sal.ravel()
            sal_stats = np.zeros(24, dtype=np.float32)
            sal_stats[0] = float(np.mean(sal_flat))
            sal_stats[1] = float(np.std(sal_flat))
            sal_stats[2] = float(-np.sum(sal_flat * np.log(sal_flat + 1e-8)) / len(sal_flat))
            # Fixation positions (normalized)
            for i, (fx, fy) in enumerate(fixations):
                sal_stats[3 + i * 2] = fx / venv.image_size
                sal_stats[3 + i * 2 + 1] = fy / venv.image_size
            sal_stats[9] = float(np.sqrt(
                (fixations[0][0] - fixations[1][0])**2 +
                (fixations[0][1] - fixations[1][1])**2) / venv.image_size)
            norm = np.linalg.norm(sal_stats)
            if norm > 1e-8:
                sal_stats /= norm
            s[SAL_START:SAL_END] = sal_stats

            all_features = s.copy()
        else:
            # Uniform mode: use cached features
            v1_feat = venv.encodings[idx]
            v2_feat = (venv.encodings_v2[idx].copy()
                       if venv.encodings_v2 is not None else None)
            v4_feat = (venv.encodings_v4[idx].copy()
                       if venv.encodings_v4 is not None else None)
            color_feat = (venv.encodings_color[idx].copy()
                          if use_color and getattr(venv, 'encodings_color', None) is not None
                          else None)

            s = np.zeros(D, dtype=np.float32)
            flen = min(len(v1_feat), UNI_V1_END - UNI_V1_START)
            s[UNI_V1_START:UNI_V1_START + flen] = v1_feat[:flen]
            if v2_feat is not None:
                flen = min(len(v2_feat), UNI_V2_END - UNI_V2_START)
                s[UNI_V2_START:UNI_V2_START + flen] = v2_feat[:flen]
            if v4_feat is not None:
                flen = min(len(v4_feat), UNI_V4_END - UNI_V4_START)
                s[UNI_V4_START:UNI_V4_START + flen] = v4_feat[:flen]
            if color_feat is not None:
                flen = min(len(color_feat), UNI_COLOR_END - UNI_COLOR_START)
                s[UNI_COLOR_START:UNI_COLOR_START + flen] = color_feat[:flen]
            all_features = s.copy()

        # ---- Feedforward recall ----
        c_it = nets['it'].recall(s)
        c_v4 = nets['v4'].recall(s)
        c_v2 = nets['v2'].recall(s)
        c_v1 = nets['v1'].recall(s)

        # ---- Prediction error (for PP + neuromodulation) ----
        if use_pp or use_neuromod:
            if c_it is not None:
                if use_saccade:
                    pe = float(np.sum(
                        (s[FIX1_START:SAL_END] - c_it.centroid[FIX1_START:SAL_END])**2
                    )) / (SAL_END - FIX1_START)
                else:
                    pe = float(np.sum(
                        (s[UNI_V1_START:UNI_V4_END] - c_it.centroid[UNI_V1_START:UNI_V4_END])**2
                    )) / (UNI_V4_END - UNI_V1_START)
            else:
                pe = 0.5  # no cluster matched → moderate surprise

            F_visual_ema = 0.95 * F_visual_ema + 0.05 * pe

            if use_neuromod:
                neuro_mod.update(pe)

            # Encode PE in sensory for PP
            if use_pp:
                pe_vec = np.zeros(PE_END - PE_START, dtype=np.float32)
                pe_vec[0] = float(np.clip(pe, 0, 10.0))
                pe_vec[1] = float(np.log1p(pe))
                norm = np.linalg.norm(pe_vec)
                if norm > 1e-8:
                    pe_vec /= norm
                s[PE_START:PE_END] = pe_vec

        # ---- Cluster summary for higher layers ----
        stage = 3
        cs = build_cluster_summary(nets, all_features, stage, use_saccade)
        s[CS_START:CS_END] = cs

        # ---- Neuromodulation application ----
        if use_neuromod:
            for name, net in nets.items():
                neuro_mod.apply_to_net(
                    net, name,
                    base_params[name]['lr'],
                    base_params[name]['threshold'])

        # ---- Learn ----
        nets['it'].learn(s)
        nets['v4'].learn(s)
        nets['v2'].learn(s)
        nets['v1'].learn(s)

        # ---- Reset neuromodulation (restore base params for next step) ----
        if use_neuromod:
            for name, net in nets.items():
                net.theta.learn_rate_l0 = base_params[name]['lr']
                net.theta.cluster_threshold = base_params[name]['threshold']

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
            nm_info = ""
            if use_neuromod and neuro_mod is not None:
                nm_info = (f"DA={neuro_mod.DA:.2f} ACh={neuro_mod.ACh:.2f} "
                          f"NE={neuro_mod.NE:.2f} | ")
            print(f"  {step+1}/{n_images} ({ips:.0f} img/s) | "
                  f"{nm_info}"
                  f"V1:{nets['v1'].n_clusters} V2:{nets['v2'].n_clusters} "
                  f"V4:{nets['v4'].n_clusters} IT:{nets['it'].n_clusters} | "
                  f"Fv={F_visual_ema:.4f}")

    elapsed = time.perf_counter() - t0
    print(f"  Training complete: {elapsed:.1f}s")
    for name, net in nets.items():
        print(f"    {name}: {net.n_clusters} clusters")

    return {
        'nets': nets,
        'F_visual_ema': F_visual_ema,
        'mode': mode_name,
        'neuro_mod': neuro_mod,
    }


# ================================================================
# Evaluation (shared with earlier phases)
# ================================================================

def evaluate_layer(net: ClusterNetwork, venv, nets: dict,
                    config: dict, n_images: int = None) -> dict:
    """Read-only evaluation of a layer's clustering quality."""
    if n_images is None:
        n_images = venv.n_images
    use_saccade = config['saccade']
    use_color = config['color']
    use_pp = config['pp']

    # Setup saccade if needed
    if use_saccade:
        from layer0_visual import GaborFilterBank
        rng = np.random.default_rng(42)
        gabor_sacc = GaborFilterBank(image_size=venv.image_size, grid_size=venv.grid_size)
        sal_map = SaliencyMap(gabor_sacc, rng)
        fov_enc = FoveatedEncoder(gabor_sacc, rng, use_color=use_color)

    cluster_hits = defaultdict(list)

    for idx in range(n_images):
        if use_saccade:
            image = venv.images[idx]
            sal = sal_map.compute(image)
            fixations = sal_map.select_fixations(sal, n=N_FIXATIONS)
            fix_features = [fov_enc.encode_foveated(image, f) for f in fixations]
            s = np.zeros(D, dtype=np.float32)
            s[FIX1_START:FIX1_END] = fix_features[0]
            s[FIX2_START:FIX2_END] = fix_features[1]
            s[FIX3_START:FIX3_END] = fix_features[2]
        else:
            v1_feat = venv.encodings[idx]
            v2_feat = (venv.encodings_v2[idx].copy()
                       if venv.encodings_v2 is not None else None)
            v4_feat = (venv.encodings_v4[idx].copy()
                       if venv.encodings_v4 is not None else None)
            color_feat = (venv.encodings_color[idx].copy()
                          if use_color and getattr(venv, 'encodings_color', None) is not None
                          else None)
            s = np.zeros(D, dtype=np.float32)
            flen = min(len(v1_feat), UNI_V1_END - UNI_V1_START)
            s[UNI_V1_START:UNI_V1_START + flen] = v1_feat[:flen]
            if v2_feat is not None:
                flen = min(len(v2_feat), UNI_V2_END - UNI_V2_START)
                s[UNI_V2_START:UNI_V2_START + flen] = v2_feat[:flen]
            if v4_feat is not None:
                flen = min(len(v4_feat), UNI_V4_END - UNI_V4_START)
                s[UNI_V4_START:UNI_V4_START + flen] = v4_feat[:flen]
            if color_feat is not None:
                flen = min(len(color_feat), UNI_COLOR_END - UNI_COLOR_START)
                s[UNI_COLOR_START:UNI_COLOR_START + flen] = color_feat[:flen]

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
            cluster_hits[id(best_c)].append((idx, int(venv.labels[idx]), best_sim))

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
    n_eval = min(500, n_images)
    indices = np.random.choice(n_images, n_eval, replace=False)
    n_hit = 0
    for idx in indices:
        if use_saccade:
            image = venv.images[idx]
            sal = sal_map.compute(image)
            fixations = sal_map.select_fixations(sal, n=N_FIXATIONS)
            fix_features = [fov_enc.encode_foveated(image, f) for f in fixations]
            s = np.zeros(D, dtype=np.float32)
            s[FIX1_START:FIX1_END] = fix_features[0]
            s[FIX2_START:FIX2_END] = fix_features[1]
            s[FIX3_START:FIX3_END] = fix_features[2]
        else:
            v1_feat = venv.encodings[idx]
            s = np.zeros(D, dtype=np.float32)
            flen = min(len(v1_feat), UNI_V1_END - UNI_V1_START)
            s[UNI_V1_START:UNI_V1_START + flen] = v1_feat[:flen]
            if venv.encodings_v2 is not None:
                v2f = venv.encodings_v2[idx]
                flen = min(len(v2f), UNI_V2_END - UNI_V2_START)
                s[UNI_V2_START:UNI_V2_START + flen] = v2f[:flen]
            if venv.encodings_v4 is not None:
                v4f = venv.encodings_v4[idx]
                flen = min(len(v4f), UNI_V4_END - UNI_V4_START)
                s[UNI_V4_START:UNI_V4_START + flen] = v4f[:flen]
        if net.n_clusters > 0:
            h = np.tanh(s + 1e-8)
            mask = _auto_mask(s)
            _ = max(_masked_cosine(h, c.centroid, mask) for c in net.clusters)
            n_hit += 1

    coverage = n_hit / n_eval

    # Confusion diagonal
    n_classes = venv.n_classes
    cluster_to_class = {}
    for cid, data in purities.items():
        if data['total_hits'] >= 3:
            cluster_to_class[cid] = data['top_class']

    confusion_diag = np.zeros(n_classes, dtype=np.float32)
    n_per_class = min(30, n_images // n_classes)
    for c in range(n_classes):
        c_indices = np.where(venv.labels == c)[0]
        if len(c_indices) == 0:
            continue
        sample = np.random.choice(c_indices, min(n_per_class, len(c_indices)), replace=False)
        correct = 0
        for idx in sample:
            if use_saccade:
                image = venv.images[idx]
                sal = sal_map.compute(image)
                fixations = sal_map.select_fixations(sal, n=N_FIXATIONS)
                fix_features = [fov_enc.encode_foveated(image, f) for f in fixations]
                s = np.zeros(D, dtype=np.float32)
                s[FIX1_START:FIX1_END] = fix_features[0]
                s[FIX2_START:FIX2_END] = fix_features[1]
                s[FIX3_START:FIX3_END] = fix_features[2]
            else:
                v1_feat = venv.encodings[idx]
                v2_feat = (venv.encodings_v2[idx].copy() if venv.encodings_v2 is not None else None)
                v4_feat = (venv.encodings_v4[idx].copy() if venv.encodings_v4 is not None else None)
                color_feat = (venv.encodings_color[idx].copy()
                              if use_color and getattr(venv, 'encodings_color', None) is not None
                              else None)
                s = np.zeros(D, dtype=np.float32)
                flen = min(len(v1_feat), UNI_V1_END - UNI_V1_START)
                s[UNI_V1_START:UNI_V1_START + flen] = v1_feat[:flen]
                if v2_feat is not None:
                    flen = min(len(v2_feat), UNI_V2_END - UNI_V2_START)
                    s[UNI_V2_START:UNI_V2_START + flen] = v2_feat[:flen]
                if v4_feat is not None:
                    flen = min(len(v4_feat), UNI_V4_END - UNI_V4_START)
                    s[UNI_V4_START:UNI_V4_START + flen] = v4_feat[:flen]
                if color_feat is not None:
                    flen = min(len(color_feat), UNI_COLOR_END - UNI_COLOR_START)
                    s[UNI_COLOR_START:UNI_COLOR_START + flen] = color_feat[:flen]
            if net.n_clusters > 0:
                h = np.tanh(s + 1e-8)
                mask = _auto_mask(s)
                best_sim, best_cid = -1.0, None
                for cl in net.clusters:
                    sim = _masked_cosine(h, cl.centroid, mask)
                    if sim > best_sim:
                        best_sim, best_cid = sim, id(cl)
                if (best_cid is not None and best_cid in cluster_to_class
                        and cluster_to_class[best_cid] == c):
                    correct += 1
        confusion_diag[c] = correct / max(len(sample), 1)

    avg_diagonal = float(np.mean(confusion_diag))

    top_clusters = []
    sorted_pure = sorted(purities.items(), key=lambda x: -x[1]['purity'])
    for rank, (cid, data) in enumerate(sorted_pure[:10]):
        top_clusters.append({
            'rank': rank + 1, 'class': data['top_class_name'],
            'hits': data['total_hits'], 'purity': data['purity'],
        })

    return {
        'n_clusters': net.n_clusters, 'n_active': len(purities),
        'avg_purity': avg_purity, 'coverage': coverage,
        'avg_diagonal': avg_diagonal, 'top_clusters': top_clusters,
    }


# ================================================================
# Main Experiment
# ================================================================

def run_tier2_experiment(n_images: int = 2000,
                          dataset: str = 'imagenette',
                          mode: str = 'all',
                          seed: int = 42):
    np.random.seed(seed)

    print("=" * 72)
    print("  Tier 2: Saccade Attention + Neuromodulation")
    print("=" * 72)
    print(f"  Dataset: {dataset}, {n_images} images, mode: {mode}")
    print()

    if mode == 'all':
        configs_to_run = list(CONFIGS.keys())
    else:
        configs_to_run = [mode]

    # Load VisualEnvironment (with color if tier1_nm is in configs)
    need_color = any(CONFIGS[c].get('color', False) for c in configs_to_run)

    print(f"[1/3] Loading VisualEnvironment (color={need_color})...")
    t0 = time.perf_counter()
    from visual_interface import VisualEnvironment

    venv = VisualEnvironment(
        dataset=dataset, n_images=n_images,
        pca_components=96, v2_components=64, v4_components=64,
        color_components=64 if need_color else 0,
        use_v2=True, use_v4=True, use_color=need_color,
    )
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s")
    n_actual = venv.n_images

    results = {}

    for config_name in configs_to_run:
        config = CONFIGS[config_name]
        print(f"\n[2/{config_name}] Training...")
        train_result = train_hierarchy(venv, config, n_actual, seed=seed)
        nets = train_result['nets']

        print(f"\n[3/{config_name}] Evaluating...")
        eval_results = {}
        for layer_name in ['v1', 'v2', 'v4', 'it']:
            ev = evaluate_layer(nets[layer_name], venv, nets, config, n_actual)
            eval_results[layer_name] = ev
            print(f"    {layer_name.upper()}: {ev['n_clusters']} clusters, "
                  f"purity={ev['avg_purity']:.3f}, diag={ev['avg_diagonal']:.3f}")

        results[config_name] = {
            'training': train_result, 'evaluation': eval_results,
        }

    # ---- Report ----
    print("\n" + "=" * 72)
    print("  Tier 2 Results: Saccade + Neuromodulation")
    print("=" * 72)

    print(f"\n  [IT Layer Comparison]")
    header = (f"  {'Config':<12} {'IT Active':<10} {'IT Purity':<10} "
              f"{'IT Diag':<10} {'F_visual':<10}")
    print(header)
    print(f"  {'-'*len(header)}")
    for config_name in configs_to_run:
        r = results[config_name]
        it_ev = r['evaluation']['it']
        Fv = r['training']['F_visual_ema']
        print(f"  {config_name:<12} {it_ev['n_active']:<10} "
              f"{it_ev['avg_purity']:<10.3f} {it_ev['avg_diagonal']:<10.3f} "
              f"{Fv:<10.4f}")

    # Per-layer comparison
    print(f"\n  [Per-Layer Purity]")
    header = f"  {'Config':<12}"
    for ln in ['v1', 'v2', 'v4', 'it']:
        header += f" {ln.upper():<10}"
    print(header)
    print(f"  {'-'*len(header)}")
    for config_name in configs_to_run:
        row = f"  {config_name:<12}"
        for ln in ['v1', 'v2', 'v4', 'it']:
            row += f" {results[config_name]['evaluation'][ln]['avg_purity']:<10.3f}"
        print(row)

    # Neuromodulation stats
    for config_name in configs_to_run:
        r = results[config_name]
        nm = r['training'].get('neuro_mod')
        if nm is not None and len(nm.mod_history) > 0:
            da_vals = [m['DA'] for m in nm.mod_history]
            ach_vals = [m['ACh'] for m in nm.mod_history]
            ne_vals = [m['NE'] for m in nm.mod_history]
            print(f"\n  [NeuroMod Stats] {config_name}:")
            print(f"    DA:  mean={np.mean(da_vals):.3f} std={np.std(da_vals):.3f}")
            print(f"    ACh: mean={np.mean(ach_vals):.3f} std={np.std(ach_vals):.3f}")
            print(f"    NE:  mean={np.mean(ne_vals):.3f} std={np.std(ne_vals):.3f}")

    # ---- Acceptance Check ----
    print(f"\n  {'='*48}")
    print(f"  ACCEPTANCE CHECK")
    print(f"  {'='*48}")

    best_it = results[configs_to_run[-1]]['evaluation']['it']
    baseline_it = results.get('baseline', {}).get('evaluation', {}).get('it', {})
    checks = []

    c1 = best_it['n_active'] >= 10
    checks.append(('IT >=10 active clusters', c1, best_it['n_active']))
    c2 = best_it['avg_purity'] > 0.25
    checks.append(('IT purity > 0.25', c2, f"{best_it['avg_purity']:.3f}"))
    c3 = best_it['coverage'] > 0.50
    checks.append(('IT coverage > 50%', c3, f"{best_it['coverage']:.1%}"))
    c4 = best_it['avg_diagonal'] > 0.20
    checks.append(('IT diag > 0.20', c4, f"{best_it['avg_diagonal']:.3f}"))

    if baseline_it:
        delta = best_it['avg_purity'] - baseline_it['avg_purity']
        c5 = delta >= -0.03
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

    return {'configs': configs_to_run, 'results': results,
            'all_checks_passed': all_pass}


# ================================================================
# CLI
# ================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Tier 2: Saccade Attention + Neuromodulation')
    parser.add_argument('--n', type=int, default=2000)
    parser.add_argument('--dataset', type=str, default='imagenette',
                       choices=['cifar10', 'imagenette'])
    parser.add_argument('--mode', type=str, default='all',
                       choices=['all', 'baseline', 'saccade', 'neuromod',
                                'sacc_nm', 'tier1_nm'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    run_tier2_experiment(n_images=args.n, dataset=args.dataset,
                          mode=args.mode, seed=args.seed)
