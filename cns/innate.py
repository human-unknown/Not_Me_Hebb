"""
innate.py — v6.0 先天反射/偏好定义

Zero-pretraining mode Agent "factory configuration".
Contains NO language knowledge — only body homeostasis reflexes,
attention biases, and plasticity parameters.
These are the innate structures human infants are born with.

Corresponding brain areas:
  - hypothalamus: body homeostasis reflexes (hunger/thirst/pain)
  - locus_coeruleus: novelty preference (NE baseline)
  - FPN: innate attention biases (sudden sounds/motion)
  - hippocampus: Hebb plasticity parameters (learn rate/forgetting rate)
"""

import numpy as np


# ============================================================
# Pure Mode Theta Overrides (adapted for zero-pretraining)
# ============================================================

PURE_MODE_THETA_OVERRIDES = {
    'cluster_threshold': 0.55,   # Lower matching threshold (fewer initial clusters)
    'learn_rate_l0': 0.12,       # Higher initial learn rate (infant critical period)
    'decay_rate': 0.005,         # Lower decay (prevent early memories vanishing)
    'exploration_bonus': 0.3,    # Higher exploration (encourage active learning)
    'critical_window': 5000,     # Extended critical period (language learning)
    'w_social': 2.0,             # Stronger social drive (infant attachment)
    'temperature': 1.5,          # Higher temperature (more random exploration)
    # v6.1: 发育优化 — 纯净模式从婴儿期开始
    'glun2b_ratio': 0.95,        # Start with very high GluN2B (max plasticity)
    'stdp_lr': 0.03,             # Higher STDP learning (rapid temporal association)
    'stdp_weight': 0.4,          # Higher STDP influence (learn causal direction)
    'pnn_formation_rate': 0.0005,  # Slower PNN formation (infant = highly plastic)
    'candidate_max': 80,         # More silent synapses (rapid vocabulary growth)
    'protection_decay': 0.99,    # Faster protection decay (more turnover)
}

# ============================================================
# Body Homeostasis Innate Setpoints (text dialogue mode, M=9)
# ============================================================

INNATE_SETPOINTS = np.array([
    0.65,   # b[0] social need — innate desire for human interaction
    0.70,   # b[1] energy/safety — needs adequate energy
    0.05,   # b[2] stress/fatigue — initially no stress
    0.20,   # b[3] novelty seeking — high innate curiosity
    0.40,   # b[4] focus/alertness — moderate alertness
    0.30,   # b[5] visual stimulation — initially low
    0.30,   # b[6] auditory stimulation — initially low
    0.50,   # b[7] cognitive load — moderate
    0.90,   # b[8] tissue integrity — healthy
], dtype=np.float32)

INNATE_DECAYS = np.array([
    -0.004,  # b[0] social need slowly drops (loneliness accumulates)
     0.000,  # b[1] energy natural balance
     0.002,  # b[2] stress slowly accumulates
     0.001,  # b[3] novelty slowly drifts
     0.000,  # b[4] focus natural balance
    -0.003,  # b[5] visual stimulation decays
    -0.003,  # b[6] auditory stimulation decays
     0.001,  # b[7] cognitive load slowly accumulates
     0.001,  # b[8] tissue integrity slowly heals
], dtype=np.float32)

# ============================================================
# Innate Attention Biases
# ============================================================

# Which sensory dimensions are innately more attended
# (infant: sudden sound > motion > text, but dialogue mode prioritizes text)
INNATE_ATTENTION_BIAS = np.zeros(516, dtype=np.float32)
INNATE_ATTENTION_BIAS[0:64] = 1.0     # text (primary input in dialogue)
INNATE_ATTENTION_BIAS[372:468] = 0.6  # auditory channel (sudden sounds)
INNATE_ATTENTION_BIAS[64:372] = 0.4   # visual channel (motion attracts)


def apply_innate_config(agent) -> None:
    """Apply innate configuration to an Agent instance.

    Call after Agent.__init__() to override defaults with pure mode params.
    Does NOT affect None/default checks — only sets values that should be overridden.

    Args:
        agent: Agent instance (already constructed via __init__)
    """
    # Theta overrides
    for key, value in PURE_MODE_THETA_OVERRIDES.items():
        if hasattr(agent.theta, key):
            setattr(agent.theta, key, value)

    # Body setpoints
    if agent.body is not None:
        M = len(agent.body.b)
        if M <= len(INNATE_SETPOINTS):
            agent.body.setpoints = INNATE_SETPOINTS[:M].copy()
            agent.body.decays = INNATE_DECAYS[:M].copy()
        else:
            agent.body.setpoints = INNATE_SETPOINTS.copy()
            agent.body.decays = INNATE_DECAYS.copy()

    # FPN innate attention bias
    if hasattr(agent, 'fpn') and agent.fpn is not None:
        agent.fpn.attention_template = (
            0.7 * agent.fpn.attention_template
            + 0.3 * INNATE_ATTENTION_BIAS
        )


def get_innate_reflexes() -> dict:
    """Return innate reflex metadata (for diagnostics/debugging).

    These are NOT tunable parameters — they are the Agent's
    hard-coded properties at "birth".
    """
    return {
        'pure_mode': True,
        'theta_overrides': PURE_MODE_THETA_OVERRIDES,
        'M_body': 9,
        'attention_bias_summary': {
            'text': float(INNATE_ATTENTION_BIAS[0:64].mean()),
            'vision': float(INNATE_ATTENTION_BIAS[64:372].mean()),
            'audio': float(INNATE_ATTENTION_BIAS[372:468].mean()),
        },
    }
