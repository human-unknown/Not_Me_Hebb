"""
layer2_inference.py — 向后兼容 re-export shim (v4.1)
实际代码位置: cerebrum/frontal_lobe/prefrontal.py
"""
from cerebrum.frontal_lobe.prefrontal import (
    compute_G, select_action, predict_next_state,
    update_social_beliefs,
)
