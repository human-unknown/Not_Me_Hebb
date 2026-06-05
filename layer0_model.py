"""
layer0_model.py — 向后兼容 re-export shim (v4.1)
实际代码位置: cerebrum/limbic_system/hippocampus.py
"""
from cerebrum.limbic_system.hippocampus import (
    predict_sensations, ClusterNetwork, sleep_cycle,
    _masked_cosine, _auto_mask,
)
