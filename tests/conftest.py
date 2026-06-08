"""
conftest.py — shared test fixtures and helpers for NotMe tests (v6.6).

Provides common test utilities previously duplicated across test files:
  - make_theta(**overrides) → Theta
  - make_deterministic_s(seed, text_signal) → np.ndarray
  - make_random_s(text_bias) → np.ndarray
"""

import sys
import os
import numpy as np

# Ensure project root is on path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from cns.data_types import D, Theta


def make_theta(**overrides) -> Theta:
    """Create a Theta instance with optional overrides."""
    t = Theta()
    for k, v in overrides.items():
        setattr(t, k, v)
    return t


def make_deterministic_s(seed: int, text_signal: float = 1.0) -> np.ndarray:
    """Generate a deterministic sensory vector with given seed."""
    rng = np.random.default_rng(seed)
    s = np.zeros(D, dtype=np.float32)
    s[:16] = rng.normal(0, text_signal, 16).astype(np.float32)
    s[16:64] = rng.normal(0, 0.1, 48).astype(np.float32)
    s[64:72] = rng.normal(0, 0.3, 8).astype(np.float32)
    return s


def make_random_s(text_bias: float = 0.5) -> np.ndarray:
    """Generate a random sensory vector."""
    s = np.zeros(D, dtype=np.float32)
    s[:64] = np.random.randn(64).astype(np.float32) * text_bias
    return s
