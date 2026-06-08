"""Test LGN V1 feedback gain modulation and tonic/burst gating."""
import sys
import os
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

import numpy as np
from cerebrum.thalamus.lgn import LGN


def test_lgn_tonic_burst_gating():
    """Burst mode should attenuate signals vs tonic mode."""
    lgn = LGN()
    M = np.ones(1024, dtype=np.float32) * 0.5
    P = np.ones(1024, dtype=np.float32) * 0.5
    K = np.ones(1024, dtype=np.float32) * 0.5

    tonic = lgn.relay(M, P, K, brainstem_arousal=0.8)
    burst = lgn.relay(M, P, K, brainstem_arousal=0.1)

    assert np.linalg.norm(burst['M']) < np.linalg.norm(tonic['M']), \
        "Burst mode should attenuate M signal"
    assert np.linalg.norm(burst['P']) < np.linalg.norm(tonic['P']), \
        "Burst mode should attenuate P signal"


def test_lgn_v1_feedback_gain():
    """V1 feedback should modulate LGN layer gains."""
    lgn = LGN()
    M = np.random.randn(1024).astype(np.float32) * 0.1
    P = np.random.randn(1024).astype(np.float32) * 0.1
    K = np.random.randn(1024).astype(np.float32) * 0.1

    # Feedback that boosts M, suppresses P
    fb = np.array([1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 0.0], dtype=np.float32)

    out = lgn.relay(M, P, K, brainstem_arousal=0.8, v1_feedback=fb)
    state = lgn.get_state()
    assert state['M_gain_mean'] > 1.0, \
        f"M gain should be boosted, got {state['M_gain_mean']}"
    assert state['P_gain_mean'] < 1.0, \
        f"P gain should be suppressed, got {state['P_gain_mean']}"


def test_lgn_output_shapes():
    """LGN relay should preserve input dimensions."""
    lgn = LGN(M_dim=512, P_dim=512, K_dim=512)
    M = np.zeros(512, dtype=np.float32)
    P = np.zeros(512, dtype=np.float32)
    K = np.zeros(512, dtype=np.float32)
    out = lgn.relay(M, P, K)
    assert out['M'].shape == (512,)
    assert out['P'].shape == (512,)
    assert out['K'].shape == (512,)


def test_lgn_tonic_mode_label():
    """High arousal should produce 'tonic' mode label."""
    lgn = LGN()
    M = np.zeros(1024, dtype=np.float32)
    P = np.zeros(1024, dtype=np.float32)
    K = np.zeros(1024, dtype=np.float32)
    lgn.relay(M, P, K, brainstem_arousal=0.8)
    state = lgn.get_state()
    assert state['mode'] == 'tonic', f"Expected 'tonic', got {state['mode']}"


def test_lgn_burst_mode_label():
    """Low arousal should produce 'burst' mode label."""
    lgn = LGN()
    M = np.zeros(1024, dtype=np.float32)
    P = np.zeros(1024, dtype=np.float32)
    K = np.zeros(1024, dtype=np.float32)
    lgn.relay(M, P, K, brainstem_arousal=0.1)
    state = lgn.get_state()
    assert state['mode'] == 'burst', f"Expected 'burst', got {state['mode']}"


if __name__ == '__main__':
    test_lgn_tonic_burst_gating()
    test_lgn_v1_feedback_gain()
    test_lgn_output_shapes()
    test_lgn_tonic_mode_label()
    test_lgn_burst_mode_label()
    print("All LGN tests PASSED")
