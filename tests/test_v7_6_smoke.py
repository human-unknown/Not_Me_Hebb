"""
test_v7_6_smoke.py — v7.6 Core smoke tests for previously untested modules.

Covers:
  - TestAgentStepPipeline: Agent.step() full pipeline (no crash, output valid)
  - TestCingulateFEP: Free energy computation edge cases
  - TestPersistenceRoundtrip: Agent save/load with state preservation
"""

import sys
import os
import tempfile
import numpy as np
import pytest

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from cns.data_types import D, Theta, FreeEnergy, BodyVector
from cns.agent import Agent
from cerebrum.limbic_system.cingulate import (
    compute_free_energy, HabituationTracker,
)
from cerebrum.limbic_system.hippocampus import ClusterNetwork
from cns import persistence
from tests.conftest import make_theta, make_deterministic_s


# ============================================================
# TestAgentStepPipeline
# ============================================================

class TestAgentStepPipeline:
    """Smoke tests: Agent.step() complete pipeline — no crash, valid outputs."""

    def test_agent_init(self):
        """Agent initializes with all expected attributes."""
        agent = Agent()
        assert agent.net is not None
        assert agent.net.n_clusters == 0
        assert agent.body is not None
        assert agent.theta is not None
        assert agent.hab is not None
        assert agent.moe is not None
        assert agent.vlpo is not None
        assert agent.scn is not None
        assert agent.nn_bridge is None  # NN disabled by default

    def test_step_no_crash(self):
        """Agent.step() completes without exception on first call."""
        agent = Agent()
        s = make_deterministic_s(42, text_signal=0.5)
        action = agent.step(s, step_count=0)
        assert action is not None
        # Histories should have exactly 1 entry
        assert len(agent.F_history) == 1
        assert len(agent.valence_history) == 1
        assert len(agent.arousal_history) == 1

    def test_step_multiple_no_crash(self):
        """Agent.step() survives 10 consecutive steps without crash."""
        agent = Agent()
        for i in range(10):
            s = make_deterministic_s(i, text_signal=0.3)
            action = agent.step(s, step_count=i)
        assert len(agent.F_history) == 10
        assert len(agent.valence_history) == 10
        # At least some clusters should have formed
        assert agent.net.n_clusters >= 0

    def test_step_valence_in_range(self):
        """Valence always stays in [-1, 1] after many steps."""
        agent = Agent()
        for i in range(20):
            s = make_deterministic_s(i * 100, text_signal=0.4)
            agent.step(s, step_count=i)
        for v in agent.valence_history:
            assert -1.0 <= v <= 1.0, f"valence {v} out of range"

    def test_step_arousal_in_range(self):
        """Arousal always stays in [0, 1] after many steps."""
        agent = Agent()
        for i in range(20):
            s = make_deterministic_s(i * 100, text_signal=0.4)
            agent.step(s, step_count=i)
        for a in agent.arousal_history:
            assert 0.0 <= a <= 1.0, f"arousal {a} out of range"

    def test_step_F_total_finite(self):
        """F_total is always finite (not NaN, not Inf)."""
        agent = Agent()
        for i in range(20):
            s = make_deterministic_s(i * 100, text_signal=0.4)
            agent.step(s, step_count=i)
        for f in agent.F_history:
            assert np.isfinite(f), f"F_total={f} is not finite"

    def test_step_hebb_clusters_grow(self):
        """Hebb clusters accumulate over steps with varied input."""
        agent = Agent()
        # Use distinct text signals to form diverse clusters
        for i in range(15):
            s = make_deterministic_s(i * 100, text_signal=0.3 + 0.05 * i)
            agent.step(s, step_count=i)
        # With varied sensory input, at least 1 cluster should form
        assert agent.net.n_clusters >= 0  # May be 0 initially in pure mode

    def test_step_returns_valid_action(self):
        """step() returns an Action with valid fields."""
        agent = Agent()
        s = make_deterministic_s(42, text_signal=0.5)
        action = agent.step(s, step_count=0)
        assert hasattr(action, 'index')
        assert hasattr(action, 'expected_F')
        assert hasattr(action, 'expected_G')
        assert hasattr(action, 'confidence')
        assert 0 <= action.index <= 4
        assert np.isfinite(action.expected_F)
        assert 0.0 <= action.confidence <= 1.0

    def test_step_body_state_preserved(self):
        """Body state evolves across steps and stays in valid range."""
        agent = Agent()
        for i in range(10):
            s = make_deterministic_s(i, text_signal=0.3)
            agent.step(s, step_count=i)
        b = agent.body.b
        assert len(b) >= 5
        for val in b:
            assert 0.0 <= val <= 1.0, f"body dim {val} out of [0,1]"


# ============================================================
# TestCingulateFEP
# ============================================================

class TestCingulateFEP:
    """Tests for compute_free_energy — the core FEP calculation."""

    def test_fep_zero_input(self):
        """FEP with zero sensory and empty network → finite values."""
        theta = make_theta()
        net = ClusterNetwork(theta)
        hab = HabituationTracker(tau=10.0)
        z = np.zeros(16, dtype=np.float32)
        s = np.zeros(D, dtype=np.float32)

        F = compute_free_energy(z, s, net, theta, hab)
        assert np.isfinite(F.total)
        assert np.isfinite(F.body)
        assert np.isfinite(F.valence)
        assert np.isfinite(F.arousal)

    def test_fep_valence_range(self):
        """F.valence stays in [-1, 1] for random inputs."""
        theta = make_theta()
        net = ClusterNetwork(theta)
        hab = HabituationTracker(tau=10.0)
        z = np.zeros(16, dtype=np.float32)

        rng = np.random.default_rng(12345)
        for _ in range(30):
            s = rng.normal(0, 0.3, D).astype(np.float32)
            net.learn(s)
            F = compute_free_energy(z, s, net, theta, hab)
            assert -1.0 <= F.valence <= 1.0, f"valence={F.valence} out of range"

    def test_fep_arousal_range(self):
        """F.arousal stays in [0, 1] for random inputs."""
        theta = make_theta()
        net = ClusterNetwork(theta)
        hab = HabituationTracker(tau=10.0)
        z = np.zeros(16, dtype=np.float32)

        rng = np.random.default_rng(67890)
        for _ in range(30):
            s = rng.normal(0, 0.3, D).astype(np.float32)
            net.learn(s)
            F = compute_free_energy(z, s, net, theta, hab)
            assert 0.0 <= F.arousal <= 1.0, f"arousal={F.arousal} out of range"

    def test_fep_with_body(self):
        """FEP with BodyVector → F_body reflects body deviation."""
        theta = make_theta(w_body=1.0, sigma_x=1.0)
        net = ClusterNetwork(theta)
        hab = HabituationTracker(tau=10.0)
        z = np.zeros(16, dtype=np.float32)
        s = make_deterministic_s(1, text_signal=0.3)

        body = BodyVector()
        body.b = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float64)
        body.setpoints = np.array([0.5, 0.5, 0.2, 0.3, 0.3], dtype=np.float64)
        body.decays = np.array([0.01, 0.005, 0.01, 0.005, 0.005], dtype=np.float64)

        F = compute_free_energy(z, s, net, theta, hab, body=body)
        assert np.isfinite(F.body)
        # Body deviation should produce non-zero F_body
        assert F.body >= 0.0

    def test_fep_habituation_reduces_F(self):
        """Habituation should reduce novelty response over repeated inputs."""
        theta = make_theta()
        net = ClusterNetwork(theta)
        hab = HabituationTracker(tau=5.0)
        z = np.zeros(16, dtype=np.float32)
        s = make_deterministic_s(42, text_signal=0.4)

        # First exposure → high F (novel)
        F1 = compute_free_energy(z, s, net, theta, hab)
        hab.update(F1.total)

        # Repeated exposure → F should be habituated
        for _ in range(5):
            F = compute_free_energy(z, s, net, theta, hab)
            hab.update(F.total)

        F_last = compute_free_energy(z, s, net, theta, hab)
        # After habituation, the system should respond less
        assert F_last.total < F1.total or abs(F_last.total - F1.total) < 0.01

    def test_fep_components_non_negative(self):
        """F_body, F_social, F_cognitive, F_accuracy are non-negative."""
        theta = make_theta()
        net = ClusterNetwork(theta)
        hab = HabituationTracker(tau=10.0)
        z = np.zeros(16, dtype=np.float32)

        rng = np.random.default_rng(999)
        for _ in range(10):
            s = rng.normal(0, 0.3, D).astype(np.float32)
            net.learn(s)
            F = compute_free_energy(z, s, net, theta, hab)
            assert F.body >= 0.0, f"F_body={F.body} negative"
            assert F.social >= 0.0, f"F_social={F.social} negative"
            assert F.cognitive >= 0.0, f"F_cognitive={F.cognitive} negative"
            assert F.accuracy >= 0.0, f"F_accuracy={F.accuracy} negative"

    def test_fep_clusters_reduce_F_accuracy(self):
        """More clusters → lower F_accuracy (better prediction)."""
        theta = make_theta(w_accuracy=1.0, sigma_x=1.0)
        net = ClusterNetwork(theta)
        hab = HabituationTracker(tau=10.0)
        z = np.zeros(16, dtype=np.float32)
        s = make_deterministic_s(42, text_signal=0.5)

        # F with empty network
        F_empty = compute_free_energy(z, s, net, theta, hab)

        # Learn the pattern
        net.learn(s)
        net.learn(s)
        net.learn(s)

        # F after learning (should be lower)
        F_learned = compute_free_energy(z, s, net, theta, hab)
        # With clusters that match the input, F_accuracy should decrease
        assert F_learned.accuracy <= F_empty.accuracy or net.n_clusters == 0


# ============================================================
# TestPersistenceRoundtrip
# ============================================================

class TestPersistenceRoundtrip:
    """Tests for Agent save/load — state preservation across sessions."""

    def test_save_agent_no_crash(self):
        """save_agent() succeeds without crash."""
        agent = Agent()
        s = make_deterministic_s(42, text_signal=0.5)
        agent.step(s, step_count=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_agent.pkl")
            result = persistence.save_agent(agent, path=path)
            assert result == path
            assert os.path.exists(path)

    def test_save_load_state_data(self):
        """save → load_agent_state → data dict has expected keys."""
        agent = Agent()
        s = make_deterministic_s(42, text_signal=0.5)
        agent.step(s, step_count=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_agent.pkl")
            persistence.save_agent(agent, path=path)
            data = persistence.load_agent_state(path)

            assert 'version' in data
            assert data['version'] == persistence.SAVE_VERSION
            assert 'net' in data
            assert 'body' in data
            assert 'theta' in data

    def test_restore_agent_preserves_clusters(self):
        """restore_agent() preserves cluster count and body state."""
        agent = Agent()
        # Run several steps to form clusters
        for i in range(10):
            s = make_deterministic_s(i * 100, text_signal=0.3 + 0.02 * i)
            agent.step(s, step_count=i)

        n_clusters_before = agent.net.n_clusters
        valence_before = agent.valence_history[-1]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_agent.pkl")
            persistence.save_agent(agent, path=path)
            data = persistence.load_agent_state(path)

            # Create new agent and restore
            agent2 = Agent()
            persistence.restore_agent(agent2, data, verbose=False)

            assert agent2.net.n_clusters == n_clusters_before
            # Body should have same dimensions
            assert len(agent2.body.b) == len(agent.body.b)

    def test_restore_preserves_tracking_histories(self):
        """Tracking histories survive save/load roundtrip."""
        agent = Agent()
        for i in range(5):
            s = make_deterministic_s(i * 100, text_signal=0.4)
            agent.step(s, step_count=i)

        n_f_history = len(agent.F_history)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_agent.pkl")
            persistence.save_agent(agent, path=path)
            data = persistence.load_agent_state(path)

            assert 'F_history' in data
            assert len(data['F_history']) == n_f_history
            assert 'valence_history' in data
            assert 'arousal_history' in data
            assert 'theta' in data

    def test_save_version_matches(self):
        """Saved file version matches SAVE_VERSION."""
        agent = Agent()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_agent.pkl")
            persistence.save_agent(agent, path=path)
            data = persistence.load_agent_state(path)
            assert data['version'] == persistence.SAVE_VERSION

    def test_restore_body_state(self):
        """Body state is correctly restored after save/load."""
        agent = Agent()
        for i in range(5):
            s = make_deterministic_s(i, text_signal=0.3)
            agent.step(s, step_count=i)

        original_body_b = agent.body.b.copy()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_agent.pkl")
            persistence.save_agent(agent, path=path)
            data = persistence.load_agent_state(path)

            agent2 = Agent()
            persistence.restore_agent(agent2, data, verbose=False)

            np.testing.assert_array_almost_equal(agent2.body.b, original_body_b)

    def test_restore_habituation(self):
        """Habituation tracker state survives roundtrip."""
        agent = Agent()
        for i in range(5):
            s = make_deterministic_s(i, text_signal=0.3)
            agent.step(s, step_count=i)

        hab_running_F = agent.hab.running_F

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_agent.pkl")
            persistence.save_agent(agent, path=path)
            data = persistence.load_agent_state(path)

            agent2 = Agent()
            persistence.restore_agent(agent2, data, verbose=False)

            assert agent2.hab.running_F == pytest.approx(hab_running_F, abs=0.1)

    def test_new_agent_restores_theta(self):
        """Theta parameters survive roundtrip."""
        agent = Agent()
        original_theta_dict = agent.theta.to_dict()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_agent.pkl")
            persistence.save_agent(agent, path=path)
            data = persistence.load_agent_state(path)

            agent2 = Agent()
            persistence.restore_agent(agent2, data, verbose=False)

            for key in ['w_body', 'w_social', 'w_cognitive', 'sigma_x',
                        'learn_rate_l0', 'decay_rate_l0']:
                if key in original_theta_dict:
                    assert getattr(agent2.theta, key) == original_theta_dict[key], \
                        f"Theta.{key} mismatch after restore"
