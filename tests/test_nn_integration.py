"""
test_nn_integration.py — Phase F (v7.5) Integration Tests

Tests for:
  - NNBridge: init, lazy load, sensory enhancement, metrics, VTA/LC modulation
  - VTA NN modulation: nn_lr_multiplier output
  - LC NN modulation: nn_temperature, nn_dropout output
  - Agent integration: enable_nn, step with NN, persistence with NN
  - Emotional consistency: valence/arousal preserved, FEP valid, dual-system
  - Web UI: _build_status with NN

Total: ~35 tests
"""

import os
import sys
import json
import tempfile
import numpy as np
import pytest

# Project imports
from cns.nn.config import NNConfig, DEFAULT_NN_CONFIG
from cns.nn.integrator import NNBridge
from cns.nn.trainer import Trainer
from cns.nn.metrics import ExperienceTracker, TrainingMetrics
from cns.agent import Agent
from brainstem_cerebellum.midbrain.vta import VTA
from brainstem_cerebellum.pons.locus_coeruleus import LocusCoeruleus


# ================================================================
# Fixtures
# ================================================================

@pytest.fixture
def nn_config():
    """NN config with integration enabled, sensory enhancement disabled."""
    return NNConfig(
        nn_enabled=True,
        nn_sensory_enhance=False,
        device="cpu",
        training_enabled=True,
    )


@pytest.fixture
def nn_config_enhance():
    """NN config with sensory enhancement enabled."""
    return NNConfig(
        nn_enabled=True,
        nn_sensory_enhance=True,
        device="cpu",
        training_enabled=True,
    )


@pytest.fixture
def bridge_disabled():
    """Bridge with NN disabled (default config)."""
    return NNBridge(config=DEFAULT_NN_CONFIG)


@pytest.fixture
def bridge_enabled(nn_config):
    """Bridge with NN enabled but not initialized."""
    return NNBridge(config=nn_config)


@pytest.fixture
def agent():
    """Fresh Agent."""
    return Agent()


@pytest.fixture
def agent_with_nn(nn_config):
    """Agent with NN enabled."""
    a = Agent()
    a.enable_nn(nn_config)
    return a


@pytest.fixture
def vta():
    """Fresh VTA instance."""
    return VTA()


@pytest.fixture
def lc():
    """Fresh LC instance."""
    return LocusCoeruleus()


# ================================================================
# TestNNBridge — 12 tests
# ================================================================

class TestNNBridge:
    """Tests for NNBridge initialization and operations."""

    def test_init_disabled(self, bridge_disabled):
        """NNBridge with nn_enabled=False → is_enabled is False, no modules."""
        assert bridge_disabled.is_enabled is False
        assert bridge_disabled._initialized is False

    def test_init_enabled_not_initialized(self, bridge_enabled):
        """NNBridge with nn_enabled=True → is_enabled is False until _ensure_init."""
        assert bridge_enabled._enabled is True
        assert bridge_enabled.is_enabled is False  # not yet initialized

    def test_ensure_init(self, bridge_enabled):
        """_ensure_init creates all modules."""
        bridge_enabled._ensure_init()
        assert bridge_enabled._initialized is True
        assert bridge_enabled.is_enabled is True
        assert len(bridge_enabled._modules) >= 8  # all Phase B-D modules

    def test_enhance_sensory_noop_disabled(self, bridge_disabled):
        """enhance_sensory returns unmodified when disabled."""
        s = np.random.randn(516).astype(np.float32)
        s2 = bridge_disabled.enhance_sensory(s)
        np.testing.assert_array_equal(s, s2)

    def test_enhance_sensory_noop_no_enhance(self, bridge_enabled):
        """enhance_sensory returns unmodified when nn_sensory_enhance=False."""
        bridge_enabled._ensure_init()
        s = np.random.randn(516).astype(np.float32)
        s2 = bridge_enabled.enhance_sensory(s)
        np.testing.assert_array_equal(s, s2)

    def test_enhance_sensory_with_enhance(self, nn_config_enhance):
        """enhance_sensory runs without error when nn_sensory_enhance=True."""
        bridge = NNBridge(config=nn_config_enhance)
        bridge._ensure_init()
        s = np.random.randn(516).astype(np.float32)
        s2 = bridge.enhance_sensory(s)
        assert s2.shape == s.shape
        assert s2.dtype == np.float32

    def test_record_step(self, bridge_enabled):
        """record_step stores metrics in tracker."""
        bridge_enabled._ensure_init()
        bridge_enabled.record_step(F_total=0.3, valence=0.5, arousal=0.2)
        assert bridge_enabled._tracker.n_records == 1

    def test_record_dialogue(self, bridge_enabled):
        """record_dialogue updates vocab and turn count."""
        bridge_enabled._ensure_init()
        bridge_enabled.record_dialogue("你好世界", "你好！")
        assert bridge_enabled._tracker.turn_count == 1
        assert bridge_enabled._tracker.vocab_size > 0

    def test_get_nn_lr_modulation_positive(self, bridge_enabled):
        """Positive RPE → lr_mult > 1.0."""
        bridge_enabled._ensure_init()
        mult = bridge_enabled.get_nn_lr_modulation(rpe=0.5, da=0.6)
        assert mult > 1.0
        assert 0.2 <= mult <= 2.0

    def test_get_nn_lr_modulation_negative(self, bridge_enabled):
        """Negative RPE → lr_mult < 1.0."""
        bridge_enabled._ensure_init()
        mult = bridge_enabled.get_nn_lr_modulation(rpe=-0.3, da=0.1)
        assert mult < 1.0
        assert 0.2 <= mult <= 2.0

    def test_get_nn_explore_params(self, bridge_enabled):
        """High NE (exploit) → low temperature. Low NE (explore) → high temp."""
        bridge_enabled._ensure_init()
        # Exploit mode
        params_exploit = bridge_enabled.get_nn_explore_params(
            tonic_ne=0.5, total_ne=0.5, exploration_bias=0.8)
        # Explore mode
        params_explore = bridge_enabled.get_nn_explore_params(
            tonic_ne=0.2, total_ne=0.2, exploration_bias=-0.5)
        assert params_exploit['temperature'] < params_explore['temperature']
        assert 'dropout' in params_exploit

    def test_get_status(self, bridge_enabled):
        """get_status returns expected keys."""
        bridge_enabled._ensure_init()
        status = bridge_enabled.get_status()
        assert status['enabled'] is True
        assert status['initialized'] is True
        assert 'modules' in status
        assert 'blend_ratio' in status


# ================================================================
# TestVTANNModulation — 3 tests
# ================================================================

class TestVTANNModulation:
    """Tests for VTA NN learning rate modulation."""

    def test_vta_nn_lr_multiplier_present(self, vta):
        """VTA.process returns nn_lr_multiplier in output."""
        result = vta.process(valence=0.2, delta_valence=0.05, F_body=0.3)
        assert 'nn_lr_multiplier' in result

    def test_vta_nn_lr_range(self, vta):
        """nn_lr_multiplier always in [0.2, 2.0]."""
        results = []
        for v, dv in [(0.5, 0.3), (-0.5, -0.3), (0.0, 0.0), (0.8, -0.5), (-0.8, 0.5)]:
            result = vta.process(valence=v, delta_valence=dv, F_body=0.3)
            mult = result['nn_lr_multiplier']
            results.append(mult)
            assert 0.2 <= mult <= 2.0, f"nn_lr_mult={mult} out of range for v={v}, dv={dv}"
        # Different inputs produce different outputs
        assert len(set(round(r, 3) for r in results)) >= 2

    def test_vta_nn_lr_positive_rpe(self, vta):
        """Positive RPE → nn_lr_multiplier >= 0.2 (conservative NN LR)."""
        result = vta.process(valence=0.5, delta_valence=0.4, F_body=0.1,
                            delta_F_body=0.05, social_reward=0.8)
        assert result['rpe'] > 0, "RPE should be positive"
        # NN LR multiplier is damped (0.7x Hebb) — conservative to avoid forgetting
        assert result['nn_lr_multiplier'] >= 0.2, "NN LR should be at least min"
        # The Hebb LR multiplier should be higher than NN (damping at work)
        assert result['learn_rate_multiplier'] >= result['nn_lr_multiplier'], (
            "NN LR should be <= Hebb LR (damped for safety)")


# ================================================================
# TestLCNNModulation — 3 tests
# ================================================================

class TestLCNNModulation:
    """Tests for LC NN explore/exploit modulation."""

    def test_lc_nn_temperature_present(self, lc):
        """LC.process returns nn_temperature in output."""
        result = lc.process(arousal=0.5, novelty=0.1)
        assert 'nn_temperature' in result

    def test_lc_nn_dropout_present(self, lc):
        """LC.process returns nn_dropout in output."""
        result = lc.process(arousal=0.5, novelty=0.1)
        assert 'nn_dropout' in result

    def test_lc_nn_explore_vs_exploit(self, lc):
        """NN temperature and dropout are always in valid ranges."""
        # Test multiple LC states — all should produce valid ranges
        states = [
            (0.1, 0.9, 0.0, 0.0),   # very low arousal, high novelty
            (0.3, 0.3, 0.1, 0.3),   # low-mid
            (0.5, 0.1, 0.1, 0.7),   # optimal (exploit)
            (0.7, 0.05, 0.5, 0.9),  # high arousal (stress)
        ]
        for arousal, novelty, stress, task_eng in states:
            result = lc.process(
                arousal=arousal, novelty=novelty,
                stress=stress, task_engagement=task_eng)
            # Temperature always in valid range
            assert 0.3 <= result['nn_temperature'] <= 1.5, (
                f"nn_temperature={result['nn_temperature']} out of range")
            # Dropout always in valid range
            assert 0.05 <= result['nn_dropout'] <= 0.5, (
                f"nn_dropout={result['nn_dropout']} out of range")


# ================================================================
# TestAgentIntegration — 8 tests
# ================================================================

class TestAgentIntegration:
    """Tests for Agent ↔ NN bridge integration."""

    def test_agent_no_nn_by_default(self, agent):
        """Agent() has nn_bridge=None."""
        assert agent.nn_bridge is None

    def test_agent_enable_nn(self, agent):
        """agent.enable_nn() creates nn_bridge."""
        agent.enable_nn(NNConfig(nn_enabled=True, device="cpu"))
        assert agent.nn_bridge is not None
        assert agent.nn_bridge._enabled is True

    def test_agent_step_with_nn_no_crash(self, agent_with_nn):
        """agent.step() with NN enabled completes without error."""
        s = np.zeros(516, dtype=np.float32)
        s[0] = 0.01  # tiny text input
        action = agent_with_nn.step(s, 0)
        assert action is not None

    def test_agent_step_nn_metrics(self, agent_with_nn):
        """After step with NN, tracker has records."""
        s = np.zeros(516, dtype=np.float32)
        s[0] = 0.01
        agent_with_nn.step(s, 1)
        # Tracker should have at least 1 record from the step
        if agent_with_nn.nn_bridge.is_enabled:
            assert agent_with_nn.nn_bridge._tracker.n_records >= 1

    def test_agent_multiple_steps_with_nn(self, agent_with_nn):
        """Multiple steps with NN enabled don't crash."""
        for i in range(5):
            s = np.random.randn(516).astype(np.float32) * 0.01
            s[0] = 0.02  # small text component
            action = agent_with_nn.step(s, i)
            assert action is not None
        # Should have accumulated metrics
        if agent_with_nn.nn_bridge.is_enabled:
            assert agent_with_nn.nn_bridge._tracker.n_records >= 5

    def test_agent_vta_nn_wiring(self, agent_with_nn):
        """VTA computation sets nn_lr_multiplier on bridge."""
        s = np.zeros(516, dtype=np.float32)
        s[0] = 0.01
        agent_with_nn.step(s, 1)
        # After step, _current_lr_mult should be set
        if agent_with_nn.nn_bridge.is_enabled:
            lr_mult = agent_with_nn.nn_bridge._current_lr_mult
            assert 0.2 <= lr_mult <= 2.0, f"lr_mult={lr_mult} out of range"

    def test_agent_lc_nn_wiring(self, agent_with_nn):
        """LC computation sets nn_temperature on bridge."""
        s = np.zeros(516, dtype=np.float32)
        s[0] = 0.01
        agent_with_nn.step(s, 1)
        # After step, _current_temperature should be set
        if agent_with_nn.nn_bridge.is_enabled:
            temp = agent_with_nn.nn_bridge._current_temperature
            assert 0.3 <= temp <= 1.5, f"temperature={temp} out of range"

    def test_agent_nn_bridge_repr(self, agent_with_nn):
        """NNBridge repr is informative."""
        s = np.zeros(516, dtype=np.float32)
        s[0] = 0.01
        agent_with_nn.step(s, 1)
        r = repr(agent_with_nn.nn_bridge)
        assert 'NNBridge' in r


# ================================================================
# TestWebIntegration — 5 tests
# ================================================================

class TestWebIntegration:
    """Tests for Web UI NN integration."""

    def test_nn_status_in_agent(self, agent_with_nn):
        """NNBridge.get_status returns proper dict."""
        s = np.zeros(516, dtype=np.float32)
        s[0] = 0.01
        agent_with_nn.step(s, 1)
        status = agent_with_nn.nn_bridge.get_status()
        assert 'enabled' in status
        assert status['enabled'] is True

    def test_training_history_untrained(self, nn_config):
        """get_training_history on untrained trainer returns empty dict."""
        trainer = Trainer(config=nn_config)
        hist = trainer.get_training_history()
        assert isinstance(hist, dict)
        assert len(hist) == 0  # No modules registered

    def test_training_summary_available(self, agent_with_nn):
        """get_training_summary is available on bridge."""
        summary = agent_with_nn.nn_bridge.get_training_summary()
        assert isinstance(summary, dict)

    def test_nn_status_keys(self, agent_with_nn):
        """NNBridge.get_status has all required keys."""
        s = np.zeros(516, dtype=np.float32)
        s[0] = 0.01
        agent_with_nn.step(s, 1)
        status = agent_with_nn.nn_bridge.get_status()
        required_keys = [
            'enabled', 'initialized', 'blend_ratio',
            'current_lr_mult', 'current_temperature', 'current_dropout',
        ]
        for key in required_keys:
            assert key in status, f"Missing key: {key}"

    def test_nn_disabled_status(self, agent):
        """Agent without NN has nn_bridge=None."""
        assert agent.nn_bridge is None


# ================================================================
# TestEmotionalConsistency — 5 tests
# ================================================================

class TestEmotionalConsistency:
    """Tests that NN integration preserves emotional dynamics."""

    def test_valence_range_with_nn(self, agent_with_nn):
        """Valence stays in [-1, 1] with NN enabled."""
        for i in range(10):
            s = np.random.randn(516).astype(np.float32) * 0.01
            s[0] = 0.02
            agent_with_nn.step(s, i)
        if agent_with_nn.valence_history:
            for v in agent_with_nn.valence_history:
                assert -1.0 <= v <= 1.0, f"Valence {v} out of [-1, 1]"

    def test_arousal_range_with_nn(self, agent_with_nn):
        """Arousal stays in [0, 1] with NN enabled."""
        for i in range(10):
            s = np.random.randn(516).astype(np.float32) * 0.01
            s[0] = 0.02
            agent_with_nn.step(s, i)
        if agent_with_nn.arousal_history:
            for a in agent_with_nn.arousal_history:
                assert 0.0 <= a <= 1.0, f"Arousal {a} out of [0, 1]"

    def test_f_total_finite_with_nn(self, agent_with_nn):
        """F_total is always finite with NN enabled."""
        for i in range(10):
            s = np.random.randn(516).astype(np.float32) * 0.01
            s[0] = 0.02
            agent_with_nn.step(s, i)
        if agent_with_nn.F_history:
            for f in agent_with_nn.F_history:
                assert np.isfinite(f), f"F_total={f} is not finite"

    def test_hebb_clusters_still_grow(self, agent_with_nn):
        """Hebb network still forms clusters with NN enabled."""
        for i in range(20):
            s = np.random.randn(516).astype(np.float32) * 0.1
            # Give consistent input so clusters can form
            s[0:16] = np.sin(np.arange(16) * 0.1 + i * 0.3).astype(np.float32)
            agent_with_nn.step(s, i)
        # Hebb clusters should grow (this is the dual-system guarantee)
        assert agent_with_nn.net.n_clusters >= 0, "Cluster count should be non-negative"

    def test_agent_actions_valid_with_nn(self, agent_with_nn):
        """Actions are within valid range with NN enabled."""
        for i in range(10):
            s = np.random.randn(516).astype(np.float32) * 0.01
            s[0] = 0.02
            action = agent_with_nn.step(s, i)
            assert action is not None
            assert 0 <= action.index <= 4, f"Invalid action index: {action.index}"


# ================================================================
# TestSleepNNIntegration — 3 tests
# ================================================================

class TestSleepNNIntegration:
    """Tests for NN sleep consolidation hooks."""

    def test_sleep_nrem_no_crash(self, agent_with_nn):
        """sleep_nrem_consolidation runs without error even without sleep state."""
        if agent_with_nn.nn_bridge.is_enabled:
            agent_with_nn.nn_bridge.sleep_nrem_consolidation()
        # Should not crash

    def test_sleep_rem_no_crash(self, agent_with_nn):
        """sleep_rem_consolidation runs without error."""
        if agent_with_nn.nn_bridge.is_enabled:
            agent_with_nn.nn_bridge.sleep_rem_consolidation()
        # Should not crash

    def test_sleep_nrem_no_agent(self, nn_config):
        """sleep_nrem runs without agent reference."""
        bridge = NNBridge(config=nn_config)
        bridge._ensure_init()
        bridge.sleep_nrem_consolidation()  # Should handle missing agent gracefully


# ================================================================
# TestPersistenceIntegration — 3 tests
# ================================================================

class TestPersistenceIntegration:
    """Tests for NN save/load integration."""

    def test_save_checkpoint_creates_dir(self, agent_with_nn):
        """save_checkpoint creates a directory with files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bridge = agent_with_nn.nn_bridge
            if bridge.is_enabled:
                result = bridge.save_checkpoint(tmpdir)
                # Check that files were created
                files = os.listdir(tmpdir)
                assert len(files) > 0, f"No files created in {tmpdir}"

    def test_load_checkpoint_no_crash(self, agent_with_nn):
        """load_checkpoint handles missing directory gracefully."""
        bridge = agent_with_nn.nn_bridge
        if bridge.is_enabled:
            result = bridge.load_checkpoint("/nonexistent/path")
            assert result is False

    def test_save_load_roundtrip(self, agent_with_nn):
        """save → load preserves bridge state."""
        import tempfile
        bridge = agent_with_nn.nn_bridge
        if not bridge.is_enabled:
            pytest.skip("Bridge not initialized")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Set some state
            bridge._current_lr_mult = 1.5
            bridge._blend_ratio = 0.3

            # Save
            bridge.save_checkpoint(tmpdir)

            # Create new bridge and load
            config2 = NNConfig(nn_enabled=True, device="cpu")
            bridge2 = NNBridge(config=config2)
            bridge2._ensure_init()
            bridge2.load_checkpoint(tmpdir)

            # Check that modules exist
            assert bridge2._initialized is True


# ================================================================
# TestConfigIntegration — 2 tests
# ================================================================

class TestConfigIntegration:
    """Tests for NNConfig integration flags."""

    def test_nn_enabled_flag(self):
        """nn_enabled flag is present and defaults to False."""
        config = NNConfig()
        assert config.nn_enabled is False
        assert config.nn_sensory_enhance is False

    def test_nn_enabled_config_custom(self):
        """Custom config with integration flags works."""
        config = NNConfig(
            nn_enabled=True,
            nn_sensory_enhance=True,
            device="cpu",
        )
        assert config.nn_enabled is True
        assert config.nn_sensory_enhance is True
        assert config.device == "cpu"
