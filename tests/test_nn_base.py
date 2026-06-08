"""Test cns.nn — 神经网络支撑层 (v7.0 Phase A).

Tests: NNConfig, bridge (numpy↔tensor), NeuralModule base class,
       save/load round-trip, encoder interfaces.
"""
import sys
import os
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

import numpy as np
import tempfile
import pytest

# Skip all tests if PyTorch not available
torch = None
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from cns.nn.config import NNConfig, DEFAULT_NN_CONFIG
from cns.nn.bridge import (
    get_device, numpy_to_torch, torch_to_numpy,
    ensure_numpy, ensure_tensor, is_torch_available,
)
from cns.nn.base import NeuralModule
from cns.nn.interfaces import TextEncoder, VisualEncoder, AudioEncoder


# ================================================================
# Test NNConfig
# ================================================================

def test_default_config():
    """Default config should have sensible values."""
    cfg = DEFAULT_NN_CONFIG
    assert cfg.device == "auto"
    assert cfg.dtype == "float32"
    assert cfg.training_enabled is True
    assert cfg.learning_rate == 1e-3
    assert cfg.grad_clip == 1.0
    assert cfg.model_dir == ".notme/models"


def test_config_effective_lr():
    """effective_lr should prefer module-level over global."""
    cfg = NNConfig(learning_rate=1e-3, text_lr=5e-4)
    # No module override → global
    assert cfg.effective_lr() == 1e-3
    assert cfg.effective_lr(None) == 1e-3
    # Module override → module-specific
    assert cfg.effective_lr(5e-4) == 5e-4
    # text_lr field → still needs explicit pass
    assert cfg.text_lr == 5e-4


def test_config_custom():
    """Custom config should override defaults."""
    cfg = NNConfig(
        device="cpu",
        dtype="float16",
        training_enabled=False,
        learning_rate=1e-4,
    )
    assert cfg.device == "cpu"
    assert cfg.dtype == "float16"
    assert cfg.training_enabled is False
    assert cfg.learning_rate == 1e-4


# ================================================================
# Test Bridge (numpy ↔ tensor)
# ================================================================

@pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch not installed")
class TestBridge:
    """Tests for numpy-tensor bridge utilities."""

    def test_is_torch_available(self):
        assert is_torch_available() is True

    def test_get_device_auto(self):
        device = get_device("auto")
        assert device in ("cpu", "cuda:0", "mps")

    def test_get_device_cpu(self):
        assert get_device("cpu") == "cpu"

    def test_numpy_to_torch_roundtrip(self):
        arr = np.random.randn(10, 64).astype(np.float32)
        tensor = numpy_to_torch(arr, device="cpu")
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (10, 64)
        assert tensor.dtype == torch.float32
        # Round-trip
        arr2 = torch_to_numpy(tensor)
        assert isinstance(arr2, np.ndarray)
        assert arr2.shape == (10, 64)
        np.testing.assert_array_almost_equal(arr, arr2, decimal=5)

    def test_numpy_to_torch_1d(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tensor = numpy_to_torch(arr, device="cpu")
        assert tensor.shape == (3,)
        assert tensor.device.type == "cpu"

    def test_numpy_to_torch_dtype(self):
        arr = np.ones(5, dtype=np.float32)
        t_f16 = numpy_to_torch(arr, device="cpu", dtype="float16")
        assert t_f16.dtype == torch.float16

    def test_torch_to_numpy_grad(self):
        """torch_to_numpy should detach gradients."""
        tensor = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        tensor2 = tensor * 2  # has grad
        arr = torch_to_numpy(tensor2)
        assert isinstance(arr, np.ndarray)
        np.testing.assert_array_equal(arr, np.array([2.0, 4.0, 6.0], dtype=np.float32))

    def test_ensure_numpy_from_tensor(self):
        tensor = torch.tensor([1.0, 2.0], dtype=torch.float32)
        arr = ensure_numpy(tensor)
        assert isinstance(arr, np.ndarray)
        np.testing.assert_array_equal(arr, np.array([1.0, 2.0], dtype=np.float32))

    def test_ensure_numpy_from_list(self):
        arr = ensure_numpy([1, 2, 3])
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float32

    def test_ensure_numpy_from_array(self):
        arr_in = np.array([1.0, 2.0], dtype=np.float64)
        arr_out = ensure_numpy(arr_in)
        assert arr_out.dtype == np.float32

    def test_ensure_tensor_from_numpy(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tensor = ensure_tensor(arr, device="cpu")
        assert isinstance(tensor, torch.Tensor)
        assert tensor.device.type == "cpu"

    def test_ensure_tensor_from_list(self):
        tensor = ensure_tensor([1, 2, 3], device="cpu")
        assert isinstance(tensor, torch.Tensor)
        assert tuple(tensor.shape) == (3,)

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            ensure_numpy("invalid")
        with pytest.raises(TypeError):
            ensure_tensor({"nope": 1})


# ================================================================
# Test NeuralModule Base Class
# ================================================================

# Minimal concrete implementation for testing
class _DummyEncoder(NeuralModule):
    """Simple linear encoder for testing NeuralModule base class."""

    def _build_network(self):
        self._net = torch.nn.Linear(10, 5)

    def _forward_impl(self, x):
        return self._net(x)

    def _train_step_impl(self, batch):
        x = batch["input"]
        target = batch.get("target", x)
        pred = self._net(x)
        loss = torch.nn.functional.mse_loss(pred, target)
        if self._optimizer is None:
            self._optimizer = torch.optim.SGD(self._net.parameters(), lr=0.01)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        return {"loss": loss.item()}


@pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch not installed")
class TestNeuralModule:
    """Tests for the NeuralModule abstract base class."""

    def test_init(self):
        cfg = NNConfig(device="cpu")
        encoder = _DummyEncoder("test_enc", config=cfg)
        assert encoder.name == "test_enc"
        assert encoder.device == "cpu"
        assert encoder.trainable is True
        assert encoder.has_network is True
        assert encoder.step_count == 0
        assert encoder.total_trained == 0

    def test_forward(self):
        encoder = _DummyEncoder("test_enc", config=NNConfig(device="cpu"))
        x = np.random.randn(10).astype(np.float32)
        out = encoder.forward(x)
        assert isinstance(out, np.ndarray)
        assert out.shape == (5,)
        assert out.dtype == np.float32

    def test_forward_batch(self):
        encoder = _DummyEncoder("test_enc", config=NNConfig(device="cpu"))
        x = np.random.randn(4, 10).astype(np.float32)
        out = encoder.forward(x)
        assert out.shape == (4, 5)

    def test_train_step(self):
        encoder = _DummyEncoder("test_enc", config=NNConfig(device="cpu"))
        x = np.random.randn(4, 10).astype(np.float32)
        t = np.random.randn(4, 5).astype(np.float32)
        losses = encoder.train_step({"input": x, "target": t})
        assert "loss" in losses
        assert losses["loss"] > 0
        assert encoder.step_count == 1
        assert encoder.total_trained == 1

    def test_train_disabled(self):
        cfg = NNConfig(device="cpu", training_enabled=False)
        encoder = _DummyEncoder("test_enc", config=cfg)
        losses = encoder.train_step({
            "input": np.random.randn(2, 10).astype(np.float32),
            "target": np.random.randn(2, 5).astype(np.float32),
        })
        # Should return zero loss without training
        assert losses["loss"] == 0.0
        assert encoder.total_trained == 0

    def test_not_trainable(self):
        encoder = _DummyEncoder("test_enc", config=NNConfig(device="cpu"),
                                trainable=False)
        losses = encoder.train_step({
            "input": np.random.randn(2, 10).astype(np.float32),
            "target": np.random.randn(2, 5).astype(np.float32),
        })
        assert losses["loss"] == 0.0

    def test_train_eval_mode(self):
        encoder = _DummyEncoder("test_enc", config=NNConfig(device="cpu"))
        encoder.train()
        assert encoder._train_mode is True
        encoder.eval()
        assert encoder._train_mode is False

    def test_repr(self):
        encoder = _DummyEncoder("test_enc", config=NNConfig(device="cpu"))
        r = repr(encoder)
        assert "test_enc" in r
        assert "_DummyEncoder" in r

    def test_version(self):
        encoder = _DummyEncoder("test_enc", config=NNConfig(device="cpu"))
        assert encoder._version == "7.0"


@pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch not installed")
class TestNeuralModulePersistence:
    """Tests for NeuralModule save/load."""

    def test_save_load_roundtrip(self):
        encoder = _DummyEncoder("test_save", config=NNConfig(device="cpu"))
        # Do a training step to get non-initial weights
        x = np.random.randn(4, 10).astype(np.float32)
        t = np.random.randn(4, 5).astype(np.float32)
        encoder.train_step({"input": x, "target": t})

        # Get weights before save
        w_before = encoder._net.weight.data.clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.pt")
            saved_path = encoder.save(path)
            assert saved_path == path
            assert os.path.exists(path)

            # Create new encoder and load
            encoder2 = _DummyEncoder("test_save", config=NNConfig(device="cpu"))
            ok = encoder2.load(path)
            assert ok is True

            # Weights should match
            w_after = encoder2._net.weight.data
            assert torch.allclose(w_before, w_after, atol=1e-6)

    def test_load_missing_file(self):
        encoder = _DummyEncoder("test_miss", config=NNConfig(device="cpu"))
        ok = encoder.load("/nonexistent/path/model.pt")
        assert ok is False

    def test_save_auto_path(self):
        encoder = _DummyEncoder("test_auto", config=NNConfig(device="cpu"))
        # Should not raise
        saved = encoder.save()
        assert saved.endswith(".pt")
        assert "test_auto" in saved
        # Cleanup
        if os.path.exists(saved):
            os.unlink(saved)

    def test_save_includes_metadata(self):
        encoder = _DummyEncoder("test_meta", config=NNConfig(device="cpu"))
        encoder.train_step({
            "input": np.random.randn(2, 10).astype(np.float32),
            "target": np.random.randn(2, 5).astype(np.float32),
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "meta.pt")
            encoder.save(path)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            assert checkpoint["name"] == "test_meta"
            assert checkpoint["version"] == "7.0"
            assert checkpoint["step_count"] == 1
            assert checkpoint["total_trained"] == 1


# ================================================================
# Test Encoder Interfaces
# ================================================================

class TestEncoderInterfaces:
    """Test that encoder interfaces define correct constants."""

    def test_text_encoder_output_dim(self):
        assert TextEncoder.OUTPUT_DIM == 64

    def test_visual_encoder_output_dim(self):
        assert VisualEncoder.OUTPUT_DIM == 308

    def test_audio_encoder_output_dim(self):
        assert AudioEncoder.OUTPUT_DIM == 96

    def test_visual_subpathways_sum(self):
        """Sub-pathway dims should sum exactly to 308."""
        total = sum(VisualEncoder.SUBPATHWAYS.values())
        assert total == 308, f"Sub-pathway sum {total} != 308"

    def test_audio_submodules_sum(self):
        """Sub-module dims should sum exactly to 96."""
        total = sum(AudioEncoder.SUBMODULES.values())
        assert total == 96, f"Sub-module sum {total} != 96"

    def test_text_encoder_is_abstract(self):
        """Cannot instantiate abstract TextEncoder directly."""
        with pytest.raises(TypeError):
            TextEncoder(config=NNConfig(device="cpu"))

    def test_visual_encoder_is_abstract(self):
        with pytest.raises(TypeError):
            VisualEncoder(config=NNConfig(device="cpu"))

    def test_audio_encoder_is_abstract(self):
        with pytest.raises(TypeError):
            AudioEncoder(config=NNConfig(device="cpu"))


# ================================================================
# Integration: full sensory pipeline with dummy modules
# ================================================================

@pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch not installed")
class TestIntegration:
    """Integration tests for NN module with simulated full pipeline."""

    def test_sensory_roundtrip_dimensions(self):
        """Verify that encoder outputs match D=516 layout dimensions."""
        cfg = NNConfig(device="cpu")

        # Create minimal concrete encoders for testing
        class _MinimalTextEncoder(TextEncoder):
            def _build_network(self):
                self._net = torch.nn.Linear(64, 64)
            def _forward_impl(self, x):
                return self._net(x)
            def encode(self, text):
                return self.forward(np.random.randn(64).astype(np.float32))

        class _MinimalVisualEncoder(VisualEncoder):
            def _build_network(self):
                self._net = torch.nn.Linear(308, 308)
            def _forward_impl(self, x):
                return self._net(x)
            def encode(self, image):
                return self.forward(np.random.randn(308).astype(np.float32))

        class _MinimalAudioEncoder(AudioEncoder):
            def _build_network(self):
                self._net = torch.nn.Linear(96, 96)
            def _forward_impl(self, x):
                return self._net(x)
            def encode(self, audio, sample_rate=16000):
                return self.forward(np.random.randn(96).astype(np.float32))

        text_enc = _MinimalTextEncoder(config=cfg)
        visual_enc = _MinimalVisualEncoder(config=cfg)
        audio_enc = _MinimalAudioEncoder(config=cfg)

        # Encode
        text_vec = text_enc.encode("hello")
        visual_vec = visual_enc.encode(np.zeros((64, 64, 3), dtype=np.uint8))
        audio_vec = audio_enc.encode(np.zeros(16000, dtype=np.float32))

        # Check dimensions match D=516 layout
        assert text_vec.shape == (64,), f"Text: expected (64,), got {text_vec.shape}"
        assert visual_vec.shape == (308,), f"Visual: expected (308,), got {visual_vec.shape}"
        assert audio_vec.shape == (96,), f"Audio: expected (96,), got {audio_vec.shape}"

        # Assemble full sensory vector
        sensory = np.concatenate([text_vec, visual_vec, audio_vec,
                                   np.zeros(48, dtype=np.float32)])  # pain stays hand-crafted
        assert sensory.shape == (516,), f"Full sensory: expected (516,), got {sensory.shape}"

    def test_train_forward_consistency(self):
        """Forward should give same result in train and eval mode (deterministic)."""
        encoder = _DummyEncoder("test_cons", config=NNConfig(device="cpu"))
        x = np.random.randn(3, 10).astype(np.float32)

        encoder.eval()
        out_eval = encoder.forward(x)

        encoder.train()
        out_train = encoder.forward(x)

        # forward() uses no_grad, so train/eval mode should match
        np.testing.assert_array_almost_equal(out_eval, out_train, decimal=5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
