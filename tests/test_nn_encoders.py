"""Test cns.nn encoders — Phase B concrete implementations (v7.0).

Tests: TrainableTextEncoder, TrainableVisualEncoder, TrainableAudioEncoder
       — init, encode, batch encode, training, persistence, integration.
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
from cns.nn.text_encoder import TrainableTextEncoder
from cns.nn.visual_encoder import TrainableVisualEncoder
from cns.nn.audio_encoder import TrainableAudioEncoder
from cns.nn.interfaces import TextEncoder, VisualEncoder, AudioEncoder


# ================================================================
# Test corpus (sample Chinese dialog lines)
# ================================================================

SAMPLE_CORPUS = [
    "你好，今天天气真不错。",
    "我想和你聊聊天。",
    "你觉得人工智能会有情感吗？",
    "自由能原理是很有趣的理论框架。",
    "情感不是标签，是身体稳态的数值产物。",
    "我今天心情很好，想出去走走。",
    "痛苦和快乐都是生命的一部分。",
    "学习需要时间和耐心。",
    "记忆是认知的核心能力。",
    "什么是意识？这是一个很难回答的问题。",
    "我喜欢和你交流。",
    "世界充满了未知和可能性。",
    "每一步都是成长。",
    "语言是人类最伟大的发明之一。",
    "音乐能触动心灵深处。",
]


# ================================================================
# Test TrainableTextEncoder
# ================================================================

@pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch not installed")
class TestTrainableTextEncoder:
    """Tests for the character-level Transformer text encoder."""

    def test_init(self):
        cfg = NNConfig(device="cpu")
        enc = TrainableTextEncoder(config=cfg)
        assert enc.name == "text_encoder"
        assert enc.OUTPUT_DIM == 64
        assert enc._version == "7.0"
        assert enc.d_model == 128
        assert enc.n_layers == 2
        assert enc.max_len == 128
        assert enc.has_network is True

    def test_is_text_encoder_subclass(self):
        cfg = NNConfig(device="cpu")
        enc = TrainableTextEncoder(config=cfg)
        assert isinstance(enc, TextEncoder)

    def test_build_vocab(self):
        cfg = NNConfig(device="cpu")
        enc = TrainableTextEncoder(config=cfg)
        n_chars = enc.build_vocab(SAMPLE_CORPUS)
        assert n_chars > 0
        assert enc.is_vocab_built
        # Should include special tokens
        assert "[PAD]" in enc._char2id
        assert "[UNK]" in enc._char2id
        assert "[MASK]" in enc._char2id
        assert "[CLS]" in enc._char2id

    def test_special_tokens(self):
        cfg = NNConfig(device="cpu")
        enc = TrainableTextEncoder(config=cfg)
        enc.build_vocab(SAMPLE_CORPUS)
        assert enc._char2id["[PAD]"] == 0
        assert enc._char2id["[UNK]"] == 1
        assert enc._char2id["[MASK]"] == 2
        assert enc._char2id["[CLS]"] == 3

    def test_tokenize(self):
        cfg = NNConfig(device="cpu")
        enc = TrainableTextEncoder(config=cfg)
        enc.build_vocab(SAMPLE_CORPUS)
        ids = enc.tokenize("你好世界")
        assert len(ids) == enc.max_len
        # First chars should be valid tokens (not PAD)
        assert ids[0] > 0
        assert ids[1] > 0
        # Later positions should be PAD
        assert all(i == 0 for i in ids[4:])

    def test_tokenize_unknown_char(self):
        cfg = NNConfig(device="cpu")
        enc = TrainableTextEncoder(config=cfg)
        enc.build_vocab(SAMPLE_CORPUS)
        ids = enc.tokenize("Hello世界💡")
        # 'H', 'e', 'l', 'l', 'o', '💡' are unknown → UNK_IDX
        # '世', '界' should be in vocab (from SAMPLE_CORPUS)
        assert ids[0] == 1  # UNK for 'H'
        assert len(ids) == enc.max_len

    def test_encode_output_shape(self):
        cfg = NNConfig(device="cpu")
        enc = TrainableTextEncoder(config=cfg)
        enc.build_vocab(SAMPLE_CORPUS)
        enc.eval()
        vec = enc.encode("你好世界")
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (64,)
        assert vec.dtype == np.float32

    def test_encode_batch(self):
        cfg = NNConfig(device="cpu")
        enc = TrainableTextEncoder(config=cfg)
        enc.build_vocab(SAMPLE_CORPUS)
        enc.eval()
        texts = ["你好世界", "今天天气不错", "人工智能"]
        batch = enc.encode_batch(texts)
        assert batch.shape == (3, 64)
        assert batch.dtype == np.float32

    def test_encode_l2_normalized(self):
        cfg = NNConfig(device="cpu")
        enc = TrainableTextEncoder(config=cfg)
        enc.build_vocab(SAMPLE_CORPUS)
        enc.eval()
        vec = enc.encode("测试文本")
        norm = float(np.linalg.norm(vec))
        assert abs(norm - 1.0) < 0.01, f"Expected L2 norm ≈ 1.0, got {norm}"

    def test_encode_deterministic(self):
        cfg = NNConfig(device="cpu")
        enc = TrainableTextEncoder(config=cfg)
        enc.build_vocab(SAMPLE_CORPUS)
        enc.eval()
        v1 = enc.encode("相同输入")
        v2 = enc.encode("相同输入")
        np.testing.assert_array_almost_equal(v1, v2, decimal=5)

    def test_different_texts_different_vectors(self):
        cfg = NNConfig(device="cpu")
        enc = TrainableTextEncoder(config=cfg)
        enc.build_vocab(SAMPLE_CORPUS)
        enc.eval()
        v1 = enc.encode("你好世界")
        v2 = enc.encode("完全不同的一段文字")
        # Should be different (cosine similarity < 0.99)
        cos_sim = float(np.dot(v1, v2))
        assert cos_sim < 0.999, f"Vectors too similar: cos_sim={cos_sim}"

    def test_train_step_runs(self):
        cfg = NNConfig(device="cpu")
        enc = TrainableTextEncoder(config=cfg)
        enc.build_vocab(SAMPLE_CORPUS)
        enc.train()

        batch_texts = SAMPLE_CORPUS[:4]
        batch_ids = np.array(
            [enc.tokenize(t) for t in batch_texts], dtype=np.int64
        )
        losses = enc.train_step({"input": batch_ids})
        assert "loss" in losses
        assert losses["loss"] > 0
        assert enc.step_count == 1
        assert enc.total_trained == 1

    def test_pretrain_smoke(self):
        cfg = NNConfig(device="cpu")
        enc = TrainableTextEncoder(config=cfg)
        # Use a larger sample for pretraining
        corpus = SAMPLE_CORPUS * 10  # 150 lines
        history = enc.pretrain(corpus, epochs=2, batch_size=8, verbose=False)
        assert len(history) == 2
        assert history[0]["loss"] > 0
        # Loss should generally decrease
        assert history[-1]["loss"] < history[0]["loss"] * 2.0, (
            f"Loss did not stabilize: {history}"
        )

    def test_encode_after_pretrain(self):
        cfg = NNConfig(device="cpu")
        enc = TrainableTextEncoder(config=cfg)
        enc.pretrain(SAMPLE_CORPUS * 5, epochs=1, batch_size=8, verbose=False)
        enc.eval()
        vec = enc.encode("记忆是认知的核心")
        assert vec.shape == (64,)
        assert np.all(np.isfinite(vec))

    def test_max_len_truncation(self):
        cfg = NNConfig(device="cpu")
        enc = TrainableTextEncoder(config=cfg, max_len=8)
        enc.build_vocab(SAMPLE_CORPUS)
        # Tokenize long text → should be truncated to max_len
        ids = enc.tokenize("这是一个很长的测试句子用来验证截断功能")
        assert len(ids) == 8


# ================================================================
# Test TrainableVisualEncoder
# ================================================================

@pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch not installed")
class TestTrainableVisualEncoder:
    """Tests for the CNN visual encoder with sub-pathway heads."""

    def test_init(self):
        cfg = NNConfig(device="cpu")
        enc = TrainableVisualEncoder(config=cfg)
        assert enc.name == "visual_encoder"
        assert enc.OUTPUT_DIM == 308
        assert enc._version == "7.0"
        assert enc.input_size == 64
        assert enc.has_network is True

    def test_is_visual_encoder_subclass(self):
        cfg = NNConfig(device="cpu")
        enc = TrainableVisualEncoder(config=cfg)
        assert isinstance(enc, VisualEncoder)

    def test_encode_output_shape(self):
        cfg = NNConfig(device="cpu")
        enc = TrainableVisualEncoder(config=cfg)
        enc.eval()
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        vec = enc.encode(img)
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (308,)
        assert vec.dtype == np.float32

    def test_encode_float_image(self):
        cfg = NNConfig(device="cpu")
        enc = TrainableVisualEncoder(config=cfg)
        enc.eval()
        img = np.random.rand(64, 64, 3).astype(np.float32)
        vec = enc.encode(img)
        assert vec.shape == (308,)

    def test_encode_batch(self):
        cfg = NNConfig(device="cpu")
        enc = TrainableVisualEncoder(config=cfg)
        enc.eval()
        images = np.random.randint(0, 255, (4, 64, 64, 3), dtype=np.uint8)
        batch = enc.encode_batch(images)
        assert batch.shape == (4, 308)

    def test_preprocess_resize_larger(self):
        cfg = NNConfig(device="cpu")
        enc = TrainableVisualEncoder(config=cfg)
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        processed = enc.preprocess_image(img)
        assert processed.shape == (1, 3, 64, 64)

    def test_preprocess_resize_smaller(self):
        cfg = NNConfig(device="cpu")
        enc = TrainableVisualEncoder(config=cfg)
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        processed = enc.preprocess_image(img)
        assert processed.shape == (1, 3, 64, 64)

    def test_preprocess_grayscale(self):
        cfg = NNConfig(device="cpu")
        enc = TrainableVisualEncoder(config=cfg)
        img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        processed = enc.preprocess_image(img)
        assert processed.shape == (1, 3, 64, 64)

    def test_subpathway_dims(self):
        cfg = NNConfig(device="cpu")
        enc = TrainableVisualEncoder(config=cfg)
        assert VisualEncoder.SUBPATHWAYS["m_pathway"] == 96
        assert VisualEncoder.SUBPATHWAYS["p_pathway"] == 112
        assert VisualEncoder.SUBPATHWAYS["k_pathway"] == 48
        assert VisualEncoder.SUBPATHWAYS["it"] == 16
        assert VisualEncoder.SUBPATHWAYS["sc"] == 16
        assert VisualEncoder.SUBPATHWAYS["pulvinar"] == 12
        assert VisualEncoder.SUBPATHWAYS["binding"] == 8

    def test_subpathway_sum_to_308(self):
        total = sum(VisualEncoder.SUBPATHWAYS.values())
        assert total == 308

    def test_forward_deterministic(self):
        cfg = NNConfig(device="cpu")
        enc = TrainableVisualEncoder(config=cfg)
        enc.eval()
        img = np.random.rand(64, 64, 3).astype(np.float32)
        v1 = enc.encode(img)
        v2 = enc.encode(img)
        np.testing.assert_array_almost_equal(v1, v2, decimal=5)

    def test_train_step_runs(self):
        cfg = NNConfig(device="cpu")
        enc = TrainableVisualEncoder(config=cfg)
        enc.train()
        # Create batch of images
        images = np.random.rand(4, 3, 64, 64).astype(np.float32)
        losses = enc.train_step({"input": images})
        assert "loss" in losses
        assert losses["loss"] > 0
        assert enc.step_count == 1

    def test_get_subpathway_outputs(self):
        cfg = NNConfig(device="cpu")
        enc = TrainableVisualEncoder(config=cfg)
        enc.eval()
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        outputs = enc.get_subpathway_outputs(img)
        assert set(outputs.keys()) == set(VisualEncoder.SUBPATHWAYS.keys())
        assert outputs["m_pathway"].shape == (96,)
        assert outputs["p_pathway"].shape == (112,)
        assert outputs["k_pathway"].shape == (48,)
        assert outputs["it"].shape == (16,)
        assert outputs["sc"].shape == (16,)
        assert outputs["pulvinar"].shape == (12,)
        assert outputs["binding"].shape == (8,)


# ================================================================
# Test TrainableAudioEncoder
# ================================================================

@pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch not installed")
class TestTrainableAudioEncoder:
    """Tests for the CNN audio encoder on Mel spectrogram."""

    def test_init(self):
        cfg = NNConfig(device="cpu")
        enc = TrainableAudioEncoder(config=cfg)
        assert enc.name == "audio_encoder"
        assert enc.OUTPUT_DIM == 96
        assert enc._version == "7.0"
        assert enc.n_mels == 64
        assert enc.fixed_time_bins == 128
        assert enc.has_network is True

    def test_is_audio_encoder_subclass(self):
        cfg = NNConfig(device="cpu")
        enc = TrainableAudioEncoder(config=cfg)
        assert isinstance(enc, AudioEncoder)

    def test_encode_output_shape(self):
        cfg = NNConfig(device="cpu")
        enc = TrainableAudioEncoder(config=cfg)
        enc.eval()
        audio = np.random.randn(16000).astype(np.float32)  # 1 second
        vec = enc.encode(audio, sample_rate=16000)
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (96,)
        assert vec.dtype == np.float32

    def test_encode_short_audio(self):
        cfg = NNConfig(device="cpu")
        enc = TrainableAudioEncoder(config=cfg)
        enc.eval()
        audio = np.random.randn(4000).astype(np.float32)  # 0.25 seconds
        vec = enc.encode(audio, sample_rate=16000)
        assert vec.shape == (96,)
        assert np.all(np.isfinite(vec))

    def test_encode_batch(self):
        cfg = NNConfig(device="cpu")
        enc = TrainableAudioEncoder(config=cfg)
        enc.eval()
        audios = [
            np.random.randn(16000).astype(np.float32),
            np.random.randn(8000).astype(np.float32),
            np.random.randn(24000).astype(np.float32),
        ]
        batch = enc.encode_batch(audios)
        assert batch.shape == (3, 96)

    def test_submodule_dims(self):
        assert AudioEncoder.SUBMODULES["cochlear_nucleus"] == 32
        assert AudioEncoder.SUBMODULES["soc"] == 24
        assert AudioEncoder.SUBMODULES["ic"] == 24
        assert AudioEncoder.SUBMODULES["auditory_cortex"] == 16

    def test_submodule_sum_to_96(self):
        total = sum(AudioEncoder.SUBMODULES.values())
        assert total == 96

    def test_mel_computation(self):
        cfg = NNConfig(device="cpu")
        enc = TrainableAudioEncoder(config=cfg)
        # Generate a simple sine wave
        t = np.linspace(0, 1, 16000, endpoint=False, dtype=np.float32)
        sine = np.sin(2 * np.pi * 440 * t) * 0.5  # A4 note
        mel = enc.compute_mel(sine, 16000)
        assert mel.shape == (1, 1, 64, 128)
        assert mel.dtype == np.float32
        assert np.all(mel >= 0)
        assert np.any(mel > 0), "Mel should have non-zero energy for a sine wave"

    def test_mel_stereo_to_mono(self):
        cfg = NNConfig(device="cpu")
        enc = TrainableAudioEncoder(config=cfg)
        audio = np.random.randn(16000, 2).astype(np.float32)
        mel = enc.compute_mel(audio, 16000)
        assert mel.shape == (1, 1, 64, 128)

    def test_forward_deterministic(self):
        cfg = NNConfig(device="cpu")
        enc = TrainableAudioEncoder(config=cfg)
        enc.eval()
        audio = np.random.randn(8000).astype(np.float32)
        v1 = enc.encode(audio, sample_rate=16000)
        v2 = enc.encode(audio, sample_rate=16000)
        np.testing.assert_array_almost_equal(v1, v2, decimal=5)

    def test_train_step_runs(self):
        cfg = NNConfig(device="cpu")
        enc = TrainableAudioEncoder(config=cfg)
        enc.train()
        # Create batch of mel spectrograms
        mels = np.random.rand(4, 1, 64, 128).astype(np.float32)
        losses = enc.train_step({"input": mels})
        assert "loss" in losses
        assert losses["loss"] > 0
        assert enc.step_count == 1

    def test_different_sample_rates(self):
        cfg = NNConfig(device="cpu")
        enc = TrainableAudioEncoder(config=cfg)
        enc.eval()
        for sr in [8000, 16000, 22050]:
            audio = np.random.randn(sr).astype(np.float32)
            vec = enc.encode(audio, sample_rate=sr)
            assert vec.shape == (96,)
            assert np.all(np.isfinite(vec))

    def test_get_submodule_outputs(self):
        cfg = NNConfig(device="cpu")
        enc = TrainableAudioEncoder(config=cfg)
        enc.eval()
        audio = np.random.randn(16000).astype(np.float32)
        outputs = enc.get_submodule_outputs(audio, 16000)
        assert set(outputs.keys()) == set(AudioEncoder.SUBMODULES.keys())
        assert outputs["cochlear_nucleus"].shape == (32,)
        assert outputs["soc"].shape == (24,)
        assert outputs["ic"].shape == (24,)
        assert outputs["auditory_cortex"].shape == (16,)


# ================================================================
# Test Encoder Persistence
# ================================================================

@pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch not installed")
class TestEncoderPersistence:
    """Tests for save/load round-trip of all three encoders."""

    def test_text_save_load_roundtrip(self):
        cfg = NNConfig(device="cpu")
        enc = TrainableTextEncoder(config=cfg)
        enc.build_vocab(SAMPLE_CORPUS)
        enc.eval()
        v_before = enc.encode("测试持久化")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "text_enc.pt")
            enc.save(path)
            assert os.path.exists(path)

            enc2 = TrainableTextEncoder(config=cfg)
            enc2.build_vocab(SAMPLE_CORPUS)
            ok = enc2.load(path)
            assert ok is True
            enc2.eval()
            v_after = enc2.encode("测试持久化")
            np.testing.assert_array_almost_equal(v_before, v_after, decimal=5)

    def test_visual_save_load_roundtrip(self):
        cfg = NNConfig(device="cpu")
        enc = TrainableVisualEncoder(config=cfg)
        enc.eval()
        img = np.random.rand(64, 64, 3).astype(np.float32)
        v_before = enc.encode(img)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "vis_enc.pt")
            enc.save(path)

            enc2 = TrainableVisualEncoder(config=cfg)
            ok = enc2.load(path)
            assert ok is True
            enc2.eval()
            v_after = enc2.encode(img)
            np.testing.assert_array_almost_equal(v_before, v_after, decimal=5)

    def test_audio_save_load_roundtrip(self):
        cfg = NNConfig(device="cpu")
        enc = TrainableAudioEncoder(config=cfg)
        enc.eval()
        audio = np.random.randn(16000).astype(np.float32)
        v_before = enc.encode(audio)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "aud_enc.pt")
            enc.save(path)

            enc2 = TrainableAudioEncoder(config=cfg)
            ok = enc2.load(path)
            assert ok is True
            enc2.eval()
            v_after = enc2.encode(audio)
            np.testing.assert_array_almost_equal(v_before, v_after, decimal=5)


# ================================================================
# Integration Tests
# ================================================================

@pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch not installed")
class TestIntegration:
    """Integration tests for the full Phase B encoder pipeline."""

    def test_full_sensory_assembly(self):
        """Verify text(64)+visual(308)+audio(96)+pain(48) = 516."""
        cfg = NNConfig(device="cpu")

        text_enc = TrainableTextEncoder(config=cfg)
        text_enc.build_vocab(SAMPLE_CORPUS)
        text_enc.eval()

        vis_enc = TrainableVisualEncoder(config=cfg)
        vis_enc.eval()

        aud_enc = TrainableAudioEncoder(config=cfg)
        aud_enc.eval()

        # Encode
        text_vec = text_enc.encode("你好世界")
        vis_vec = vis_enc.encode(
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        )
        aud_vec = aud_enc.encode(
            np.random.randn(16000).astype(np.float32)
        )

        # Pain stays as zeros (hand-crafted system, Phase B preserved)
        pain_vec = np.zeros(48, dtype=np.float32)

        # Assemble
        sensory = np.concatenate([text_vec, vis_vec, aud_vec, pain_vec])
        assert sensory.shape == (516,)
        assert sensory.dtype == np.float32

    def test_train_all_three_no_crash(self):
        """All three encoders can train simultaneously without errors."""
        cfg = NNConfig(device="cpu")

        text_enc = TrainableTextEncoder(config=cfg)
        text_enc.build_vocab(SAMPLE_CORPUS)
        text_enc.train()

        vis_enc = TrainableVisualEncoder(config=cfg)
        vis_enc.train()

        aud_enc = TrainableAudioEncoder(config=cfg)
        aud_enc.train()

        # Train each with appropriate data
        batch_ids = np.array(
            [text_enc.tokenize(t) for t in SAMPLE_CORPUS[:4]],
            dtype=np.int64,
        )
        t_loss = text_enc.train_step({"input": batch_ids})

        vis_imgs = np.random.rand(4, 3, 64, 64).astype(np.float32)
        v_loss = vis_enc.train_step({"input": vis_imgs})

        aud_mels = np.random.rand(4, 1, 64, 128).astype(np.float32)
        a_loss = aud_enc.train_step({"input": aud_mels})

        assert t_loss["loss"] > 0
        assert v_loss["loss"] > 0
        assert a_loss["loss"] > 0

    def test_encoder_outputs_finite(self):
        """All encoder outputs should be finite and reasonably scaled."""
        cfg = NNConfig(device="cpu")

        text_enc = TrainableTextEncoder(config=cfg)
        text_enc.build_vocab(SAMPLE_CORPUS)
        text_enc.eval()

        vis_enc = TrainableVisualEncoder(config=cfg)
        vis_enc.eval()

        aud_enc = TrainableAudioEncoder(config=cfg)
        aud_enc.eval()

        t_vec = text_enc.encode("测试")
        v_vec = vis_enc.encode(
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        )
        a_vec = aud_enc.encode(np.random.randn(16000).astype(np.float32))

        for name, vec in [("text", t_vec), ("visual", v_vec), ("audio", a_vec)]:
            assert np.all(np.isfinite(vec)), f"{name} has NaN/Inf"
            assert np.abs(vec).max() < 100, (
                f"{name} has very large values: {np.abs(vec).max()}"
            )

    def test_encoder_versions_all_v7(self):
        """All encoders should report version 7.0."""
        cfg = NNConfig(device="cpu")
        for enc in [
            TrainableTextEncoder(config=cfg),
            TrainableVisualEncoder(config=cfg),
            TrainableAudioEncoder(config=cfg),
        ]:
            assert enc._version == "7.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
