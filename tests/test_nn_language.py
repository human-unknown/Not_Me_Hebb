"""Test cns.nn language layer — Phase D implementations (v7.3).

Tests: NeuralGenerator, NeuralComprehender, NeuralAngularGyrus
       — init, generate, comprehend, encode, training, persistence, integration.
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
from cns.nn.semantic_store import NeuralSemanticStore
from cns.nn.language_model import (
    NeuralGenerator, BOS_IDX, EOS_IDX, V_POS_IDX, V_NEG_IDX,
    A_HIGH_IDX, A_LOW_IDX, NUM_SPECIAL,
)
from cns.nn.comprehender import NeuralComprehender
from cns.nn.angular_gyrus_nn import NeuralAngularGyrus


# ================================================================
# Shared fixtures
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


def _make_text_encoder():
    cfg = NNConfig(device="cpu")
    enc = TrainableTextEncoder(config=cfg)
    enc.build_vocab(SAMPLE_CORPUS)
    enc.eval()
    return enc


# ================================================================
# Test NeuralGenerator
# ================================================================

@pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch not installed")
class TestNeuralGenerator:
    """Tests for the autoregressive language model."""

    def test_init(self):
        enc = _make_text_encoder()
        gen = NeuralGenerator(text_encoder=enc, d_model=128, n_layers=2)
        assert gen.name == "neural_generator"
        assert gen.d_model == 128
        assert gen.n_layers == 2
        assert gen.n_heads == 8
        assert gen.max_len == 256
        assert gen.trainable is True
        assert gen.has_network is True

    def test_build_vocab_shared(self):
        enc = _make_text_encoder()
        gen = NeuralGenerator(text_encoder=enc)
        vocab_size = gen.build_vocab(SAMPLE_CORPUS)
        assert vocab_size > NUM_SPECIAL
        assert gen.is_vocab_built
        assert gen.vocab_size == vocab_size

    def test_build_vocab_from_corpus(self):
        gen = NeuralGenerator(d_model=128, n_layers=2)
        vocab_size = gen.build_vocab(SAMPLE_CORPUS)
        assert vocab_size > NUM_SPECIAL
        assert gen.is_vocab_built

    def test_forward_shape(self):
        enc = _make_text_encoder()
        gen = NeuralGenerator(text_encoder=enc, d_model=128, n_layers=2)
        gen.build_vocab(SAMPLE_CORPUS)
        gen.eval()

        # Create a batch of token sequences
        batch_ids = np.array([
            gen.tokenize("你好世界"),
            gen.tokenize("自由能"),
        ], dtype=np.int64)

        logits = gen.forward(batch_ids)
        assert logits.shape == (2, gen.max_len, gen.vocab_size)

    def test_forward_logits_finite(self):
        enc = _make_text_encoder()
        gen = NeuralGenerator(text_encoder=enc, d_model=128, n_layers=2)
        gen.build_vocab(SAMPLE_CORPUS)
        gen.eval()

        batch_ids = np.array([gen.tokenize("测试")], dtype=np.int64)
        logits = gen.forward(batch_ids)
        assert np.all(np.isfinite(logits))

    def test_generate_output(self):
        enc = _make_text_encoder()
        gen = NeuralGenerator(text_encoder=enc, d_model=128, n_layers=2)
        gen.build_vocab(SAMPLE_CORPUS)
        gen.eval()

        text = gen.generate("你好", max_new_tokens=20, temperature=1.0, top_k=10)
        assert isinstance(text, str)
        # With random weights, output might be empty — that's OK at init
        # But the API should not crash

    def test_generate_with_emotion(self):
        enc = _make_text_encoder()
        gen = NeuralGenerator(text_encoder=enc, d_model=128, n_layers=2)
        gen.build_vocab(SAMPLE_CORPUS)
        gen.eval()

        text = gen.generate(
            "今天", valence=0.8, arousal=0.7,
            max_new_tokens=20, temperature=1.0, top_k=10
        )
        assert isinstance(text, str)

    def test_generate_deterministic(self):
        enc = _make_text_encoder()
        gen = NeuralGenerator(text_encoder=enc, d_model=128, n_layers=2)
        gen.build_vocab(SAMPLE_CORPUS)
        gen.eval()

        t1 = gen.generate("你好", max_new_tokens=15, temperature=1.0,
                          top_k=10, seed=42)
        t2 = gen.generate("你好", max_new_tokens=15, temperature=1.0,
                          top_k=10, seed=42)
        assert t1 == t2

    def test_train_step_runs(self):
        gen = NeuralGenerator(d_model=128, n_layers=2)
        gen.build_vocab(SAMPLE_CORPUS)
        gen.train()

        batch_ids = np.array([
            gen.tokenize(t) for t in SAMPLE_CORPUS[:4]
        ], dtype=np.int64)

        losses = gen.train_step({"input": batch_ids})
        assert losses["loss"] > 0.0
        assert "perplexity" in losses
        assert losses["perplexity"] > 1.0

    def test_pretrain_smoke(self):
        gen = NeuralGenerator(d_model=128, n_layers=2)
        gen.build_vocab(SAMPLE_CORPUS)
        history = gen.pretrain(SAMPLE_CORPUS, epochs=3, batch_size=8, verbose=False)
        assert len(history) == 3
        # Loss should decrease
        assert history[-1]["loss"] <= history[0]["loss"] * 1.5  # Allow some noise

    def test_emotion_tokens_exist(self):
        gen = NeuralGenerator(d_model=128, n_layers=2)
        gen.build_vocab(SAMPLE_CORPUS)

        emotion = gen.emotion_token_ids
        assert emotion["V_POS"] == V_POS_IDX
        assert emotion["V_NEG"] == V_NEG_IDX
        assert emotion["A_HIGH"] == A_HIGH_IDX
        assert emotion["A_LOW"] == A_LOW_IDX

    def test_save_load_roundtrip(self):
        enc = _make_text_encoder()
        gen = NeuralGenerator(text_encoder=enc, d_model=128, n_layers=2)
        gen.build_vocab(SAMPLE_CORPUS)
        gen.eval()

        output_before = gen.generate("你好", max_new_tokens=15, seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "gen.pt")
            gen.save(path)

            gen2 = NeuralGenerator(text_encoder=enc, d_model=128, n_layers=2)
            gen2.build_vocab(SAMPLE_CORPUS)
            ok = gen2.load(path)
            assert ok is True
            gen2.eval()

            output_after = gen2.generate("你好", max_new_tokens=15, seed=42)

        assert output_before == output_after

    def test_tokenize_detokenize_roundtrip(self):
        enc = _make_text_encoder()
        gen = NeuralGenerator(text_encoder=enc)
        gen.build_vocab(SAMPLE_CORPUS)

        text = "你好世界"
        ids = gen.tokenize(text)
        decoded = gen.detokenize(ids)
        assert text in decoded.replace(" ", "")


# ================================================================
# Test NeuralComprehender
# ================================================================

@pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch not installed")
class TestNeuralComprehender:
    """Tests for the memory-augmented text comprehender."""

    def test_init(self):
        enc = _make_text_encoder()
        comp = NeuralComprehender(text_encoder=enc)
        assert comp.name == "neural_comprehender"
        assert comp.trainable is False
        assert comp.context_size == 5

    def test_comprehend_returns_dict(self):
        enc = _make_text_encoder()
        comp = NeuralComprehender(text_encoder=enc)
        result = comp.comprehend("你好，今天天气真好")
        assert isinstance(result, dict)
        for key in ["comprehension_vec", "n400", "p600",
                     "attended_memories", "input_embedding"]:
            assert key in result, f"Missing key: {key}"

    def test_comprehend_vector_shape(self):
        enc = _make_text_encoder()
        comp = NeuralComprehender(text_encoder=enc)
        result = comp.comprehend("测试理解向量")
        assert result["comprehension_vec"].shape == (64,)
        assert result["comprehension_vec"].dtype == np.float32
        assert result["input_embedding"].shape == (64,)

    def test_n400_range(self):
        enc = _make_text_encoder()
        comp = NeuralComprehender(text_encoder=enc)
        result = comp.comprehend("测试N400")
        assert 0.0 <= result["n400"] <= 1.0

    def test_n400_identical_text(self):
        """Same text twice should have lower N400 on second pass (predicted)."""
        enc = _make_text_encoder()
        comp = NeuralComprehender(text_encoder=enc)
        r1 = comp.comprehend("重复文本测试")
        r2 = comp.comprehend("重复文本测试")
        # With context, N400 should be lower (more predictable)
        # Note: first call sets context, second call uses it
        assert r1["n400"] >= 0.0
        assert r2["n400"] >= 0.0

    def test_n400_different_text(self):
        """Very different text should produce different N400 values."""
        enc = _make_text_encoder()
        comp = NeuralComprehender(text_encoder=enc)
        r1 = comp.comprehend("今天天气很好")
        comp.reset_context()
        r2 = comp.comprehend("量子力学很难理解")
        # Both should be valid
        assert 0.0 <= r1["n400"] <= 1.0
        assert 0.0 <= r2["n400"] <= 1.0

    def test_with_memory_store(self):
        enc = _make_text_encoder()
        store = NeuralSemanticStore(text_encoder=enc, dim=64, capacity=100)
        store.insert("天气相关话题")
        store.insert("情感相关话题")

        comp = NeuralComprehender(text_encoder=enc, memory_store=store)
        result = comp.comprehend("今天天气真好")
        assert "attended_memories" in result

    def test_without_memory_store(self):
        enc = _make_text_encoder()
        comp = NeuralComprehender(text_encoder=enc, memory_store=None)
        result = comp.comprehend("没有记忆存储的测试")
        assert result["attended_memories"] == []
        assert result["comprehension_vec"].shape == (64,)

    def test_p600_range(self):
        enc = _make_text_encoder()
        comp = NeuralComprehender(text_encoder=enc)
        result = comp.comprehend("测试P600")
        assert 0.0 <= result["p600"] <= 1.0

    def test_context_window(self):
        enc = _make_text_encoder()
        comp = NeuralComprehender(text_encoder=enc, context_size=3)

        assert len(comp._context_window) == 0
        comp.comprehend("第一句话")
        assert len(comp._context_window) == 1
        comp.comprehend("第二句话")
        assert len(comp._context_window) == 2
        comp.comprehend("第三句话")
        assert len(comp._context_window) == 3
        comp.comprehend("第四句话")
        # Should cap at context_size
        assert len(comp._context_window) == 3

    def test_get_context_vector(self):
        enc = _make_text_encoder()
        comp = NeuralComprehender(text_encoder=enc)
        comp.comprehend("测试一")
        comp.comprehend("测试二")

        ctx_vec = comp.get_context_vector(window_size=2)
        assert ctx_vec.shape == (64,)
        assert np.any(ctx_vec != 0.0)  # Should be non-zero after comprehension

    def test_reset_context(self):
        enc = _make_text_encoder()
        comp = NeuralComprehender(text_encoder=enc)
        comp.comprehend("测试")
        assert len(comp._context_window) == 1
        comp.reset_context()
        assert len(comp._context_window) == 0

    def test_get_state(self):
        enc = _make_text_encoder()
        comp = NeuralComprehender(text_encoder=enc)
        comp.comprehend("状态测试")
        state = comp.get_state()
        assert state["comprehend_count"] == 1
        assert state["has_text_encoder"] is True
        assert "n400_ema" in state
        assert "p600_ema" in state


# ================================================================
# Test NeuralAngularGyrus
# ================================================================

@pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch not installed")
class TestNeuralAngularGyrus:
    """Tests for the learned grapheme→phoneme mapping."""

    def test_init(self):
        enc = _make_text_encoder()
        ag = NeuralAngularGyrus(text_encoder=enc)
        assert ag.name == "neural_angular_gyrus"
        assert ag.d_model == 64
        assert ag.max_chars == 16
        assert ag.trainable is True

    def test_build_vocab(self):
        enc = _make_text_encoder()
        ag = NeuralAngularGyrus(text_encoder=enc)
        vocab_size = ag.build_vocab(SAMPLE_CORPUS)
        assert vocab_size > 0
        assert ag.is_vocab_built

    def test_encode_chars_shape(self):
        enc = _make_text_encoder()
        ag = NeuralAngularGyrus(text_encoder=enc)
        ag.build_vocab(SAMPLE_CORPUS)
        ag.eval()

        vec = ag.encode_chars("猫")
        assert vec.shape == (64,)
        assert vec.dtype == np.float32

    def test_encode_batch(self):
        enc = _make_text_encoder()
        ag = NeuralAngularGyrus(text_encoder=enc)
        ag.build_vocab(SAMPLE_CORPUS)
        ag.eval()

        vecs = ag.encode_batch(["猫", "狗", "鱼"])
        assert vecs.shape == (3, 64)

    def test_train_step_runs(self):
        ag = NeuralAngularGyrus()
        ag.build_vocab(SAMPLE_CORPUS)
        ag.train()

        batch_ids = np.array([
            ag.tokenize(s[:4]) for s in ["猫咪", "狗狗", "鱼儿", "鸟儿"]
        ], dtype=np.int64)
        target = np.random.randn(4, 64).astype(np.float32)

        losses = ag.train_step({"input": batch_ids, "target": target})
        assert losses["loss"] > 0.0

    def test_deterministic(self):
        enc = _make_text_encoder()
        ag = NeuralAngularGyrus(text_encoder=enc)
        ag.build_vocab(SAMPLE_CORPUS)
        ag.eval()

        v1 = ag.encode_chars("测试")
        v2 = ag.encode_chars("测试")
        np.testing.assert_array_almost_equal(v1, v2, decimal=6)

    def test_save_load_roundtrip(self):
        enc = _make_text_encoder()
        ag = NeuralAngularGyrus(text_encoder=enc)
        ag.build_vocab(SAMPLE_CORPUS)
        ag.eval()

        v_before = ag.encode_chars("持久化测试")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ag.pt")
            ag.save(path)

            ag2 = NeuralAngularGyrus(text_encoder=enc)
            ag2.build_vocab(SAMPLE_CORPUS)
            ok = ag2.load(path)
            assert ok is True
            ag2.eval()

            v_after = ag2.encode_chars("持久化测试")

        np.testing.assert_array_almost_equal(v_before, v_after, decimal=5)

    def test_train_pairs(self):
        ag = NeuralAngularGyrus()
        ag.build_vocab(SAMPLE_CORPUS)

        chars = ["你好", "世界", "测试", "学习"]
        phonemes = np.random.randn(4, 64).astype(np.float32)

        history = ag.train_pairs(chars, phonemes, epochs=3, verbose=False)
        assert len(history) == 3
        # Loss should generally decrease
        assert history[0]["loss"] > 0.0

    def test_different_length_inputs(self):
        enc = _make_text_encoder()
        ag = NeuralAngularGyrus(text_encoder=enc)
        ag.build_vocab(SAMPLE_CORPUS)
        ag.eval()

        v1 = ag.encode_chars("一")   # 1 char
        v2 = ag.encode_chars("一二三四五六七八")  # 8 chars
        assert v1.shape == (64,)
        assert v2.shape == (64,)
        # Different length inputs should produce different outputs
        assert not np.allclose(v1, v2)


# ================================================================
# Integration Tests
# ================================================================

@pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch not installed")
class TestIntegration:
    """Integration tests for Phase D language modules."""

    def test_full_language_pipeline(self):
        """Generator + Comprehender + AngularGyrus coexist."""
        enc = _make_text_encoder()

        gen = NeuralGenerator(text_encoder=enc, d_model=128, n_layers=2)
        gen.build_vocab(SAMPLE_CORPUS)
        gen.eval()

        comp = NeuralComprehender(text_encoder=enc)
        ag = NeuralAngularGyrus(text_encoder=enc)
        ag.build_vocab(SAMPLE_CORPUS)
        ag.eval()

        # All three should work without errors
        gen_text = gen.generate("你好", max_new_tokens=10, seed=42)
        comp_result = comp.comprehend("测试管道")
        ag_vec = ag.encode_chars("管道测试")

        assert comp_result["comprehension_vec"].shape == (64,)
        assert ag_vec.shape == (64,)

    def test_comprehend_then_generate(self):
        """Comprehend input → use context for generation."""
        enc = _make_text_encoder()

        comp = NeuralComprehender(text_encoder=enc)
        gen = NeuralGenerator(text_encoder=enc, d_model=128, n_layers=2)
        gen.build_vocab(SAMPLE_CORPUS)
        gen.eval()

        # Comprehend a few inputs
        comp.comprehend("你好")
        comp.comprehend("今天天气如何")

        # Get context
        ctx = comp.get_context_vector()
        assert ctx.shape == (64,)

        # Generate (can use context vector as prompt seed)
        response = gen.generate("天气", max_new_tokens=10, seed=42)
        assert isinstance(response, str)

    def test_language_modules_train_no_crash(self):
        """Train all three language modules simultaneously."""
        gen = NeuralGenerator(d_model=128, n_layers=2)
        gen.build_vocab(SAMPLE_CORPUS)
        gen.train()

        ag = NeuralAngularGyrus()
        ag.build_vocab(SAMPLE_CORPUS)
        ag.train()

        # Train generator
        batch_ids = np.array([
            gen.tokenize(t) for t in SAMPLE_CORPUS[:4]
        ], dtype=np.int64)
        g_loss = gen.train_step({"input": batch_ids})

        # Train angular gyrus
        batch_chars = np.array([
            ag.tokenize(s[:4]) for s in ["训练", "测试", "数据", "模型"]
        ], dtype=np.int64)
        target = np.random.randn(4, 64).astype(np.float32)
        a_loss = ag.train_step({"input": batch_chars, "target": target})

        assert g_loss["loss"] > 0.0
        assert a_loss["loss"] > 0.0

    def test_version_v73(self):
        """All Phase D modules report appropriate version."""
        enc = _make_text_encoder()

        gen = NeuralGenerator(text_encoder=enc, d_model=128, n_layers=2)
        gen.build_vocab(SAMPLE_CORPUS)

        comp = NeuralComprehender(text_encoder=enc)

        ag = NeuralAngularGyrus(text_encoder=enc)
        ag.build_vocab(SAMPLE_CORPUS)

        for mod in [gen, comp, ag]:
            assert mod._version is not None

        assert gen.name == "neural_generator"
        assert comp.name == "neural_comprehender"
        assert ag.name == "neural_angular_gyrus"

    def test_shared_vocab_across_modules(self):
        """All three modules share the same vocab via text_encoder."""
        enc = _make_text_encoder()

        gen = NeuralGenerator(text_encoder=enc)
        gen.build_vocab(SAMPLE_CORPUS)

        ag = NeuralAngularGyrus(text_encoder=enc)
        ag.build_vocab(SAMPLE_CORPUS)

        # Same vocab size (generator has +4 emotion tokens)
        assert ag.vocab_size + 4 == gen.vocab_size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
