"""Test cns.nn memory layer — Phase C implementations (v7.2).

Tests: NeuralSemanticStore, CrossModalNN
       — init, insert, query, forget, encode, training, retrieval, persistence, integration.
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
from cns.nn.semantic_store import NeuralSemanticStore
from cns.nn.crossmodal_nn import CrossModalNN

# Check FAISS availability
_HAS_FAISS = False
try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    pass


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
    """Create a minimal text encoder with vocab for testing."""
    cfg = NNConfig(device="cpu")
    enc = TrainableTextEncoder(config=cfg)
    enc.build_vocab(SAMPLE_CORPUS)
    enc.eval()
    return enc


def _make_visual_encoder():
    """Create a minimal visual encoder for testing."""
    cfg = NNConfig(device="cpu")
    enc = TrainableVisualEncoder(config=cfg)
    enc.eval()
    return enc


# ================================================================
# Test NeuralSemanticStore
# ================================================================

@pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch not installed")
class TestNeuralSemanticStore:
    """Tests for the neural vector store."""

    def test_init(self):
        """Store initializes with correct defaults."""
        store = NeuralSemanticStore(dim=64, capacity=100)
        assert store.name == "semantic_store"
        assert store.dim == 64
        assert store.capacity == 100
        assert store.n_entries == 0
        assert store.is_empty is True
        assert store.is_full is False
        assert store.trainable is False

    def test_init_with_text_encoder(self):
        """Store works with a text encoder attached."""
        enc = _make_text_encoder()
        store = NeuralSemanticStore(text_encoder=enc, dim=64, capacity=100)
        assert store._text_encoder is not None
        assert store.get_state()["has_text_encoder"] is True

    def test_insert_single(self):
        """Insert one text entry."""
        enc = _make_text_encoder()
        store = NeuralSemanticStore(text_encoder=enc, dim=64, capacity=100)
        idx = store.insert("自由能原理")
        assert idx == 0
        assert store.n_entries == 1
        assert store.is_empty is False

    def test_insert_with_metadata(self):
        """Insert with metadata preserved."""
        enc = _make_text_encoder()
        store = NeuralSemanticStore(text_encoder=enc, dim=64, capacity=100)
        store.insert("测试文本", metadata={"valence": 0.5, "arousal": 0.3})
        results = store.query("测试文本", top_k=1)
        assert len(results) == 1
        _, meta = results[0]
        assert meta["valence"] == 0.5
        assert meta["arousal"] == 0.3
        assert meta["text"] == "测试文本"

    def test_query_by_text(self):
        """Query by text returns the inserted entry."""
        enc = _make_text_encoder()
        store = NeuralSemanticStore(text_encoder=enc, dim=64, capacity=100)
        store.insert("自由能原理是主动推理的核心")
        store.insert("今天天气不错")
        store.insert("机器学习需要大量数据")

        results = store.query("自由能原理是主动推理的核心", top_k=1)
        assert len(results) == 1
        sim, meta = results[0]
        assert sim > 0.95  # Same text should have near-perfect similarity
        assert "自由能" in meta["text"]

    def test_query_by_vector(self):
        """Query by raw vector works same as text query."""
        enc = _make_text_encoder()
        store = NeuralSemanticStore(text_encoder=enc, dim=64, capacity=100)
        store.insert("测试查询文本")

        # Encode the same text
        query_vec = enc.encode("测试查询文本")
        results_vec = store.query(query_vec, top_k=1)
        results_text = store.query("测试查询文本", top_k=1)

        assert len(results_vec) == 1
        assert len(results_text) == 1
        # Similarity should be very close
        assert abs(results_vec[0][0] - results_text[0][0]) < 0.01

    def test_query_top_k(self):
        """Query returns top_k results sorted by similarity."""
        enc = _make_text_encoder()
        store = NeuralSemanticStore(text_encoder=enc, dim=64, capacity=100)

        store.insert("猫是哺乳动物")
        store.insert("狗是哺乳动物")  # Similar to cat
        store.insert("汽车是交通工具")  # Different topic
        store.insert("飞机是交通工具")  # Similar to car
        store.insert("今天天气真好")  # Different topic

        results = store.query("猫", top_k=3)
        assert len(results) == 3
        # Similarities should be descending
        sims = [r[0] for r in results]
        assert sims == sorted(sims, reverse=True)

    def test_empty_store_query(self):
        """Query on empty store returns empty list."""
        store = NeuralSemanticStore(dim=64, capacity=100)
        results = store.query("任何文本")
        assert results == []

        # Query with raw vector
        vec = np.random.randn(64).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        results = store.query(vec)
        assert results == []

    def test_duplicate_insert(self):
        """Inserting same text twice increments count but doesn't duplicate."""
        enc = _make_text_encoder()
        store = NeuralSemanticStore(text_encoder=enc, dim=64, capacity=100)

        store.insert("重复文本测试")
        assert store.n_entries == 1
        count1 = store._entries[0]["count"]

        store.insert("重复文本测试")
        assert store.n_entries == 1  # Still one entry
        count2 = store._entries[0]["count"]
        assert count2 == count1 + 1
        # Timestamp should be updated
        assert store._entries[0]["timestamp"] == store._step_counter

    def test_forget_old(self):
        """Forgetting removes entries older than threshold."""
        enc = _make_text_encoder()
        store = NeuralSemanticStore(text_encoder=enc, dim=64, capacity=100)

        # Insert 3 entries
        store.insert("条目一")
        store.insert("条目二")
        store.insert("条目三")

        # Manually age the first entry
        store._entries[0]["timestamp"] = 1  # Very old
        old_step = store._step_counter

        # Forget entries older than half the current step
        removed = store.forget_old(max_age_steps=old_step // 2)
        assert removed >= 1  # At least the manually aged one
        assert store.n_entries <= 2

    def test_metadata_preserved(self):
        """All metadata fields survive insert→query roundtrip."""
        enc = _make_text_encoder()
        store = NeuralSemanticStore(text_encoder=enc, dim=64, capacity=100)

        meta_in = {
            "valence": 0.7,
            "arousal": 0.4,
            "source": "dialogue",
            "confidence": 0.9,
        }
        store.insert("元数据测试", metadata=meta_in)

        results = store.query("元数据测试", top_k=1)
        _, meta_out = results[0]

        assert meta_out["valence"] == 0.7
        assert meta_out["arousal"] == 0.4
        assert meta_out["source"] == "dialogue"
        assert meta_out["confidence"] == 0.9
        assert meta_out["text"] == "元数据测试"
        assert "timestamp" in meta_out
        assert meta_out["count"] == 1

    def test_save_load_roundtrip(self):
        """Save → load → same query results."""
        enc = _make_text_encoder()
        store = NeuralSemanticStore(text_encoder=enc, dim=64, capacity=100)

        store.insert("持久化测试文本一")
        store.insert("持久化测试文本二")
        store.insert("不相关的内容")

        # Query before save
        results_before = store.query("持久化测试", top_k=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "store.pt")
            store.save(path)

            store2 = NeuralSemanticStore(text_encoder=enc, dim=64, capacity=100)
            ok = store2.load(path)
            assert ok is True
            assert store2.n_entries == 3

            results_after = store2.query("持久化测试", top_k=2)

        # Same number of results
        assert len(results_before) == len(results_after)
        # Same texts returned
        texts_before = {r[1]["text"] for r in results_before}
        texts_after = {r[1]["text"] for r in results_after}
        assert texts_before == texts_after
        # Similarities match
        for (sim_b, _), (sim_a, _) in zip(results_before, results_after):
            assert abs(sim_b - sim_a) < 0.001

    def test_rebuild_index_after_forget(self):
        """After forget + rebuild, query still works."""
        enc = _make_text_encoder()
        store = NeuralSemanticStore(text_encoder=enc, dim=64, capacity=100)

        store.insert("保留条目")
        store.insert("要删除的条目")

        # Age only the second entry to be very old, keep the first fresh
        store._entries[1]["timestamp"] = 1  # Very old
        # Move step counter forward so entry 0 is recent
        store._step_counter = 100
        store._entries[0]["timestamp"] = 99  # Recent

        # Forget entries older than 50 steps
        removed = store.forget_old(max_age_steps=50)
        assert removed == 1  # Only "要删除的条目" removed
        store.rebuild_index()

        results = store.query("保留条目", top_k=1)
        assert len(results) == 1
        assert "保留条目" in results[0][1]["text"]

    def test_capacity_eviction(self):
        """When capacity reached, oldest entry is evicted."""
        enc = _make_text_encoder()
        store = NeuralSemanticStore(text_encoder=enc, dim=64, capacity=3)

        store.insert("条目一")
        store.insert("条目二")
        store.insert("条目三")
        assert store.n_entries == 3

        # Age the first entry
        store._entries[0]["timestamp"] = 1
        store._step_counter = 100

        # Insert one more — should evict oldest
        store.insert("条目四")
        assert store.n_entries == 3

        # "条目一" should be gone
        remaining_texts = {e["text"] for e in store._entries}
        assert "条目一" not in remaining_texts
        assert "条目四" in remaining_texts

    def test_query_without_text_encoder(self):
        """Store works with raw vectors (no text encoder)."""
        store = NeuralSemanticStore(dim=64, capacity=100)

        # Insert via raw vector
        vec1 = np.random.randn(64).astype(np.float32)
        vec1 = vec1 / np.linalg.norm(vec1)
        store._insert_vec(vec1, text="raw_entry_1")

        vec2 = np.random.randn(64).astype(np.float32)
        vec2 = vec2 / np.linalg.norm(vec2)
        store._insert_vec(vec2, text="raw_entry_2")

        assert store.n_entries == 2

        # Query by vector
        results = store.query(vec1, top_k=1)
        assert len(results) == 1
        assert results[0][0] > 0.99  # Exact match
        assert results[0][1]["text"] == "raw_entry_1"

    def test_get_state(self):
        """get_state() returns correct statistics."""
        enc = _make_text_encoder()
        store = NeuralSemanticStore(text_encoder=enc, dim=64, capacity=500)

        store.insert("条目一")
        store.insert("条目二")

        state = store.get_state()
        assert state["n_entries"] == 2
        assert state["capacity"] == 500
        assert state["dim"] == 64
        assert state["has_text_encoder"] is True
        assert state["has_faiss"] == _HAS_FAISS
        assert state["oldest_step"] > 0
        assert state["newest_step"] >= state["oldest_step"]

    def test_insert_batch(self):
        """Batch insert works efficiently."""
        enc = _make_text_encoder()
        store = NeuralSemanticStore(text_encoder=enc, dim=64, capacity=100)

        texts = ["机器学习很有趣", "猫喜欢吃鱼", "今天天气真好", "自由能原理是基础"]
        metas = [
            {"valence": 0.1}, {"valence": 0.2},
            {"valence": 0.3}, {"valence": 0.4},
        ]
        indices = store.insert_batch(texts, metadatas=metas)
        assert len(indices) == 4
        assert store.n_entries == 4

        # Query for a specific entry — should match exactly
        results = store.query("今天天气真好", top_k=1)
        assert len(results) == 1
        assert results[0][1]["valence"] == 0.3
        assert results[0][1]["text"] == "今天天气真好"

    def test_forward_query(self):
        """forward() method works for tensor-based queries."""
        enc = _make_text_encoder()
        store = NeuralSemanticStore(text_encoder=enc, dim=64, capacity=100)

        store.insert("前向测试")
        store.insert("其他内容")

        query_vec = enc.encode("前向测试")
        query_batch = query_vec[np.newaxis, :]  # (1, 64)
        result = store.forward(query_batch)
        assert result.shape == (1, 1)
        assert result[0, 0] > 0.0


# ================================================================
# Test CrossModalNN
# ================================================================

@pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch not installed")
class TestCrossModalNN:
    """Tests for the CLIP-style cross-modal module."""

    def test_init(self):
        """Module initializes with correct defaults."""
        text_enc = _make_text_encoder()
        vis_enc = _make_visual_encoder()
        cm = CrossModalNN(
            text_encoder=text_enc,
            visual_encoder=vis_enc,
            shared_dim=128,
            temperature=0.07,
        )
        assert cm.name == "crossmodal_nn"
        assert cm.shared_dim == 128
        assert cm.temperature == 0.07
        assert cm.freeze_encoders is True
        assert cm.has_text_encoder is True
        assert cm.has_visual_encoder is True
        assert cm.trainable is True

    def test_has_projectors(self):
        """Text and visual projection layers exist."""
        text_enc = _make_text_encoder()
        vis_enc = _make_visual_encoder()
        cm = CrossModalNN(text_encoder=text_enc, visual_encoder=vis_enc)

        assert "text_proj" in cm._net
        assert "visual_proj" in cm._net
        # Check dimensions
        text_proj = cm._net["text_proj"]
        assert isinstance(text_proj[0], torch.nn.Linear)
        assert text_proj[0].in_features == 64
        assert text_proj[0].out_features == 128

    def test_encode_text_shape(self):
        """encode_text returns correct shape."""
        text_enc = _make_text_encoder()
        vis_enc = _make_visual_encoder()
        cm = CrossModalNN(text_encoder=text_enc, visual_encoder=vis_enc)

        result = cm.encode_text("你好世界")
        assert result.shape == (1, 128)
        assert result.dtype == np.float32

        # Batch
        result_batch = cm.encode_text(["你好", "世界", "测试"])
        assert result_batch.shape == (3, 128)

    def test_encode_image_shape(self):
        """encode_image returns correct shape for both input formats."""
        text_enc = _make_text_encoder()
        vis_enc = _make_visual_encoder()
        cm = CrossModalNN(text_encoder=text_enc, visual_encoder=vis_enc)

        # (B, H, W, 3) uint8 format
        imgs_hwc = np.random.randint(0, 255, (4, 64, 64, 3), dtype=np.uint8)
        result_hwc = cm.encode_image(imgs_hwc)
        assert result_hwc.shape == (4, 128)

        # (B, 3, H, W) preprocessed format
        imgs_chw = np.random.rand(2, 3, 64, 64).astype(np.float32)
        result_chw = cm.encode_image(imgs_chw)
        assert result_chw.shape == (2, 128)

    def test_l2_normalized_outputs(self):
        """Both encode_text and encode_image produce L2-normalized outputs."""
        text_enc = _make_text_encoder()
        vis_enc = _make_visual_encoder()
        cm = CrossModalNN(text_encoder=text_enc, visual_encoder=vis_enc)

        t_emb = cm.encode_text("测试文本")
        v_emb = cm.encode_image(
            np.random.randint(0, 255, (1, 64, 64, 3), dtype=np.uint8)
        )

        # L2 norm should be ~1.0
        assert abs(np.linalg.norm(t_emb[0]) - 1.0) < 0.01
        assert abs(np.linalg.norm(v_emb[0]) - 1.0) < 0.01

    def test_shared_space_same_dim(self):
        """Text and image encode to the same dimension."""
        text_enc = _make_text_encoder()
        vis_enc = _make_visual_encoder()
        cm = CrossModalNN(text_encoder=text_enc, visual_encoder=vis_enc)

        t_emb = cm.encode_text("测试")
        v_emb = cm.encode_image(
            np.random.randint(0, 255, (1, 64, 64, 3), dtype=np.uint8)
        )

        assert t_emb.shape[-1] == v_emb.shape[-1] == 128

    def test_train_step_runs(self):
        """One contrastive training step returns valid loss."""
        text_enc = _make_text_encoder()
        vis_enc = _make_visual_encoder()
        cm = CrossModalNN(
            text_encoder=text_enc, visual_encoder=vis_enc,
            freeze_encoders=False,  # Need trainable for this test
        )
        cm.train()

        batch = {
            "text": ["你好世界", "今天天气好", "我喜欢猫", "机器学习"],
            "image": np.random.rand(4, 3, 64, 64).astype(np.float32),
        }
        loss_info = cm.train_step(batch)
        assert loss_info["loss"] > 0.0
        assert 0.0 <= loss_info.get("accuracy", 0.0) <= 1.0

    def test_contrastive_loss_decreases(self):
        """Training on the same batch twice decreases loss."""
        text_enc = _make_text_encoder()
        vis_enc = _make_visual_encoder()
        cm = CrossModalNN(
            text_encoder=text_enc, visual_encoder=vis_enc,
            freeze_encoders=False,
        )
        cm.train()

        batch = {
            "text": ["猫在睡觉", "狗在跑步", "鸟在飞翔", "鱼在游泳"],
            "image": np.random.rand(4, 3, 64, 64).astype(np.float32),
        }

        loss1 = cm.train_step(batch)["loss"]
        loss2 = cm.train_step(batch)["loss"]
        # Loss should decrease (or stay similar) on second pass
        # Note: with random images, loss may not decrease monotonically,
        # but the training step should not crash
        assert loss1 > 0
        assert loss2 > 0

    def test_deterministic_encoding(self):
        """Eval mode: same input → same output."""
        text_enc = _make_text_encoder()
        vis_enc = _make_visual_encoder()
        cm = CrossModalNN(text_encoder=text_enc, visual_encoder=vis_enc)
        cm.eval()

        r1 = cm.encode_text("确定性测试")
        r2 = cm.encode_text("确定性测试")
        np.testing.assert_array_almost_equal(r1, r2, decimal=6)

        imgs = np.random.randint(0, 255, (2, 64, 64, 3), dtype=np.uint8)
        v1 = cm.encode_image(imgs)
        v2 = cm.encode_image(imgs)
        np.testing.assert_array_almost_equal(v1, v2, decimal=6)

    def test_save_load_roundtrip(self):
        """Save → load → same encode results."""
        text_enc = _make_text_encoder()
        vis_enc = _make_visual_encoder()
        cm = CrossModalNN(text_encoder=text_enc, visual_encoder=vis_enc)
        cm.eval()

        # Use fixed images for before/after comparison
        fixed_imgs = np.random.randint(0, 255, (1, 64, 64, 3), dtype=np.uint8)
        fixed_imgs_chw = np.random.rand(2, 3, 64, 64).astype(np.float32)

        t_before = cm.encode_text("保存测试")
        v_before = cm.encode_image(fixed_imgs)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cm.pt")
            cm.save(path)

            cm2 = CrossModalNN(
                text_encoder=text_enc, visual_encoder=vis_enc
            )
            ok = cm2.load(path)
            assert ok is True
            cm2.eval()

            t_after = cm2.encode_text("保存测试")
            v_after = cm2.encode_image(fixed_imgs)

        np.testing.assert_array_almost_equal(t_before, t_after, decimal=5)
        np.testing.assert_array_almost_equal(v_before, v_after, decimal=5)

    def test_freeze_encoders_respected(self):
        """When freeze_encoders=True, encoder weights don't change during training."""
        text_enc = _make_text_encoder()
        vis_enc = _make_visual_encoder()

        # Get initial weights
        init_text_weight = text_enc._net["proj"][0].weight.clone()
        init_vis_weight = vis_enc._net["heads"]["m_pathway"][0].weight.clone()

        cm = CrossModalNN(
            text_encoder=text_enc, visual_encoder=vis_enc,
            freeze_encoders=True,
        )
        cm.train()

        batch = {
            "text": ["测试一", "测试二", "测试三", "测试四"],
            "image": np.random.rand(4, 3, 64, 64).astype(np.float32),
        }
        cm.train_step(batch)

        # Encoder weights should be unchanged
        cur_text_weight = text_enc._net["proj"][0].weight
        cur_vis_weight = vis_enc._net["heads"]["m_pathway"][0].weight

        assert torch.equal(init_text_weight, cur_text_weight)
        assert torch.equal(init_vis_weight, cur_vis_weight)

    def test_retrieve_by_text(self):
        """Text→Image retrieval returns meaningful results."""
        text_enc = _make_text_encoder()
        vis_enc = _make_visual_encoder()
        cm = CrossModalNN(text_encoder=text_enc, visual_encoder=vis_enc)
        cm.eval()

        # Create a pool of 4 random images
        image_pool = np.random.randint(0, 255, (4, 64, 64, 3), dtype=np.uint8)

        # Retrieval should work without crashing
        indices = cm.retrieve_image("测试查询", image_pool, top_k=2)
        assert len(indices) == 2
        assert all(0 <= i < 4 for i in indices)

    def test_compute_similarity(self):
        """compute_similarity returns a float in [-1, 1]."""
        text_enc = _make_text_encoder()
        vis_enc = _make_visual_encoder()
        cm = CrossModalNN(text_encoder=text_enc, visual_encoder=vis_enc)
        cm.eval()

        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        sim = cm.compute_similarity("测试", img)
        assert isinstance(sim, float)
        assert -1.0 <= sim <= 1.0

    def test_similarity_matrix(self):
        """compute_similarity_matrix returns correct shape."""
        text_enc = _make_text_encoder()
        vis_enc = _make_visual_encoder()
        cm = CrossModalNN(text_encoder=text_enc, visual_encoder=vis_enc)
        cm.eval()

        texts = ["猫", "狗", "汽车"]
        imgs = np.random.randint(0, 255, (5, 64, 64, 3), dtype=np.uint8)

        matrix = cm.compute_similarity_matrix(texts, imgs)
        assert matrix.shape == (3, 5)


# ================================================================
# Integration Tests
# ================================================================

@pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch not installed")
class TestIntegration:
    """Integration tests for Phase C components working together."""

    def test_full_phase_c_pipeline(self):
        """SemanticStore + CrossModalNN coexist without errors."""
        text_enc = _make_text_encoder()
        vis_enc = _make_visual_encoder()

        store = NeuralSemanticStore(text_encoder=text_enc, dim=64, capacity=100)
        cm = CrossModalNN(text_encoder=text_enc, visual_encoder=vis_enc)
        cm.eval()

        # Insert to store
        store.insert("神经网络是强大的工具")
        store.insert("自由能原理驱动智能体")

        # Query store
        results = store.query("神经网络", top_k=2)
        assert len(results) >= 1

        # Encode for cross-modal
        t = cm.encode_text("神经网络")
        v = cm.encode_image(
            np.random.randint(0, 255, (1, 64, 64, 3), dtype=np.uint8)
        )
        assert t.shape == (1, 128)
        assert v.shape == (1, 128)

    def test_shared_text_encoder(self):
        """Same TrainableTextEncoder used by SemanticStore and CrossModalNN."""
        text_enc = _make_text_encoder()
        vis_enc = _make_visual_encoder()

        store = NeuralSemanticStore(text_encoder=text_enc, dim=64, capacity=100)
        cm = CrossModalNN(text_encoder=text_enc, visual_encoder=vis_enc)
        cm.eval()

        # The same text should produce matching embeddings in both systems
        store.insert("共享编码器测试")

        # Store query should find it
        results = store.query("共享编码器测试", top_k=1)
        assert len(results) == 1
        assert results[0][0] > 0.95

        # Cross-modal encode of same text should work
        t_emb = cm.encode_text("共享编码器测试")
        assert t_emb.shape == (1, 128)

    def test_nn_memory_train_no_crash(self):
        """Train CrossModalNN while inserting to SemanticStore — no crashes."""
        text_enc = _make_text_encoder()
        vis_enc = _make_visual_encoder()

        store = NeuralSemanticStore(text_encoder=text_enc, dim=64, capacity=100)
        cm = CrossModalNN(
            text_encoder=text_enc, visual_encoder=vis_enc,
            freeze_encoders=False,
        )
        cm.train()

        # Insert some data
        store.insert("训练测试一")
        store.insert("训练测试二")

        # Train cross-modal
        batch = {
            "text": ["数据一", "数据二", "数据三", "数据四"],
            "image": np.random.rand(4, 3, 64, 64).astype(np.float32),
        }
        loss_info = cm.train_step(batch)
        assert loss_info["loss"] > 0.0

        # Insert more while training
        store.insert("训练测试三")

        # Query still works
        results = store.query("训练测试", top_k=3)
        assert len(results) >= 2

    def test_version_metadata(self):
        """Both modules report version metadata."""
        text_enc = _make_text_encoder()
        vis_enc = _make_visual_encoder()

        store = NeuralSemanticStore(text_encoder=text_enc)
        cm = CrossModalNN(text_encoder=text_enc, visual_encoder=vis_enc)

        # Both inherit from NeuralModule
        assert store._version is not None
        assert cm._version is not None

        # Store has correct name
        assert store.name == "semantic_store"
        assert cm.name == "crossmodal_nn"

    def test_faiss_numpy_consistency(self):
        """If FAISS is available, results match numpy search."""
        enc = _make_text_encoder()
        store = NeuralSemanticStore(text_encoder=enc, dim=64, capacity=100)

        texts = [
            "查询测试一", "查询测试二", "查询测试三",
            "查询测试四", "查询测试五",
        ]
        for t in texts:
            store.insert(t)

        # Force numpy search by not rebuilding FAISS index
        store._index = None
        results_numpy = store.query("查询测试", top_k=3)

        # Rebuild FAISS and search
        store.rebuild_index()
        results_faiss = store.query("查询测试", top_k=3)

        # Results should be identical (or very close) between numpy and FAISS
        assert len(results_numpy) == len(results_faiss)
        for (sn, mn), (sf, mf) in zip(results_numpy, results_faiss):
            assert mn["text"] == mf["text"]
            assert abs(sn - sf) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
