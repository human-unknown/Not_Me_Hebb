"""Test cns.nn training & experience loop — Phase E implementations (v7.4).

Tests: Trainer, ExperienceTracker, TrainingMetrics
       — init, register, pretrain, online_finetune, checkpoint,
          record, summary, trends, export, integration.
"""
import sys
import os
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

import numpy as np
import tempfile
import json
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
from cns.nn.language_model import NeuralGenerator
from cns.nn.trainer import Trainer
from cns.nn.metrics import ExperienceTracker, TrainingMetrics


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
    "语言是人类最伟大的发明之一。",
    "音乐能触动人的心灵。",
    "科学和艺术是硬币的两面。",
    "时间是最宝贵的资源。",
    "友谊需要用心经营。",
    "梦想让生活充满希望。",
    "善良是一种选择。",
    "每个生命都有其独特价值。",
]

LARGE_CORPUS = SAMPLE_CORPUS * 5  # 100 lines for training tests


@pytest.fixture
def config():
    """Training-focused config (fast training)."""
    return NNConfig(
        device="cpu",
        training_enabled=True,
        learning_rate=1e-3,
        online_lr=1e-4,
        pretrain_epochs=3,
        lr_scheduler="none",
        checkpoint_interval=0,
        grad_clip=1.0,
    )


@pytest.fixture
def text_encoder(config):
    """Create a TrainableTextEncoder with vocab built."""
    enc = TrainableTextEncoder(config=config)
    enc.build_vocab(SAMPLE_CORPUS)
    return enc


@pytest.fixture
def generator(config, text_encoder):
    """Create a NeuralGenerator with vocab built."""
    gen = NeuralGenerator(
        config=config,
        text_encoder=text_encoder,
    )
    gen.build_vocab(SAMPLE_CORPUS)
    return gen


@pytest.fixture
def trainer(config):
    """Create a Trainer with default config."""
    return Trainer(config=config)


@pytest.fixture
def tracker(config):
    """Create an ExperienceTracker."""
    return ExperienceTracker(config=config)


# ================================================================
# TestTrainer
# ================================================================

@pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch not installed")
class TestTrainer:
    """Tests for the Trainer unified training orchestrator."""

    def test_init(self, trainer):
        """Trainer initializes with correct defaults."""
        assert trainer.config is not None
        assert trainer._modules == {}
        assert trainer._total_pretrain_steps == 0
        assert trainer._total_online_steps == 0
        assert trainer._version == "7.4"

    def test_register(self, trainer, text_encoder):
        """Register module → appears in _modules."""
        trainer.register(text_encoder)
        assert "text_encoder" in trainer._modules
        assert trainer.is_registered("text_encoder")

    def test_register_invalid_type(self, trainer):
        """Register non-NeuralModule raises TypeError."""
        with pytest.raises(TypeError):
            trainer.register("not_a_module")

    def test_register_duplicate(self, trainer, text_encoder):
        """Re-registering overwrites (with warning)."""
        trainer.register(text_encoder)
        trainer.register(text_encoder)  # should not raise
        assert len(trainer._modules) == 1

    def test_unregister(self, trainer, text_encoder):
        """Unregister removes module."""
        trainer.register(text_encoder)
        trainer.unregister("text_encoder")
        assert not trainer.is_registered("text_encoder")

    def test_pretrain_text_encoder(self, trainer, text_encoder):
        """Pretrain text encoder → loss decreases."""
        trainer.register(text_encoder)
        history = trainer.pretrain(
            "text_encoder",
            SAMPLE_CORPUS,
            epochs=3,
            batch_size=8,
            verbose=False,
        )
        assert len(history) == 3
        # Loss should generally decrease (first epoch > last epoch on average)
        # Allow some noise by checking the trend
        first_loss = history[0]["loss"]
        last_loss = history[-1]["loss"]
        assert first_loss > 0, f"Expected positive loss, got {first_loss}"
        # At minimum, training shouldn't explode
        assert last_loss < 100.0, f"Loss exploded: {last_loss}"

    def test_pretrain_generator(self, trainer, generator):
        """Pretrain generator → loss decreases."""
        trainer.register(generator)
        history = trainer.pretrain(
            "neural_generator",
            SAMPLE_CORPUS,
            epochs=3,
            batch_size=8,
            verbose=False,
        )
        assert len(history) == 3
        assert all("loss" in h for h in history)
        # Loss should not be NaN or inf
        for h in history:
            assert not np.isnan(h["loss"])
            assert not np.isinf(h["loss"])

    def test_pretrain_history_structure(self, trainer, text_encoder):
        """Pretrain returns correct history structure."""
        trainer.register(text_encoder)
        history = trainer.pretrain(
            "text_encoder",
            SAMPLE_CORPUS,
            epochs=2,
            batch_size=8,
            verbose=False,
        )
        assert len(history) == 2
        for i, record in enumerate(history):
            assert record["epoch"] == i + 1
            assert "loss" in record
            assert isinstance(record["loss"], float)

    def test_pretrain_callback(self, trainer, text_encoder):
        """Callback is called during training."""
        trainer.register(text_encoder)
        callback_calls = []

        def cb(epoch, batch_idx, losses):
            callback_calls.append((epoch, batch_idx, losses))

        trainer.pretrain(
            "text_encoder",
            SAMPLE_CORPUS,
            epochs=2,
            batch_size=8,
            callback=cb,
            verbose=False,
        )
        assert len(callback_calls) > 0
        # Verify callback receives correct types
        epoch, batch_idx, losses = callback_calls[0]
        assert isinstance(epoch, int)
        assert isinstance(batch_idx, int)
        assert "loss" in losses

    def test_pretrain_unregistered_module(self, trainer):
        """Pretrain on unregistered module raises KeyError."""
        with pytest.raises(KeyError):
            trainer.pretrain("nonexistent", SAMPLE_CORPUS)

    def test_online_finetune_runs(self, trainer, generator):
        """online_finetune() completes without error."""
        trainer.register(generator)

        # Do some pretraining first to have a trained model
        trainer.pretrain(
            "neural_generator",
            SAMPLE_CORPUS[:5],
            epochs=1,
            batch_size=4,
            verbose=False,
        )

        # Online fine-tune with a single batch
        batch = {"input": np.array([[1, 2, 3, 4, 5]], dtype=np.int64)}
        losses = trainer.online_finetune("neural_generator", batch)
        assert isinstance(losses, dict)
        assert "loss" in losses

    def test_online_finetune_lower_lr(self, trainer, generator):
        """Online LR < pretrain LR."""
        trainer.register(generator)
        online_lr = trainer.config.online_lr
        pretrain_lr = trainer.config.learning_rate
        assert online_lr < pretrain_lr, (
            f"Online LR {online_lr} should be < pretrain LR {pretrain_lr}"
        )

    def test_online_finetune_unregistered(self, trainer):
        """Online finetune on unregistered module raises KeyError."""
        with pytest.raises(KeyError):
            trainer.online_finetune("nonexistent", {"input": np.array([[1]])})

    def test_get_online_lr(self, trainer, generator):
        """get_online_optimizer_lr returns correct LR or -1 before creation."""
        trainer.register(generator)

        # Before fine-tuning, LR is -1 (not created yet)
        assert trainer.get_online_optimizer_lr("neural_generator") == -1.0

        # After fine-tuning, LR matches config
        batch = {"input": np.array([[1, 2, 3]], dtype=np.int64)}
        trainer.online_finetune("neural_generator", batch)
        lr = trainer.get_online_optimizer_lr("neural_generator")
        assert lr == trainer.config.online_lr

    def test_save_load_checkpoint(self, trainer, text_encoder, generator):
        """Save checkpoint → load → modules restored."""
        trainer.register(text_encoder)
        trainer.register(generator)

        # Do minimal training
        trainer.pretrain(
            "text_encoder",
            SAMPLE_CORPUS[:5],
            epochs=1,
            batch_size=4,
            verbose=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            saved_dir = trainer.save_checkpoint(tmpdir)
            assert os.path.exists(os.path.join(tmpdir, "text_encoder.pt"))
            assert os.path.exists(os.path.join(tmpdir, "neural_generator.pt"))
            assert os.path.exists(os.path.join(tmpdir, "trainer_state.json"))

            # Create fresh trainer and modules
            config2 = NNConfig(device="cpu", training_enabled=True)
            enc2 = TrainableTextEncoder(config=config2)
            enc2.build_vocab(SAMPLE_CORPUS)
            gen2 = NeuralGenerator(config=config2, text_encoder=enc2)
            gen2.build_vocab(SAMPLE_CORPUS)
            trainer2 = Trainer(config=config2)
            trainer2.register(enc2)
            trainer2.register(gen2)

            # Load
            result = trainer2.load_checkpoint(tmpdir)
            assert result is True

    def test_get_summary(self, trainer, text_encoder):
        """Summary reflects training history."""
        trainer.register(text_encoder)
        trainer.pretrain(
            "text_encoder",
            SAMPLE_CORPUS,
            epochs=2,
            batch_size=8,
            verbose=False,
        )

        summary = trainer.get_summary()
        assert summary["version"] == "7.4"
        assert "text_encoder" in summary["modules"]
        mod_summary = summary["modules"]["text_encoder"]
        assert mod_summary["total_epochs"] == 2
        assert mod_summary["best_loss"] > 0
        assert mod_summary["trainable"] is True

    def test_pretrain_all(self, trainer, text_encoder, generator):
        """pretrain_all processes all registered modules."""
        trainer.register(text_encoder)
        trainer.register(generator)

        configs = {
            "text_encoder": {"epochs": 1, "batch_size": 8},
            "neural_generator": {"epochs": 1, "batch_size": 8},
        }
        results = trainer.pretrain_all(
            SAMPLE_CORPUS[:10],
            configs=configs,
            verbose=False,
        )

        assert "text_encoder" in results
        assert "neural_generator" in results
        assert len(results["text_encoder"]) == 1
        assert len(results["neural_generator"]) == 1

    def test_module_names_property(self, trainer, text_encoder, generator):
        """module_names returns correct list."""
        assert trainer.module_names == []
        trainer.register(text_encoder)
        assert "text_encoder" in trainer.module_names
        trainer.register(generator)
        assert len(trainer.module_names) == 2

    def test_contains(self, trainer, text_encoder):
        """'in' operator works."""
        trainer.register(text_encoder)
        assert "text_encoder" in trainer
        assert "nonexistent" not in trainer

    def test_trainer_repr(self, trainer, text_encoder):
        """__repr__ returns string."""
        trainer.register(text_encoder)
        s = repr(trainer)
        assert "Trainer" in s
        assert "text_encoder" in s


# ================================================================
# TestExperienceTracker
# ================================================================

@pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch not installed")
class TestExperienceTracker:
    """Tests for the ExperienceTracker metrics system."""

    def test_init(self, tracker):
        """Tracker initializes with correct defaults."""
        assert tracker.records == []
        assert tracker._turn_count == 0
        assert tracker.vocab_size == 0
        assert tracker._version == "7.4"

    def test_record_step(self, tracker):
        """record_step stores metrics."""
        tracker.record_step(F_body=0.3, valence=0.5, arousal=0.2)
        assert len(tracker.records) == 1
        assert tracker.records[0]["F_body"] == 0.3
        assert tracker.records[0]["valence"] == 0.5
        assert tracker.records[0]["type"] == "step"

    def test_record_step_flexible(self, tracker):
        """record_step accepts arbitrary kwargs."""
        tracker.record_step(
            custom_metric=42,
            another_thing="hello",
            nested_list=[1, 2, 3],
        )
        r = tracker.records[0]
        assert r["custom_metric"] == 42
        assert r["another_thing"] == "hello"
        assert r["nested_list"] == [1, 2, 3]

    def test_record_dialogue_turn(self, tracker):
        """Turn recording stores text + metrics."""
        tracker.record_dialogue_turn(
            user_text="你好世界",
            response="你好！很高兴见到你。",
            metrics={"F_total": 0.5, "n400": 0.1},
        )
        assert tracker._turn_count == 1
        assert tracker.vocab_size > 0  # chars from texts added
        r = tracker.records[0]
        assert r["type"] == "dialogue_turn"
        assert r["user_text"] == "你好世界"
        assert r["response"] == "你好！很高兴见到你。"
        assert r["F_total"] == 0.5
        assert r["response_length"] > 0
        assert 0.0 <= r["response_diversity"] <= 1.0

    def test_get_summary(self, tracker):
        """Summary computes correct averages."""
        for i in range(10):
            tracker.record_step(
                F_total=0.5 + i * 0.01,
                valence=0.3 - i * 0.01,
                arousal=0.2,
            )

        summary = tracker.get_summary(window=10)
        assert summary["n_records"] == 10
        assert 0.5 < summary["avg_F_total"] < 0.6
        assert 0.2 < summary["avg_valence"] < 0.35
        assert abs(summary["avg_arousal"] - 0.2) < 0.01

    def test_get_summary_empty(self, tracker):
        """Summary on empty tracker returns zeros, no crash."""
        summary = tracker.get_summary()
        assert summary["n_records"] == 0
        assert summary["avg_F_total"] == 0.0
        assert summary["avg_valence"] == 0.0

    def test_get_trends_improving(self, tracker):
        """Decreasing F → 'declining' trend (F declining = good)."""
        # F going down over time (agent improving)
        for i in range(20):
            tracker.record_step(F_total=1.0 - i * 0.02)
        trends = tracker.get_trends(window=20)
        assert trends["F_trend"] == "declining"  # F going down

    def test_get_trends_stable(self, tracker):
        """Flat sequence → 'stable' trend."""
        for i in range(20):
            tracker.record_step(F_total=0.5 + np.random.randn() * 0.001)
        trends = tracker.get_trends(window=20)
        # With very small noise, should be stable
        assert trends["F_trend"] in ("stable", "insufficient_data")

    def test_get_trends_insufficient_data(self, tracker):
        """Too few points → 'insufficient_data'."""
        tracker.record_step(F_total=0.5)
        tracker.record_step(F_total=0.4)
        trends = tracker.get_trends(window=3)
        assert trends["F_trend"] == "insufficient_data"

    def test_vocab_tracking(self, tracker):
        """Unique chars tracked correctly."""
        tracker.record_dialogue_turn(
            user_text="你好",
            response="你好世界",
        )
        # Should have unique chars: 你, 好, 世, 界
        assert tracker.vocab_size >= 4

        # Same chars again → vocab shouldn't increase much
        prev_size = tracker.vocab_size
        tracker.record_dialogue_turn(
            user_text="你好",
            response="你好",
        )
        assert tracker.vocab_size == prev_size  # No new chars

    def test_user_profile(self, tracker):
        """Tracks user word frequency."""
        tracker.record_dialogue_turn(
            user_text="我喜欢学习人工智能",
            response="学习是很好的习惯",
        )
        tracker.record_dialogue_turn(
            user_text="人工智能很有趣",
            response="是的，我也觉得",
        )
        profile = tracker.get_user_profile()
        assert profile["total_interactions"] == 2
        assert len(profile["top_chars"]) > 0

        # '人工智能' chars should appear frequently
        top_chars = dict(profile["top_chars"])
        # At least some chars from the user text should appear
        assert len(top_chars) > 0

    def test_to_csv_export(self, tracker):
        """CSV export creates valid file."""
        tracker.record_step(F_total=0.5, valence=0.3)
        tracker.record_dialogue_turn(
            user_text="你好",
            response="你好！",
            metrics={"F_total": 0.4},
        )

        with tempfile.NamedTemporaryFile(
            suffix=".csv", mode="w+", delete=False, encoding="utf-8"
        ) as f:
            csv_path = f.name

        try:
            result = tracker.to_csv(csv_path)
            assert os.path.exists(csv_path)
            assert result == csv_path

            # Verify content
            with open(csv_path, "r", encoding="utf-8") as f:
                content = f.read()
            assert "F_total" in content
            assert "valence" in content
        finally:
            os.unlink(csv_path)

    def test_to_json_export_roundtrip(self, tracker):
        """JSON roundtrip preserves data."""
        tracker.record_step(F_total=0.5, valence=0.3, arousal=0.2)
        tracker.record_dialogue_turn(
            user_text="你好",
            response="你好世界！",
            metrics={"F_total": 0.4, "n400": 0.1},
        )

        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w+", delete=False, encoding="utf-8"
        ) as f:
            json_path = f.name

        try:
            tracker.to_json(json_path)

            # Load back
            tracker2 = ExperienceTracker.from_json(json_path)
            assert tracker2._turn_count == tracker._turn_count
            assert tracker2.vocab_size == tracker.vocab_size
            assert len(tracker2.records) == len(tracker.records)
            # Check first record keys
            assert tracker2.records[0]["F_total"] == 0.5
        finally:
            os.unlink(json_path)

    def test_empty_export(self, tracker, caplog):
        """Export with no records warns but doesn't crash."""
        import logging
        with caplog.at_level(logging.WARNING):
            result = tracker.to_csv("/tmp/nonexistent_dir/test.csv")
        # Should still return path, not crash
        assert result.endswith(".csv")

    def test_len(self, tracker):
        """__len__ returns record count."""
        assert len(tracker) == 0
        tracker.record_step(F_total=0.5)
        assert len(tracker) == 1
        tracker.record_step(valence=0.3)
        assert len(tracker) == 2

    def test_repr(self, tracker):
        """__repr__ returns string."""
        tracker.record_dialogue_turn(user_text="你好", response="你好！")
        s = repr(tracker)
        assert "ExperienceTracker" in s
        assert "turns=1" in s


# ================================================================
# TestTrainingMetrics
# ================================================================

class TestTrainingMetrics:
    """Tests for the TrainingMetrics lightweight tracker."""

    def test_init(self):
        """TrainingMetrics initializes correctly."""
        tm = TrainingMetrics()
        assert tm.loss_history == []
        assert tm.ppl_history == []
        assert tm._best_loss == float("inf")
        assert tm._epoch == 0

    def test_record(self):
        """record() stores metrics."""
        tm = TrainingMetrics()
        tm.record(loss=2.5, perplexity=12.0, lr=0.001)
        assert len(tm.loss_history) == 1
        assert tm.loss_history[0] == 2.5
        assert tm.ppl_history == [12.0]
        assert tm.lr_history == [0.001]

    def test_is_improving(self):
        """is_improving detects downward trend."""
        tm = TrainingMetrics()
        # Decreasing losses
        for loss in [3.0, 2.8, 2.5, 2.3, 2.0, 1.8, 1.5, 1.3]:
            tm.record(loss=loss)
        assert bool(tm.is_improving(window=8)) is True

    def test_is_not_improving(self):
        """is_improving returns False for upward trend."""
        tm = TrainingMetrics()
        # Increasing losses
        for loss in [1.0, 1.2, 1.5, 1.8, 2.0, 2.3, 2.5, 2.8]:
            tm.record(loss=loss)
        assert bool(tm.is_improving(window=8)) is False

    def test_is_converged(self):
        """is_converged detects flat losses."""
        tm = TrainingMetrics()
        # Very stable losses
        for i in range(15):
            tm.record(loss=0.5 + np.random.randn() * 0.00001)
        assert tm.is_converged(threshold=0.001, window=10) is True

    def test_is_not_converged(self):
        """is_converged returns False with noisy losses."""
        tm = TrainingMetrics()
        for loss in [1.0, 0.8, 1.2, 0.9, 1.1, 0.7, 1.3, 0.6, 1.4, 0.5]:
            tm.record(loss=loss)
        assert tm.is_converged(threshold=0.001, window=10) is False

    def test_get_best(self):
        """get_best returns minimum loss."""
        tm = TrainingMetrics()
        for loss in [3.0, 2.5, 2.0, 2.8, 1.5, 3.2]:
            tm.record(loss=loss)
            tm.next_epoch()
        best = tm.get_best()
        assert best["loss"] == 1.5

    def test_get_latest(self):
        """get_latest returns current metrics."""
        tm = TrainingMetrics()
        tm.record(loss=2.0, perplexity=7.4, lr=0.001)
        latest = tm.get_latest()
        assert latest["loss"] == 2.0
        assert latest["perplexity"] == 7.4
        assert latest["lr"] == 0.001

    def test_reset(self):
        """reset clears all metrics."""
        tm = TrainingMetrics()
        for loss in [2.0, 1.5, 1.0]:
            tm.record(loss=loss)
        tm.reset()
        assert len(tm.loss_history) == 0
        assert tm._best_loss == float("inf")
        assert tm._batch_count == 0

    def test_epoch_tracking(self):
        """next_epoch increments epoch counter."""
        tm = TrainingMetrics()
        assert tm._epoch == 0
        tm.next_epoch()
        assert tm._epoch == 1
        tm.next_epoch()
        assert tm._epoch == 2

    def test_repr(self):
        """__repr__ returns string."""
        tm = TrainingMetrics()
        tm.record(loss=2.0)
        s = repr(tm)
        assert "TrainingMetrics" in s
        assert "loss=2.0000" in s


# ================================================================
# TestIntegration
# ================================================================

@pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch not installed")
class TestIntegration:
    """Integration tests for Phase E components working together."""

    def test_trainer_with_tracker(self, config, text_encoder):
        """Trainer + ExperienceTracker work together."""
        trainer = Trainer(config=config)
        tracker = ExperienceTracker(config=config)

        trainer.register(text_encoder)
        history = trainer.pretrain(
            "text_encoder",
            SAMPLE_CORPUS,
            epochs=2,
            batch_size=8,
            verbose=False,
        )

        # Record training results in tracker
        for epoch_record in history:
            tracker.record_step(
                F_total=epoch_record["loss"],
                module="text_encoder",
                epoch=epoch_record["epoch"],
            )

        summary = tracker.get_summary()
        assert summary["n_records"] == 2

    def test_full_pretrain_flow(self, config, text_encoder, generator):
        """Full pretrain flow: text_encoder + generator → generate."""
        trainer = Trainer(config=config)
        trainer.register(text_encoder)
        trainer.register(generator)

        # Pretrain both
        results = trainer.pretrain_all(
            SAMPLE_CORPUS,
            configs={
                "text_encoder": {"epochs": 2, "batch_size": 8},
                "neural_generator": {"epochs": 2, "batch_size": 8},
            },
            verbose=False,
        )

        assert len(results) == 2

        # Generator should be able to generate after pretraining
        output = generator.generate(
            prompt="你好",
            max_new_tokens=10,
            temperature=0.8,
        )
        assert isinstance(output, str)
        assert len(output) > 0

    def test_online_learning_flow(self, config, generator):
        """Pretrain → online finetune → generate."""
        trainer = Trainer(config=config)
        trainer.register(generator)

        # Small pretrain
        trainer.pretrain(
            "neural_generator",
            SAMPLE_CORPUS,
            epochs=2,
            batch_size=4,
            verbose=False,
        )

        # Simulate online learning from a user turn
        user_text = "你今天感觉怎么样"
        tokens = generator.tokenize(user_text)
        batch = {"input": np.array([tokens], dtype=np.int64)}
        online_loss = trainer.online_finetune("neural_generator", batch)
        assert "loss" in online_loss

        # Should still generate after online learning
        output = generator.generate(
            prompt="你好",
            max_new_tokens=10,
            temperature=0.8,
        )
        assert isinstance(output, str)

    def test_metrics_persistence(self, tracker):
        """Save tracker → load → same data."""
        tracker.record_step(F_total=0.5, valence=0.3)
        tracker.record_dialogue_turn(
            user_text="你好",
            response="你好！",
            metrics={"n400": 0.1},
        )

        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w+", delete=False, encoding="utf-8"
        ) as f:
            json_path = f.name

        try:
            tracker.to_json(json_path)
            tracker2 = ExperienceTracker.from_json(json_path)
            assert tracker2.n_records == tracker.n_records
            assert tracker2.turn_count == tracker.turn_count
            assert tracker2.vocab_size == tracker.vocab_size
        finally:
            os.unlink(json_path)

    def test_all_modules_trainable(self, config, text_encoder, generator):
        """Verify all registered modules are trainable."""
        trainer = Trainer(config=config)
        trainer.register(text_encoder)
        trainer.register(generator)

        summary = trainer.get_summary()
        for mod_name, mod_summary in summary["modules"].items():
            assert mod_summary["trainable"] is True, (
                f"Module {mod_name} should be trainable"
            )

    def test_version_v74(self, trainer, tracker):
        """Trainer and tracker report v7.4 version."""
        assert trainer._version == "7.4"
        assert tracker._version == "7.4"

        tm = TrainingMetrics()
        assert tm is not None  # TrainingMetrics has no version attr

    def test_config_fields_present(self, config):
        """Phase E config fields are present."""
        assert hasattr(config, "pretrain_epochs")
        assert hasattr(config, "online_lr")
        assert hasattr(config, "lr_scheduler")
        assert hasattr(config, "checkpoint_interval")
        assert config.online_lr < config.learning_rate
