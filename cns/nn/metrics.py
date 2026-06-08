"""
metrics.py — Observable Metrics & Experience Tracking (v7.4 Phase E)

Tracks per-step agent metrics and per-dialogue-turn experience data
for the training-and-experience feedback loop.

Classes:
  - ExperienceTracker: per-step/dialogue-turn metric recording,
    trend analysis, CSV/JSON export, personalization tracking
  - TrainingMetrics: lightweight training progress tracker
    (loss history, perplexity, LR tracking)

Design:
  - Flexible record_step(**kwargs) — no fixed schema
  - Trend analysis via simple linear regression on recent window
  - Char-level vocabulary tracking (suited for Chinese)
  - CSV export compatible with existing tools/telemetry.py
  - Zero modifications to Agent, FEP, body, or sleep systems
"""

import os
import json
import csv
import logging
from typing import Optional, Dict, List, Any, Set
import numpy as np

from cns.nn.config import NNConfig, DEFAULT_NN_CONFIG

logger = logging.getLogger(__name__)


class ExperienceTracker:
    """Track observable metrics during agent operation.

    Records per-step metrics and per-dialogue-turn experience data.
    Provides trend analysis, summary statistics, and CSV/JSON export.

    Tracks:
      - Free energy components (F_body, F_social, F_cognitive, F_accuracy, F_total)
      - Valence and arousal
      - Vocabulary growth (unique chars, unique words)
      - Response metrics (length, diversity = unique/total chars ratio)
      - Training progress (loss, perplexity from Trainer)
      - User-specific word frequency (personalization)

    Usage:
        tracker = ExperienceTracker()

        # During agent.step()
        tracker.record_step(
            F_body=0.3, F_social=0.1, valence=0.5, arousal=0.2,
        )

        # After dialogue turn
        tracker.record_dialogue_turn(
            user_text="你好", response="你好！今天怎么样？",
            metrics={"F_total": 0.5, "n400": 0.1}
        )

        # Export
        tracker.to_csv("telemetry/experience.csv")

        # Summary
        print(tracker.get_summary(window=10))
    """

    def __init__(self, config: Optional[NNConfig] = None):
        """Initialize ExperienceTracker.

        Args:
            config: NNConfig (None = use DEFAULT_NN_CONFIG)
        """
        self.config = config or DEFAULT_NN_CONFIG
        self.records: List[Dict[str, Any]] = []
        self._vocabulary: Set[str] = set()
        self._user_word_freq: Dict[str, int] = {}
        self._turn_count: int = 0
        self._version: str = "7.4"

    # ================================================================
    # Recording
    # ================================================================

    def record_step(self, **metrics):
        """Record one agent step's metrics.

        Flexible kwargs — different modules may report different metrics.
        Common keys: F_body, F_social, F_cognitive, F_accuracy, F_total,
        valence, arousal, step_count, vocab_size.

        Args:
            **metrics: Arbitrary metric key-value pairs (must be JSON-serializable)
        """
        record = {
            "type": "step",
            "index": len(self.records),
            **metrics,
        }
        self.records.append(record)

    def record_dialogue_turn(
        self,
        user_text: str,
        response: str,
        metrics: Optional[Dict[str, Any]] = None,
    ):
        """Record a full dialogue turn.

        Args:
            user_text: The user's input text
            response: The agent's response text
            metrics: Optional dict of additional metrics
                     (e.g., F_total, n400, p600, generation_loss)
        """
        self._turn_count += 1

        # Update vocabulary
        for char in user_text + response:
            if char and char.strip():
                self._vocabulary.add(char)

        # Update user word frequency (personalization)
        for char in user_text:
            if char and char.strip():
                self._user_word_freq[char] = (
                    self._user_word_freq.get(char, 0) + 1
                )

        # Response metrics
        resp_chars = [c for c in response if c and c.strip()]
        resp_unique = set(resp_chars)
        response_length = len(resp_chars)
        response_diversity = (
            len(resp_unique) / max(1, response_length)
        )

        record = {
            "type": "dialogue_turn",
            "index": len(self.records),
            "turn": self._turn_count,
            "user_text": user_text,
            "response": response,
            "response_length": response_length,
            "response_diversity": round(response_diversity, 4),
            "vocab_size": self.vocab_size,
            "user_vocab_size": len(self._user_word_freq),
            **(metrics or {}),
        }
        self.records.append(record)

    # ================================================================
    # Summary & Trends
    # ================================================================

    def get_summary(self, window: int = 10) -> Dict[str, Any]:
        """Compute summary statistics over the recent N records.

        Args:
            window: Number of recent records to consider (0 = all)

        Returns:
            Dict with avg F, valence, arousal, vocab size, etc.
        """
        records = self.records[-window:] if window > 0 else self.records

        if not records:
            return {
                "n_records": 0,
                "n_turns": 0,
                "avg_F_total": 0.0,
                "avg_valence": 0.0,
                "avg_arousal": 0.0,
                "avg_response_length": 0,
                "avg_response_diversity": 0.0,
                "vocab_size": self.vocab_size,
                "user_vocab_size": len(self._user_word_freq),
            }

        # Collect numeric metrics
        f_totals = []
        valences = []
        arousals = []
        resp_lengths = []
        resp_diversities = []

        for r in records:
            if "F_total" in r:
                f_totals.append(r["F_total"])
            if "valence" in r:
                valences.append(r["valence"])
            if "arousal" in r:
                arousals.append(r["arousal"])
            if "response_length" in r:
                resp_lengths.append(r["response_length"])
            if "response_diversity" in r:
                resp_diversities.append(r["response_diversity"])

        dialogue_records = [r for r in records if r.get("type") == "dialogue_turn"]

        return {
            "n_records": len(records),
            "n_turns": len(dialogue_records),
            "avg_F_total": round(float(np.mean(f_totals)), 4) if f_totals else 0.0,
            "avg_valence": round(float(np.mean(valences)), 4) if valences else 0.0,
            "avg_arousal": round(float(np.mean(arousals)), 4) if arousals else 0.0,
            "avg_response_length": round(float(np.mean(resp_lengths)), 1) if resp_lengths else 0,
            "avg_response_diversity": round(float(np.mean(resp_diversities)), 4) if resp_diversities else 0.0,
            "vocab_size": self.vocab_size,
            "user_vocab_size": len(self._user_word_freq),
        }

    def get_trends(self, window: int = 50) -> Dict[str, str]:
        """Analyze metric trends over the recent window.

        Uses simple linear regression slope to determine direction.
        Requires at least 5 data points for classification.

        Args:
            window: Number of recent records to analyze

        Returns:
            Dict with trend classifications:
            F_trend, valence_trend, arousal_trend: 'improving'|'stable'|'declining'|'insufficient_data'
            vocab_growth_rate: new chars per dialogue turn (float)
        """
        records = self.records[-window:] if window > 0 else self.records

        trends = {
            "F_trend": "insufficient_data",
            "valence_trend": "insufficient_data",
            "arousal_trend": "insufficient_data",
            "vocab_growth_rate": 0.0,
        }

        # Extract sequences
        def extract_sequence(key: str) -> List[float]:
            return [r[key] for r in records if key in r]

        for metric, trend_key in [
            ("F_total", "F_trend"),
            ("valence", "valence_trend"),
            ("arousal", "arousal_trend"),
        ]:
            seq = extract_sequence(metric)
            if len(seq) >= 5:
                trends[trend_key] = self._classify_trend(seq)

        # Vocab growth rate
        dialogue_turns = [r for r in records if r.get("type") == "dialogue_turn"]
        if len(dialogue_turns) >= 2:
            first_vocab = dialogue_turns[0].get("vocab_size", 0)
            last_vocab = dialogue_turns[-1].get("vocab_size", 0)
            n_turns = len(dialogue_turns)
            trends["vocab_growth_rate"] = round(
                (last_vocab - first_vocab) / max(1, n_turns), 2
            )

        return trends

    def _classify_trend(self, sequence: List[float]) -> str:
        """Classify trend direction using linear regression slope.

        Args:
            sequence: List of numeric values

        Returns:
            'improving', 'stable', or 'declining'
        """
        if len(sequence) < 5:
            return "insufficient_data"

        x = np.arange(len(sequence), dtype=np.float64)
        y = np.array(sequence, dtype=np.float64)
        slope = np.polyfit(x, y, 1)[0]

        # Normalize slope by mean absolute value
        mean_abs = np.mean(np.abs(y))
        if mean_abs < 1e-8:
            return "stable"
        normalized_slope = slope / mean_abs

        if normalized_slope < -0.01:
            return "declining"
        elif normalized_slope > 0.01:
            return "improving"
        else:
            return "stable"

    # ================================================================
    # Personalization
    # ================================================================

    def get_user_profile(self, top_n: int = 20) -> Dict[str, Any]:
        """Get personalization profile based on user interaction history.

        Args:
            top_n: Number of top items to include

        Returns:
            {
                "total_interactions": int,
                "top_chars": [(char, count), ...],
                "unique_chars_used": int,
                "unique_agent_chars": int,
            }
        """
        sorted_chars = sorted(
            self._user_word_freq.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return {
            "total_interactions": self._turn_count,
            "top_chars": sorted_chars[:top_n],
            "unique_chars_used": len(self._user_word_freq),
            "unique_agent_chars": self.vocab_size - len(self._user_word_freq),
        }

    # ================================================================
    # Export
    # ================================================================

    def to_csv(self, path: str) -> str:
        """Export records as CSV file.

        Compatible with existing tools/telemetry.py CSV schema.
        Missing keys in any record are filled with empty string.

        Args:
            path: Output CSV file path

        Returns:
            Path to the saved file
        """
        if not self.records:
            logger.warning("No records to export")
            return path

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        # Collect all possible column names
        all_keys = set()
        for r in self.records:
            all_keys.update(r.keys())
        columns = sorted(all_keys)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for r in self.records:
                # Clean non-serializable values
                row = {}
                for k in columns:
                    v = r.get(k, "")
                    if isinstance(v, (list, dict)):
                        row[k] = json.dumps(v, ensure_ascii=False)
                    elif isinstance(v, float):
                        row[k] = round(v, 6)
                    else:
                        row[k] = v
                writer.writerow(row)

        logger.info(f"[ExperienceTracker] Exported {len(self.records)} records to {path}")
        return path

    def to_json(self, path: str) -> str:
        """Export all records + vocabulary as JSON file.

        Args:
            path: Output JSON file path

        Returns:
            Path to the saved file
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        data = {
            "version": self._version,
            "turn_count": self._turn_count,
            "vocab_size": self.vocab_size,
            "user_vocab_size": len(self._user_word_freq),
            "user_profile": self.get_user_profile(),
            "records": self.records,
            "vocabulary": sorted(list(self._vocabulary)),
            "user_word_freq": dict(
                sorted(
                    self._user_word_freq.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
            ),
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"[ExperienceTracker] Exported JSON to {path}")
        return path

    @classmethod
    def from_json(cls, path: str, config: Optional[NNConfig] = None) -> "ExperienceTracker":
        """Load ExperienceTracker from JSON export.

        Args:
            path: JSON file path
            config: Optional NNConfig

        Returns:
            Reconstructed ExperienceTracker
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tracker = cls(config=config)
        tracker._turn_count = data.get("turn_count", 0)
        tracker.records = data.get("records", [])
        tracker._vocabulary = set(data.get("vocabulary", []))
        tracker._user_word_freq = dict(data.get("user_word_freq", {}))
        tracker._version = data.get("version", "7.4")

        return tracker

    # ================================================================
    # Properties
    # ================================================================

    @property
    def vocab_size(self) -> int:
        """Number of unique characters seen (agent + user)."""
        return len(self._vocabulary)

    @property
    def turn_count(self) -> int:
        """Number of dialogue turns recorded."""
        return self._turn_count

    @property
    def n_records(self) -> int:
        """Total number of records (steps + dialogue turns)."""
        return len(self.records)

    # ================================================================
    # Dunder
    # ================================================================

    def __repr__(self) -> str:
        return (
            f"ExperienceTracker(turns={self._turn_count}, "
            f"records={len(self.records)}, "
            f"vocab={self.vocab_size})"
        )

    def __len__(self) -> int:
        return len(self.records)


# ================================================================
# TrainingMetrics — Lightweight training progress tracker
# ================================================================

class TrainingMetrics:
    """Lightweight training progress tracker for use by Trainer.

    Tracks per-batch metrics during pretraining and online fine-tuning.
    Provides smoothing, best-value tracking, and convergence detection.

    Usage:
        tm = TrainingMetrics()
        tm.record(loss=2.3, perplexity=10.0, lr=0.001)
        tm.record(loss=2.1, perplexity=8.2, lr=0.001)
        print(tm.is_improving())  # True
        print(tm.get_best())      # {'loss': 2.1, 'perplexity': 8.2}
    """

    def __init__(self, smoothing: float = 0.1):
        """Initialize TrainingMetrics.

        Args:
            smoothing: EMA smoothing factor for smoothed loss (0-1, lower = smoother)
        """
        self.loss_history: List[float] = []
        self.ppl_history: List[float] = []
        self.lr_history: List[float] = []
        self._smoothing = smoothing
        self._smoothed_loss: Optional[float] = None
        self._best_loss: float = float("inf")
        self._best_epoch: int = 0
        self._epoch: int = 0
        self._batch_count: int = 0

    def record(
        self,
        loss: float,
        perplexity: Optional[float] = None,
        lr: Optional[float] = None,
    ):
        """Record one training step's metrics.

        Args:
            loss: Training loss value
            perplexity: Optional perplexity
            lr: Optional current learning rate
        """
        self.loss_history.append(loss)
        self._batch_count += 1

        if perplexity is not None:
            self.ppl_history.append(perplexity)
        if lr is not None:
            self.lr_history.append(lr)

        # EMA smoothed loss
        if self._smoothed_loss is None:
            self._smoothed_loss = loss
        else:
            self._smoothed_loss = (
                self._smoothing * loss
                + (1 - self._smoothing) * self._smoothed_loss
            )

        # Track best
        if loss < self._best_loss:
            self._best_loss = loss
            self._best_epoch = self._epoch

    def next_epoch(self):
        """Mark end of epoch."""
        self._epoch += 1

    def is_improving(self, window: int = 5) -> bool:
        """Check if loss is trending down over recent batches.

        Args:
            window: Number of recent loss values to consider

        Returns:
            True if loss is trending downward
        """
        if len(self.loss_history) < max(window, 3):
            return False
        recent = self.loss_history[-window:]
        if len(recent) < 2:
            return False
        # Simple check: is the last loss lower than the average of the first half?
        mid = len(recent) // 2
        first_half_avg = np.mean(recent[:mid])
        last_half_avg = np.mean(recent[mid:])
        return last_half_avg < first_half_avg

    def is_converged(self, threshold: float = 1e-4, window: int = 10) -> bool:
        """Check if training has converged.

        Args:
            threshold: Maximum loss change to consider converged
            window: Number of recent loss values to check

        Returns:
            True if loss change is below threshold
        """
        if len(self.loss_history) < window:
            return False
        recent = self.loss_history[-window:]
        loss_range = max(recent) - min(recent)
        return loss_range < threshold

    def get_best(self) -> Dict[str, Any]:
        """Return best (lowest) loss and the epoch it occurred.

        Returns:
            {'loss': float, 'epoch': int, 'perplexity': float | None}
        """
        best_ppl = None
        if self.ppl_history and self._best_epoch > 0:
            # Find the perplexity at the best epoch
            best_ppl = min(self.ppl_history) if self.ppl_history else None

        return {
            "loss": self._best_loss if self._best_loss != float("inf") else 0.0,
            "epoch": self._best_epoch,
            "perplexity": best_ppl,
        }

    def get_latest(self) -> Dict[str, Any]:
        """Return latest metrics.

        Returns:
            {'loss': float, 'smoothed_loss': float, 'perplexity': float | None}
        """
        return {
            "loss": self.loss_history[-1] if self.loss_history else 0.0,
            "smoothed_loss": self._smoothed_loss or 0.0,
            "perplexity": self.ppl_history[-1] if self.ppl_history else None,
            "lr": self.lr_history[-1] if self.lr_history else None,
            "batch": self._batch_count,
            "epoch": self._epoch,
        }

    def reset(self):
        """Reset all metrics."""
        self.loss_history.clear()
        self.ppl_history.clear()
        self.lr_history.clear()
        self._smoothed_loss = None
        self._best_loss = float("inf")
        self._best_epoch = 0
        self._epoch = 0
        self._batch_count = 0

    def __repr__(self) -> str:
        latest = self.get_latest()
        return (
            f"TrainingMetrics(epoch={latest['epoch']}, "
            f"batch={latest['batch']}, "
            f"loss={latest['loss']:.4f}, "
            f"best_loss={self._best_loss:.4f})"
        )
