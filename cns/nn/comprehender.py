"""
comprehender.py — NeuralComprehender (v7.3 Phase D)

Memory-augmented text comprehension with N400/P600 prediction errors.

Replaces: Wernicke memory-retrieval comprehension
With:     Text encoder + memory cross-attention + prediction error computation

Architecture:
  Input text → TrainableTextEncoder.encode() → (64,) embedding
  → Query memory store (NeuralSemanticStore or episodic context)
  → Cross-attention: weighted blend of retrieved memories + input
  → N400 = 1 - cosine_sim(input_emb, predicted_emb)  (semantic surprise)
  → P600 = syntactic anomaly score (placeholder, character-level entropy)

Complements (not replaces) the Hebb-based Wernicke — dual-system architecture.

Usage:
    comp = NeuralComprehender(text_encoder=text_enc, memory_store=store)
    result = comp.comprehend("你好，今天天气真好")
    # result = {"comprehension_vec": (64,), "n400": 0.1, "p600": 0.05, ...}
"""

import logging
from typing import Optional, List, Dict, Tuple
import numpy as np

from cns.nn.base import NeuralModule
from cns.nn.config import NNConfig
from cns.nn.bridge import _get_torch

logger = logging.getLogger(__name__)


class NeuralComprehender(NeuralModule):
    """Memory-augmented text comprehension.

    Uses a text encoder to embed input, retrieves relevant memories,
    integrates them into a comprehension vector, and computes
    prediction errors (N400/P600 analogs).

    N400 = semantic prediction error — how surprising is this input?
    P600 = syntactic processing difficulty — placeholder based on char entropy.

    No trainable parameters — comprehension is retrieval + integration.
    The text encoder can be pre-trained externally.

    Usage:
        text_enc = TrainableTextEncoder(); text_enc.build_vocab(corpus)
        store = NeuralSemanticStore(text_encoder=text_enc)
        comp = NeuralComprehender(text_encoder=text_enc, memory_store=store)
        result = comp.comprehend("自由能原理很有趣")
    """

    def __init__(
        self,
        config: Optional[NNConfig] = None,
        text_encoder: Optional[object] = None,
        memory_store: Optional[object] = None,
        context_size: int = 5,
        n400_ema_alpha: float = 0.1,
    ):
        """Initialize the comprehender.

        Args:
            config: NNConfig
            text_encoder: TrainableTextEncoder for encoding input
            memory_store: NeuralSemanticStore (or any object with .query(text, top_k))
            context_size: Number of recent comprehension vectors to keep as context
            n400_ema_alpha: EMA smoothing for N400 baseline
        """
        self._text_encoder = text_encoder
        self._memory_store = memory_store
        self.context_size = context_size
        self.n400_ema_alpha = n400_ema_alpha

        # Context window: list of comprehension vectors
        self._context_window: List[np.ndarray] = []
        # Baseline tracking
        self._n400_ema: float = 0.5
        self._p600_ema: float = 0.3
        self._comprehend_count: int = 0

        super().__init__(
            name="neural_comprehender", config=config, trainable=False
        )

    # ================================================================
    # Build (no trainable params)
    # ================================================================

    def _build_network(self):
        """No trainable parameters — comprehension is pure computation."""
        torch = _get_torch()
        self._net = torch.nn.ModuleDict({})

    def _forward_impl(self, x):
        """Forward with pre-encoded text embeddings — returns comprehension.

        Args:
            x: (B, 64) pre-encoded text embeddings

        Returns:
            (B, 64) comprehension vectors
        """
        # For tensor interface: simple pass-through with prediction error
        # Real comprehension is done via comprehend()
        return x

    # ================================================================
    # Comprehension pipeline
    # ================================================================

    def comprehend(
        self,
        text: str,
        context_window: Optional[List[np.ndarray]] = None,
    ) -> Dict:
        """Full comprehension pipeline for a text input.

        1. Encode input text
        2. Predict expected input from context
        3. Compute N400 (semantic prediction error)
        4. Compute P600 (syntactic processing cost)
        5. Retrieve relevant memories
        6. Integrate: blend input + memories into comprehension vector

        Args:
            text: Input text string
            context_window: Optional external context vectors

        Returns:
            dict with keys:
                comprehension_vec: (64,) float32 — integrated understanding
                n400: float — semantic prediction error [0, 1]
                p600: float — syntactic processing cost [0, 1]
                attended_memories: list of (similarity, metadata) tuples
                input_embedding: (64,) float32 — raw text encoding
        """
        if self._text_encoder is None:
            raise RuntimeError(
                "No text_encoder provided. Cannot comprehend text."
            )

        # Step 1: Encode input
        input_emb = self._text_encoder.encode(text)  # (64,)
        input_emb = input_emb.astype(np.float32)

        # Step 2: Predict expected input from context
        predicted_emb = self._predict_from_context(context_window)

        # Step 3: N400 = 1 - cosine_sim(input, predicted)
        n400 = self._compute_n400(input_emb, predicted_emb)

        # Step 4: P600 = char-level entropy approximation
        p600 = self._compute_p600(text)

        # Step 5: Retrieve relevant memories
        attended = self._retrieve_memories(text, top_k=5)

        # Step 6: Integrate — blend input + attended memories
        comprehension_vec = self._integrate(input_emb, attended)

        # Update context
        self._context_window.append(comprehension_vec.copy())
        if len(self._context_window) > self.context_size:
            self._context_window.pop(0)

        # Update EMA baselines
        self._n400_ema = (
            (1 - self.n400_ema_alpha) * self._n400_ema
            + self.n400_ema_alpha * n400
        )
        self._p600_ema = (
            (1 - self.n400_ema_alpha) * self._p600_ema
            + self.n400_ema_alpha * p600
        )
        self._comprehend_count += 1

        return {
            "comprehension_vec": comprehension_vec.astype(np.float32),
            "n400": float(n400),
            "p600": float(p600),
            "attended_memories": attended,
            "input_embedding": input_emb.astype(np.float32),
        }

    # ================================================================
    # N400 / P600 computation
    # ================================================================

    def _predict_from_context(
        self,
        context_window: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """Predict expected input embedding from context.

        Uses EMA of recent comprehension vectors as the prediction.
        If no context, returns zeros (maximum surprise).

        Args:
            context_window: Optional external context

        Returns:
            (64,) predicted embedding
        """
        ctx = context_window if context_window is not None else self._context_window

        if len(ctx) == 0:
            return np.zeros(64, dtype=np.float32)

        # Weighted average: recent items have higher weight
        weights = np.exp(np.linspace(-1, 0, len(ctx)))
        weights = weights / weights.sum()
        predicted = sum(w * v for w, v in zip(weights, ctx))
        return predicted.astype(np.float32)

    def _compute_n400(
        self, input_emb: np.ndarray, predicted_emb: np.ndarray
    ) -> float:
        """Compute N400: semantic prediction error.

        N400 = 1 - cosine_similarity(input, predicted)
        Higher N400 = more semantically surprising.

        If predicted is zero vector (no context), N400 defaults to 0.5
        (moderate uncertainty rather than extreme surprise).

        Args:
            input_emb: (64,) actual input embedding
            predicted_emb: (64,) predicted embedding

        Returns:
            n400 ∈ [0, 1]
        """
        pred_norm = np.linalg.norm(predicted_emb)
        if pred_norm < 1e-8:
            # No prediction available — return 0.5 (moderate uncertainty)
            return 0.5

        input_norm = np.linalg.norm(input_emb)
        if input_norm < 1e-8:
            return 0.0

        cos_sim = np.dot(input_emb, predicted_emb) / (input_norm * pred_norm)
        cos_sim = np.clip(cos_sim, -1.0, 1.0)

        # N400 = 1 - cos_sim, scaled to [0, 1]
        n400 = (1.0 - float(cos_sim)) / 2.0
        return n400

    def _compute_p600(self, text: str) -> float:
        """Compute P600: syntactic processing cost.

        Simplified version: char-level entropy as a proxy for
        processing difficulty. More varied text = higher P600.

        In a full implementation, this would use a syntactic parser.

        Args:
            text: Input text

        Returns:
            p600 ∈ [0, 1]
        """
        if len(text) == 0:
            return 0.0

        # Character diversity as a simple proxy
        unique_ratio = len(set(text)) / max(len(text), 1)

        # Length factor: longer sentences are syntactically more complex
        length_factor = min(len(text) / 50.0, 1.0)

        # Heuristic: P600 = blend of diversity and length
        p600 = 0.3 * unique_ratio + 0.7 * length_factor
        return float(np.clip(p600, 0.0, 1.0))

    # ================================================================
    # Memory retrieval
    # ================================================================

    def _retrieve_memories(
        self, text: str, top_k: int = 5
    ) -> List[Tuple[float, Dict]]:
        """Retrieve relevant memories for the input text.

        Args:
            text: Input text
            top_k: Number of memories to retrieve

        Returns:
            List of (similarity, metadata) tuples
        """
        if self._memory_store is None:
            return []

        try:
            results = self._memory_store.query(text, top_k=top_k)
            return results
        except Exception as e:
            logger.debug(f"Memory retrieval failed: {e}")
            return []

    def _integrate(
        self,
        input_emb: np.ndarray,
        attended_memories: List[Tuple[float, Dict]],
    ) -> np.ndarray:
        """Integrate input embedding with attended memories.

        Blends the raw input with relevant memory vectors.
        Without memories, comprehension = raw input.

        Args:
            input_emb: (64,) input embedding
            attended_memories: List of (similarity, metadata)

        Returns:
            (64,) integrated comprehension vector
        """
        if not attended_memories:
            return input_emb.copy()

        # Weighted blend: input (0.6) + top memory (0.4)
        # The first memory is the most relevant
        top_sim, top_meta = attended_memories[0]

        # Encode the memory text for blending
        memory_text = top_meta.get("text", "")
        if memory_text and self._text_encoder is not None:
            try:
                memory_emb = self._text_encoder.encode(memory_text)
            except Exception:
                memory_emb = input_emb  # Fallback to input
        else:
            memory_emb = input_emb

        # Blend: input_weight decreases with stronger memory match
        blend_weight = float(np.clip(top_sim, 0.0, 0.8))
        integrated = (1.0 - blend_weight) * input_emb + blend_weight * memory_emb

        # Re-normalize
        norm = np.linalg.norm(integrated)
        if norm > 0:
            integrated = integrated / norm

        return integrated.astype(np.float32)

    # ================================================================
    # Context management
    # ================================================================

    def get_context_vector(self, window_size: int = 3) -> np.ndarray:
        """Get recent comprehension context as a single vector.

        Useful for feeding context into a generator module.

        Args:
            window_size: Number of recent turns to include

        Returns:
            (64,) context vector (mean of recent comprehension vectors)
        """
        recent = self._context_window[-window_size:]
        if not recent:
            return np.zeros(64, dtype=np.float32)
        return np.mean(recent, axis=0).astype(np.float32)

    def reset_context(self):
        """Clear the context window."""
        self._context_window = []

    # ================================================================
    # Stats
    # ================================================================

    @property
    def n400_baseline(self) -> float:
        """EMA-smoothed N400 baseline."""
        return self._n400_ema

    @property
    def p600_baseline(self) -> float:
        """EMA-smoothed P600 baseline."""
        return self._p600_ema

    @property
    def n400_surprise(self) -> float:
        """Normalized surprise: how much current N400 exceeds baseline."""
        if self._n400_ema < 0.01:
            return 0.0
        last_n400 = 0.5  # Default if no context
        if self._context_window:
            last_n400 = self._compute_n400(
                self._context_window[-1],
                self._predict_from_context(self._context_window[:-1])
                if len(self._context_window) > 1 else None,
            )
        return float(np.clip(last_n400 / self._n400_ema - 1.0, -1.0, 1.0))

    def get_state(self) -> Dict:
        """Return comprehender state summary."""
        return {
            "comprehend_count": self._comprehend_count,
            "context_size": len(self._context_window),
            "n400_ema": float(self._n400_ema),
            "p600_ema": float(self._p600_ema),
            "has_memory_store": self._memory_store is not None,
            "has_text_encoder": self._text_encoder is not None,
        }

    def __repr__(self) -> str:
        return (
            f"NeuralComprehender(context={len(self._context_window)}/"
            f"{self.context_size}, n400_ema={self._n400_ema:.3f}, "
            f"has_store={self._memory_store is not None})"
        )
