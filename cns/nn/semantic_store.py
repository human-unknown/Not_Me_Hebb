"""
semantic_store.py — NeuralSemanticStore (v7.2 Phase C)

Neural vector store for semantic knowledge using FAISS or numpy ANN search.

Complements (not replaces) the Hebb-based SemanticMemory — together they form
a dual-system architecture: fast Hebb one-shot + precise neural retrieval.

Architecture:
  Text → TrainableTextEncoder → (64,) L2-norm embedding
  → Store in FAISS IndexFlatIP (or numpy array for exact search)
  → Metadata: {text, valence, arousal, timestamp, source, count}

Usage:
    store = NeuralSemanticStore(text_encoder=text_enc)
    store.insert("猫是哺乳动物", metadata={"valence": 0.3})
    results = store.query("猫")  # → [(0.95, {"text": "猫是哺乳动物", ...}), ...]
    store.forget_old(max_age_steps=10000)
"""

import os
import logging
import time
from typing import Optional, List, Dict, Union, Tuple
import numpy as np

from cns.nn.base import NeuralModule
from cns.nn.config import NNConfig
from cns.nn.bridge import _get_torch

logger = logging.getLogger(__name__)

# Lazy FAISS detection
_HAS_FAISS = False
try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    pass


class NeuralSemanticStore(NeuralModule):
    """Neural vector store for semantic knowledge.

    Uses a text encoder to embed facts/concepts, stores them in a
    vector index with ANN search (FAISS) or exact cosine similarity (numpy).

    This is the "cortical semantic" complement to the "hippocampal episodic"
    Hebb network — precise retrieval for known facts, while Hebb handles
    fast one-shot pattern completion.

    No gradient-based training — the store is updated via insert/forget.
    The text encoder can be pre-trained externally.

    Usage:
        store = NeuralSemanticStore(text_encoder=text_enc)
        idx = store.insert("自由能原理是主动推理的核心")
        results = store.query("自由能")  # → [(sim, metadata), ...]
    """

    def __init__(
        self,
        config: Optional[NNConfig] = None,
        text_encoder: Optional[object] = None,
        dim: int = 64,
        capacity: int = 10000,
    ):
        """Initialize the semantic store.

        Args:
            config: NNConfig (device, etc.)
            text_encoder: TrainableTextEncoder instance (or None for raw vectors)
            dim: Embedding dimension (must match text_encoder output)
            capacity: Maximum number of entries (oldest evicted when full)
        """
        self.dim = dim
        self.capacity = capacity
        self._text_encoder = text_encoder
        self._has_faiss = _HAS_FAISS

        # Storage
        self._embeddings: Optional[np.ndarray] = None  # (n, dim) float32
        self._entries: List[Dict] = []                  # metadata list
        self._index: Optional[object] = None            # FAISS index or None
        self._step_counter: int = 0                     # monotonic timestamp

        super().__init__(name="semantic_store", config=config, trainable=False)

    # ================================================================
    # Build (no trainable params — this is a store)
    # ================================================================

    def _build_network(self):
        """No neural network — the store manages embeddings directly."""
        # Empty ModuleDict to satisfy NeuralModule interface
        torch = _get_torch()
        self._net = torch.nn.ModuleDict({})

    def _forward_impl(self, x):
        """Query by embedding vector — not typically used directly.

        Args:
            x: (B, dim) query embedding vectors

        Returns:
            (B, top_k) similarity scores per query
        """
        if self.n_entries == 0:
            return self._torch.zeros(x.shape[0], 1)
        # Use numpy-based search (tensor→numpy→search→tensor)
        query_np = x.detach().cpu().numpy().astype(np.float32)
        results = []
        for i in range(len(query_np)):
            sims, _ = self._search_numpy(query_np[i], top_k=1)
            results.append([sims[0] if len(sims) > 0 else 0.0])
        return self._torch.tensor(results, dtype=self._torch.float32)

    # ================================================================
    # Encoding
    # ================================================================

    def _encode_text(self, text: str) -> np.ndarray:
        """Encode text to embedding vector using the text encoder.

        Args:
            text: Input text string

        Returns:
            (dim,) float32 L2-normalized embedding
        """
        if self._text_encoder is not None:
            vec = self._text_encoder.encode(text)
            # Ensure L2-normalized
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            return vec.astype(np.float32)
        else:
            raise RuntimeError(
                "No text_encoder provided. Insert raw vectors via _insert_vec()."
            )

    # ================================================================
    # Insert
    # ================================================================

    def insert(
        self,
        text: str,
        metadata: Optional[Dict] = None,
    ) -> int:
        """Insert a text fact into the store.

        Encodes text via the text encoder, then adds to the index.

        Args:
            text: The fact/concept text
            metadata: Optional dict with keys like valence, arousal, source

        Returns:
            Entry index (position in store)
        """
        vec = self._encode_text(text)
        return self._insert_vec(vec, text=text, metadata=metadata)

    def _insert_vec(
        self,
        vec: np.ndarray,
        text: str = "",
        metadata: Optional[Dict] = None,
    ) -> int:
        """Insert a raw embedding vector into the store.

        Args:
            vec: (dim,) float32 embedding (should be L2-normalized)
            text: Associated text (for metadata)
            metadata: Optional metadata dict

        Returns:
            Entry index
        """
        vec = vec.astype(np.float32).ravel()
        if len(vec) != self.dim:
            raise ValueError(
                f"Expected dim={self.dim}, got {len(vec)}"
            )

        # L2-normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        self._step_counter += 1

        # Check for duplicate (same text)
        if text:
            for i, entry in enumerate(self._entries):
                if entry.get("text") == text:
                    # Update count, touch timestamp
                    self._entries[i]["count"] = entry.get("count", 1) + 1
                    self._entries[i]["timestamp"] = self._step_counter
                    # Update embedding (moving average)
                    alpha = 0.3
                    self._embeddings[i] = (
                        (1 - alpha) * self._embeddings[i] + alpha * vec
                    )
                    # Re-normalize
                    n = np.linalg.norm(self._embeddings[i])
                    if n > 0:
                        self._embeddings[i] /= n
                    self._invalidate_index()
                    return i

        # Build entry
        entry = {
            "text": text,
            "timestamp": self._step_counter,
            "count": 1,
        }
        if metadata:
            entry.update(metadata)

        # Evict oldest if at capacity
        if len(self._entries) >= self.capacity:
            oldest_idx = min(
                range(len(self._entries)),
                key=lambda i: self._entries[i].get("timestamp", 0),
            )
            self._entries.pop(oldest_idx)
            if self._embeddings is not None:
                self._embeddings = np.delete(self._embeddings, oldest_idx, axis=0)
            self._invalidate_index()

        # Add
        if self._embeddings is None:
            self._embeddings = vec[np.newaxis, :]
        else:
            self._embeddings = np.concatenate(
                [self._embeddings, vec[np.newaxis, :]], axis=0
            )

        self._entries.append(entry)
        self._invalidate_index()
        return len(self._entries) - 1

    # ================================================================
    # Query
    # ================================================================

    def query(
        self,
        text_or_vec: Union[str, np.ndarray],
        top_k: int = 5,
    ) -> List[Tuple[float, Dict]]:
        """Query the store for similar entries.

        Args:
            text_or_vec: Text string (encoded via text_encoder) or raw (dim,) vector
            top_k: Number of top results to return

        Returns:
            List of (similarity, metadata_dict) sorted by similarity descending.
            similarity ∈ [-1, 1] — cosine similarity.
            Empty list if store is empty.
        """
        if self.n_entries == 0:
            return []

        if isinstance(text_or_vec, str):
            query_vec = self._encode_text(text_or_vec)
        else:
            query_vec = text_or_vec.ravel().astype(np.float32)
            if len(query_vec) != self.dim:
                raise ValueError(
                    f"Query vector dim={len(query_vec)}, expected {self.dim}"
                )
            norm = np.linalg.norm(query_vec)
            if norm > 0:
                query_vec = query_vec / norm

        top_k = min(top_k, self.n_entries)
        sims, indices = self._search_numpy(query_vec, top_k=top_k)

        results = []
        for sim, idx in zip(sims, indices):
            if idx < len(self._entries):
                results.append((float(sim), dict(self._entries[idx])))

        return results

    def _search_numpy(
        self, query_vec: np.ndarray, top_k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Exact cosine similarity search using numpy.

        Args:
            query_vec: (dim,) normalized query vector
            top_k: Number of results

        Returns:
            (similarities, indices) — both shape (top_k,)
        """
        if self._embeddings is None or self.n_entries == 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.int64)

        # Cosine similarity = dot product (both L2-normalized)
        sims = np.dot(self._embeddings, query_vec)  # (n_entries,)

        # Top-k via argpartition
        k = min(top_k, len(sims))
        if k >= len(sims):
            top_indices = np.argsort(-sims)
        else:
            top_indices = np.argpartition(-sims, k - 1)[:k]
            top_indices = top_indices[np.argsort(-sims[top_indices])]

        return sims[top_indices], top_indices

    def _search_faiss(
        self, query_vec: np.ndarray, top_k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """FAISS-based ANN search.

        Args:
            query_vec: (dim,) normalized query vector
            top_k: Number of results

        Returns:
            (similarities, indices)
        """
        if self._index is None:
            self._build_faiss_index()

        query_batch = query_vec.astype(np.float32).reshape(1, -1)
        sims, indices = self._index.search(query_batch, min(top_k, self.n_entries))
        return sims[0], indices[0]

    # ================================================================
    # Index management
    # ================================================================

    def _build_faiss_index(self):
        """Build FAISS index from current embeddings."""
        if not self._has_faiss or self._embeddings is None or self.n_entries == 0:
            self._index = None
            return

        emb = self._embeddings.astype(np.float32).copy()
        # IndexFlatIP = inner product (cosine similarity for L2-normed vectors)
        self._index = faiss.IndexFlatIP(self.dim)
        self._index.add(emb)

    def _invalidate_index(self):
        """Mark FAISS index as stale (needs rebuild)."""
        self._index = None

    def rebuild_index(self):
        """Force rebuild of the search index (after batch operations)."""
        if self._has_faiss:
            self._build_faiss_index()
            if self._index is not None:
                logger.debug(
                    f"FAISS index rebuilt: {self._index.ntotal} vectors"
                )

    # ================================================================
    # Forgetting
    # ================================================================

    def forget_old(self, max_age_steps: int = 1000) -> int:
        """Remove entries older than max_age_steps.

        Args:
            max_age_steps: Maximum age in step-counter units

        Returns:
            Number of entries removed
        """
        if self.n_entries == 0:
            return 0

        current = self._step_counter
        keep_mask = np.array([
            current - e.get("timestamp", 0) < max_age_steps
            for e in self._entries
        ])

        n_removed = int((~keep_mask).sum())
        if n_removed == 0:
            return 0

        self._entries = [
            e for e, keep in zip(self._entries, keep_mask) if keep
        ]
        self._embeddings = self._embeddings[keep_mask]
        self._invalidate_index()

        logger.debug(
            f"Forget: removed {n_removed} old entries, "
            f"{self.n_entries} remaining"
        )
        return n_removed

    # ================================================================
    # Bulk operations
    # ================================================================

    def insert_batch(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict]] = None,
    ) -> List[int]:
        """Insert multiple texts at once (more efficient than single insert).

        Args:
            texts: List of text strings
            metadatas: Optional list of metadata dicts (same length)

        Returns:
            List of entry indices
        """
        if metadatas is None:
            metadatas = [None] * len(texts)

        indices = []
        for text, meta in zip(texts, metadatas):
            idx = self.insert(text, metadata=meta)
            indices.append(idx)

        self.rebuild_index()
        return indices

    # ================================================================
    # Save / Load (override for vector data)
    # ================================================================

    def save(self, path: Optional[str] = None) -> str:
        """Save store to disk (embeddings + metadata, not neural weights).

        Args:
            path: Save path (None = auto)

        Returns:
            Actual save path
        """
        if path is None:
            os.makedirs(self.config.model_dir, exist_ok=True)
            path = os.path.join(self.config.model_dir, f"{self.name}.pt")

        data = {
            "name": self.name,
            "version": self._version,
            "dim": self.dim,
            "capacity": self.capacity,
            "step_counter": self._step_counter,
            "has_faiss": self._has_faiss,
            "n_entries": self.n_entries,
            "embeddings": self._embeddings,
            "entries": self._entries,
        }

        self._torch.save(data, path)
        logger.debug(f"[{self.name}] Saved {self.n_entries} entries to {path}")
        return path

    def load(self, path: Optional[str] = None) -> bool:
        """Load store from disk.

        Args:
            path: Load path (None = auto)

        Returns:
            True if loaded successfully
        """
        if path is None:
            path = os.path.join(self.config.model_dir, f"{self.name}.pt")

        if not os.path.exists(path):
            logger.warning(f"[{self.name}] Store file not found: {path}")
            return False

        checkpoint = self._torch.load(
            path, map_location="cpu", weights_only=False
        )

        self.dim = checkpoint.get("dim", self.dim)
        self.capacity = checkpoint.get("capacity", self.capacity)
        self._step_counter = checkpoint.get("step_counter", 0)
        self._has_faiss = checkpoint.get("has_faiss", _HAS_FAISS)
        self._embeddings = checkpoint.get("embeddings", None)
        self._entries = checkpoint.get("entries", [])

        if self._embeddings is not None:
            self._embeddings = self._embeddings.astype(np.float32)

        self._invalidate_index()
        self.rebuild_index()

        logger.debug(
            f"[{self.name}] Loaded {self.n_entries} entries from {path}"
        )
        return True

    # ================================================================
    # Stats
    # ================================================================

    @property
    def n_entries(self) -> int:
        """Number of stored entries."""
        return len(self._entries)

    @property
    def is_empty(self) -> bool:
        return self.n_entries == 0

    @property
    def is_full(self) -> bool:
        return self.n_entries >= self.capacity

    def get_state(self) -> Dict:
        """Return store state summary."""
        timestamps = [e.get("timestamp", 0) for e in self._entries]
        return {
            "n_entries": self.n_entries,
            "capacity": self.capacity,
            "dim": self.dim,
            "has_faiss": self._has_faiss,
            "step_counter": self._step_counter,
            "oldest_step": min(timestamps) if timestamps else 0,
            "newest_step": max(timestamps) if timestamps else 0,
            "has_text_encoder": self._text_encoder is not None,
        }

    def __repr__(self) -> str:
        return (
            f"NeuralSemanticStore(entries={self.n_entries}/{self.capacity}, "
            f"dim={self.dim}, faiss={self._has_faiss}, "
            f"has_encoder={self._text_encoder is not None})"
        )
