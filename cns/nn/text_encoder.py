"""
text_encoder.py — TrainableTextEncoder (v7.0 Phase B)

Small character-level Transformer for Chinese text encoding.

Replaces: MiniLM-L6-v2 (384d) → PCA (64d) external pre-trained model
With:     2-layer Transformer encoder, char-level, trainable on corpus.txt

Architecture:
  char_embedding (vocab_size → 128)
  + learned positional encoding (max_len=128)
  + 2× TransformerEncoderLayer (d_model=128, nhead=4, ff=256)
  + mean pooling over sequence
  + Linear projection (128 → 64) + LayerNorm
  + L2 normalization

Training: Masked Language Modeling (MLM) on corpus.txt
Inference: encode(text) → mean-pooled encoder output → projection → L2 norm
"""

import os
import re
import logging
from typing import Optional, List, Dict, Union
import numpy as np

from cns.nn.base import NeuralModule
from cns.nn.config import NNConfig
from cns.nn.interfaces import TextEncoder
from cns.nn.bridge import _get_torch, numpy_to_torch, torch_to_numpy

logger = logging.getLogger(__name__)

# Special tokens
PAD_IDX = 0
UNK_IDX = 1
MASK_IDX = 2
CLS_IDX = 3
NUM_SPECIAL = 4


class TrainableTextEncoder(TextEncoder):
    """Small character-level Transformer for Chinese text encoding.

    Trained via Masked Language Modeling on corpus.txt.
    Produces 64-dim L2-normalized semantic vectors matching D=516 text channel.

    Usage:
        encoder = TrainableTextEncoder()
        encoder.build_vocab(corpus_lines)
        encoder.pretrain(corpus_lines, epochs=5)
        vec = encoder.encode("你好世界")  # → (64,) float32
    """

    def __init__(
        self,
        config: Optional[NNConfig] = None,
        trainable: bool = True,
        vocab_size: int = 6000,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        max_len: int = 128,
        mlm_mask_prob: float = 0.15,
    ):
        """Initialize the text encoder.

        Args:
            config: NNConfig (device, dtype, etc.)
            trainable: Whether weights are trainable
            vocab_size: Maximum character vocabulary size
            d_model: Internal embedding dimension
            n_layers: Number of Transformer encoder layers
            n_heads: Number of attention heads
            max_len: Maximum sequence length (chars)
            mlm_mask_prob: MLM masking probability
        """
        # Set attributes BEFORE super().__init__() — _build_network() needs them
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_len = max_len
        self.mlm_mask_prob = mlm_mask_prob

        # Vocabulary (built from corpus)
        self._char2id: Dict[str, int] = {}
        self._id2char: Dict[int, str] = {}
        self._vocab_built: bool = False

        super().__init__(config=config, trainable=trainable)

    # ================================================================
    # Build network
    # ================================================================

    def _build_network(self):
        torch = _get_torch()

        self._char_embed = torch.nn.Embedding(
            self.vocab_size, self.d_model, padding_idx=PAD_IDX
        )
        # Use Embedding instead of raw Parameter — ensures inclusion in state_dict
        self._pos_embed = torch.nn.Embedding(self.max_len, self.d_model)
        torch.nn.init.normal_(self._pos_embed.weight, std=0.02)

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_model * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self._transformer = torch.nn.TransformerEncoder(
            encoder_layer, num_layers=self.n_layers,
            enable_nested_tensor=False,
        )

        self._proj = torch.nn.Sequential(
            torch.nn.Linear(self.d_model, self.OUTPUT_DIM),
            torch.nn.LayerNorm(self.OUTPUT_DIM),
        )

        # MLM head: predict masked tokens
        self._mlm_head = torch.nn.Linear(self.d_model, self.vocab_size)

        # Wrap in ModuleDict for clean state_dict
        self._net = torch.nn.ModuleDict({
            "char_embed": self._char_embed,
            "pos_embed": self._pos_embed,
            "transformer": self._transformer,
            "proj": self._proj,
            "mlm_head": self._mlm_head,
        })

    # ================================================================
    # Vocabulary
    # ================================================================

    def build_vocab(self, corpus: List[str], max_chars: int = 5500):
        """Build character vocabulary from a text corpus.

        Args:
            corpus: List of text strings
            max_chars: Maximum vocabulary size (excluding special tokens)
        """
        from collections import Counter

        char_counts = Counter()
        for line in corpus:
            if isinstance(line, str):
                char_counts.update(line)

        # Most common characters
        top_chars = [c for c, _ in char_counts.most_common(max_chars)]

        # Build mappings (special tokens first)
        self._char2id = {
            "[PAD]": PAD_IDX,
            "[UNK]": UNK_IDX,
            "[MASK]": MASK_IDX,
            "[CLS]": CLS_IDX,
        }
        for i, c in enumerate(top_chars):
            self._char2id[c] = i + NUM_SPECIAL

        self._id2char = {v: k for k, v in self._char2id.items()}
        self._vocab_built = True

        # Update actual vocab size and re-build embeddings if needed
        actual_vocab = len(self._char2id)
        if actual_vocab > self.vocab_size:
            logger.warning(
                f"Vocabulary size {actual_vocab} exceeds configured "
                f"{self.vocab_size}, using {actual_vocab}"
            )
            self.vocab_size = max(self.vocab_size, actual_vocab)
            self._build_network()

        logger.info(
            f"Vocabulary built: {actual_vocab} chars "
            f"({NUM_SPECIAL} special + {actual_vocab - NUM_SPECIAL} chars)"
        )
        return actual_vocab

    def tokenize(self, text: str) -> List[int]:
        """Convert text to token indices.

        Args:
            text: Input text string

        Returns:
            List of token indices (padded/truncated to max_len)
        """
        if not self._vocab_built:
            raise RuntimeError(
                "Vocabulary not built. Call build_vocab() first."
            )

        ids = []
        for c in text[:self.max_len]:
            ids.append(self._char2id.get(c, UNK_IDX))

        # Pad
        if len(ids) < self.max_len:
            ids.extend([PAD_IDX] * (self.max_len - len(ids)))

        return ids

    # ================================================================
    # Core interface
    # ================================================================

    def encode(self, text: str) -> np.ndarray:
        """Encode a single text to a 64-dim semantic vector.

        Args:
            text: Chinese/English text string

        Returns:
            (64,) float32 L2-normalized vector
        """
        ids = self.tokenize(text)
        token_tensor = np.array([ids], dtype=np.int64)
        result = self.forward(token_tensor)
        return result[0]  # (64,)

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode a batch of texts.

        Args:
            texts: List of text strings

        Returns:
            (B, 64) float32 vectors
        """
        batch_ids = np.array(
            [self.tokenize(t) for t in texts], dtype=np.int64
        )
        return self.forward(batch_ids)

    # ================================================================
    # Forward implementation
    # ================================================================

    def _forward_impl(self, x):
        """Forward pass — tensor in, tensor out.

        Args:
            x: (B, max_len) token indices (may be float from bridge)

        Returns:
            (B, 64) float32 L2-normalized tensor
        """
        B = x.shape[0]
        # Ensure int64 for embedding lookup (bridge may convert to float32)
        x = x.long()

        # Embeddings
        L = x.shape[1]
        embed = self._char_embed(x)  # (B, L, d_model)
        positions = self._torch.arange(L, device=x.device).unsqueeze(0)  # (1, L)
        embed = embed + self._pos_embed(positions)  # (B, L, d_model)

        # Create padding mask
        pad_mask = (x == PAD_IDX)  # (B, L)

        # Transformer
        encoded = self._transformer(
            embed, src_key_padding_mask=pad_mask
        )  # (B, L, d_model)

        # Mean pool (exclude padding)
        if pad_mask.any():
            # Masked mean: zero out padded positions
            mask_expanded = (~pad_mask).float().unsqueeze(-1)  # (B, L, 1)
            pooled = (encoded * mask_expanded).sum(dim=1) / (
                mask_expanded.sum(dim=1) + 1e-8
            )
        else:
            pooled = encoded.mean(dim=1)  # (B, d_model)

        # Project and normalize
        output = self._proj(pooled)  # (B, 64)
        output = self._torch.nn.functional.normalize(output, p=2, dim=-1)

        return output

    # ================================================================
    # Training
    # ================================================================

    def _train_step_impl(self, batch):
        """Single MLM training step.

        Args:
            batch: dict with 'input' (B, max_len) int64 token indices
                   and optionally 'mask' (B, max_len) bool mask

        Returns:
            {'loss': float}
        """
        torch = _get_torch()

        x = batch["input"].long()
        mask = batch.get("mask", None)

        B, L = x.shape

        # Create random mask if not provided
        if mask is None:
            # Don't mask special tokens
            is_special = x < NUM_SPECIAL
            rand = torch.rand(B, L, device=x.device)
            mask = (rand < self.mlm_mask_prob) & (~is_special)

        # Replace masked tokens with [MASK]
        x_masked = x.clone()
        x_masked[mask] = MASK_IDX

        # Forward
        embed = self._char_embed(x_masked)
        positions = self._torch.arange(L, device=x.device).unsqueeze(0)
        embed = embed + self._pos_embed(positions)
        pad_mask = (x == PAD_IDX)
        encoded = self._transformer(embed, src_key_padding_mask=pad_mask)

        # MLM prediction
        logits = self._mlm_head(encoded)  # (B, L, vocab_size)

        # Loss only on masked positions
        loss = torch.nn.functional.cross_entropy(
            logits[mask], x[mask], reduction="mean"
        )

        # Backward
        if self._optimizer is None:
            self._optimizer = torch.optim.AdamW(
                self._net.parameters(),
                lr=self.config.effective_lr(self.config.text_lr),
                weight_decay=0.01,
            )

        self._optimizer.zero_grad()
        loss.backward()

        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self._net.parameters(), self.config.grad_clip
            )

        self._optimizer.step()

        # Accuracy on masked tokens
        with torch.no_grad():
            preds = logits[mask].argmax(dim=-1)
            acc = (preds == x[mask]).float().mean()

        return {"loss": float(loss.item()), "accuracy": float(acc.item())}

    # ================================================================
    # Pre-training
    # ================================================================

    def pretrain(
        self,
        corpus: List[str],
        epochs: int = 5,
        batch_size: int = 64,
        lr: Optional[float] = None,
        verbose: bool = True,
    ) -> List[Dict[str, float]]:
        """Pre-train the encoder on a text corpus using MLM.

        Args:
            corpus: List of text strings
            epochs: Number of training epochs
            batch_size: Training batch size
            lr: Learning rate override (None = use config default)
            verbose: Print progress

        Returns:
            Training history: [{'epoch': N, 'loss': float, 'accuracy': float}, ...]
        """
        torch = _get_torch()

        # Build vocab if not yet built
        if not self._vocab_built:
            self.build_vocab(corpus)

        # Prepare optimizer
        effective_lr = lr or self.config.effective_lr(self.config.text_lr)
        if self._optimizer is None:
            self._optimizer = torch.optim.AdamW(
                self._net.parameters(),
                lr=effective_lr,
                weight_decay=0.01,
            )

        # Tokenize all texts
        all_ids = []
        for text in corpus:
            if text and isinstance(text, str) and len(text.strip()) > 0:
                ids = self.tokenize(text.strip())
                all_ids.append(ids)

        if len(all_ids) == 0:
            logger.warning("No valid texts for pre-training")
            return []

        data = np.array(all_ids, dtype=np.int64)
        n_samples = len(data)
        history = []

        for epoch in range(epochs):
            # Shuffle
            perm = np.random.permutation(n_samples)
            epoch_losses = []
            epoch_accs = []

            for i in range(0, n_samples, batch_size):
                idx = perm[i : i + batch_size]
                batch_data = data[idx]
                losses = self.train_step({"input": batch_data})
                epoch_losses.append(losses["loss"])
                epoch_accs.append(losses.get("accuracy", 0.0))

            avg_loss = float(np.mean(epoch_losses))
            avg_acc = float(np.mean(epoch_accs))
            history.append({
                "epoch": epoch + 1,
                "loss": avg_loss,
                "accuracy": avg_acc,
            })

            if verbose or self.config.log_verbose:
                logger.info(
                    f"[text_encoder] Epoch {epoch+1}/{epochs}: "
                    f"loss={avg_loss:.4f}, acc={avg_acc:.3f}"
                )

        return history

    @property
    def is_vocab_built(self) -> bool:
        return self._vocab_built

    def __repr__(self) -> str:
        return (
            f"TrainableTextEncoder(vocab={len(self._char2id)}, "
            f"d_model={self.d_model}, layers={self.n_layers}, "
            f"device={self._device_str}, trainable={self.trainable})"
        )
