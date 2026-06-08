"""
angular_gyrus_nn.py — NeuralAngularGyrus (v7.3 Phase D)

Learned grapheme→phoneme mapping (character→pronunciation).

Replaces: Hebb-based AngularGyrus (grapheme_to_phoneme ClusterNetwork)
With:     Small CNN seq2vec model trained on character→phoneme pairs

Architecture:
  Input: character sequence (max_len=16 chars)
    → char embedding (vocab_size → 64)
    → 2× Conv1d (kernel=3, channels: 64→128→256) + BatchNorm + ReLU + MaxPool
    → GlobalAvgPool1d → (256,)
    → Linear(256→128) → ReLU → Linear(128→64)
    → Output: (64,) phoneme/pronunciation feature vector

Training: Paired (character_sequence, phoneme_vector) MSE
  - Target phoneme vectors from word_spectrum_dataset
  - Maps written character form → pronunciation features

Inference: encode_chars("猫") → (64,) phoneme vector
  - Used in reading pathway: visual text → phoneme → Wernicke comprehension

Complements (not replaces) the Hebb-based AngularGyrus — dual-system architecture.

Usage:
    ag = NeuralAngularGyrus(text_encoder=text_enc)
    ag.build_vocab(chars)
    ag.train_pairs(char_sequences, phoneme_vectors, epochs=5)
    phoneme_vec = ag.encode_chars("猫")
"""

import logging
from typing import Optional, List, Dict
import numpy as np

from cns.nn.base import NeuralModule
from cns.nn.config import NNConfig
from cns.nn.bridge import _get_torch, numpy_to_torch, torch_to_numpy

logger = logging.getLogger(__name__)

# Special tokens (matching TrainableTextEncoder)
PAD_IDX = 0
UNK_IDX = 1
NUM_SPECIAL = 4  # PAD, UNK, MASK, CLS


class NeuralAngularGyrus(NeuralModule):
    """Learned grapheme→phoneme mapping via small CNN seq2vec.

    Takes a character sequence (e.g., "猫") and produces a 64-dim
    phoneme/pronunciation feature vector. Trained on character→phoneme
    pairs from word_spectrum_dataset.

    Usage:
        ag = NeuralAngularGyrus(text_encoder=text_enc)
        ag.build_vocab(corpus)
        ag.train_pairs(char_seqs, phoneme_vecs, epochs=5)
        phoneme_vec = ag.encode_chars("猫")  # → (64,) float32
    """

    def __init__(
        self,
        config: Optional[NNConfig] = None,
        text_encoder: Optional[object] = None,
        d_model: int = 64,
        max_chars: int = 16,
    ):
        """Initialize the neural angular gyrus.

        Args:
            config: NNConfig (device, dtype, etc.)
            text_encoder: TrainableTextEncoder for shared vocab
            d_model: Phoneme output dimension
            max_chars: Maximum character sequence length
        """
        # Set attributes before super().__init__()
        self.d_model = d_model
        self.max_chars = max_chars
        self._text_encoder = text_encoder

        # Vocab
        self._vocab_size = NUM_SPECIAL
        self._char2id: Dict[str, int] = {}
        self._id2char: Dict[int, str] = {}
        self._vocab_built: bool = False

        super().__init__(
            name="neural_angular_gyrus", config=config, trainable=True
        )

    # ================================================================
    # Build network
    # ================================================================

    def _build_network(self):
        torch = _get_torch()

        # Character embedding
        self._char_embed = torch.nn.Embedding(
            self._vocab_size, self.d_model, padding_idx=PAD_IDX
        )

        # Conv1d backbone
        self._conv_stack = torch.nn.Sequential(
            # Block 1: 64 → 128
            torch.nn.Conv1d(self.d_model, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool1d(2),  # 16→8

            # Block 2: 128 → 256
            torch.nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool1d(2),  # 8→4
        )

        # Output projection
        self._output_proj = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, self.d_model),
        )

        self._net = torch.nn.ModuleDict({
            "char_embed": self._char_embed,
            "conv_stack": self._conv_stack,
            "output_proj": self._output_proj,
        })

    # ================================================================
    # Vocabulary
    # ================================================================

    def build_vocab(self, corpus: List[str]):
        """Build or share character vocabulary.

        If a text_encoder with a built vocab is available, shares its mapping.
        Otherwise builds from corpus.

        Args:
            corpus: List of text strings (or list of chars)
        """
        if self._text_encoder is not None and getattr(
            self._text_encoder, '_vocab_built', False
        ):
            self._char2id = dict(self._text_encoder._char2id)
            self._id2char = dict(self._text_encoder._id2char)
            self._vocab_size = len(self._char2id)
            self._vocab_built = True
            self._rebuild_with_new_vocab()
            logger.info(f"Shared vocab with text_encoder: {self._vocab_size} chars")
            return self._vocab_size

        # Build from corpus
        from collections import Counter
        char_counts = Counter()
        for line in corpus:
            if isinstance(line, str):
                char_counts.update(line)

        max_chars = 5500
        top_chars = [c for c, _ in char_counts.most_common(max_chars)]

        self._char2id = {
            "[PAD]": PAD_IDX,
            "[UNK]": UNK_IDX,
            "[MASK]": 2,
            "[CLS]": 3,
        }
        for i, c in enumerate(top_chars):
            self._char2id[c] = i + NUM_SPECIAL

        self._id2char = {v: k for k, v in self._char2id.items()}
        self._vocab_size = len(self._char2id)
        self._vocab_built = True
        self._rebuild_with_new_vocab()

        logger.info(f"Vocabulary built: {self._vocab_size} chars")
        return self._vocab_size

    def _rebuild_with_new_vocab(self):
        """Rebuild char embedding when vocab size changes."""
        torch = _get_torch()
        old_embed = self._char_embed if hasattr(self, '_char_embed') else None

        self._char_embed = torch.nn.Embedding(
            self._vocab_size, self.d_model, padding_idx=PAD_IDX
        )

        if old_embed is not None:
            old_vocab = old_embed.weight.shape[0]
            copy_size = min(old_vocab, self._vocab_size)
            with torch.no_grad():
                self._char_embed.weight[:copy_size] = old_embed.weight[:copy_size]

        self._net["char_embed"] = self._char_embed

    def tokenize(self, chars: str) -> List[int]:
        """Convert character string to token indices.

        Args:
            chars: Input character string (e.g., "猫" or "你好")

        Returns:
            List of token indices (padded/truncated to max_chars)
        """
        if not self._vocab_built:
            raise RuntimeError("Vocabulary not built. Call build_vocab() first.")

        ids = []
        for c in chars[:self.max_chars]:
            ids.append(self._char2id.get(c, UNK_IDX))

        if len(ids) < self.max_chars:
            ids.extend([PAD_IDX] * (self.max_chars - len(ids)))

        return ids

    # ================================================================
    # Core interface
    # ================================================================

    def encode_chars(self, chars: str) -> np.ndarray:
        """Encode a character sequence to a phoneme vector.

        Args:
            chars: Input character string

        Returns:
            (64,) float32 phoneme/pronunciation feature vector
        """
        ids = self.tokenize(chars)
        token_tensor = np.array([ids], dtype=np.int64)
        result = self.forward(token_tensor)
        return result[0]  # (64,)

    def encode_batch(self, chars_list: List[str]) -> np.ndarray:
        """Batch encode character sequences.

        Args:
            chars_list: List of character strings

        Returns:
            (B, 64) float32
        """
        batch_ids = np.array(
            [self.tokenize(c) for c in chars_list], dtype=np.int64
        )
        return self.forward(batch_ids)

    # ================================================================
    # Forward implementation
    # ================================================================

    def _forward_impl(self, x):
        """Forward pass — tensor in, tensor out.

        Args:
            x: (B, max_chars) token indices

        Returns:
            (B, d_model) phoneme vectors
        """
        x = x.long()

        # Embed: (B, max_chars, d_model) → (B, d_model, max_chars)
        embed = self._char_embed(x)  # (B, max_chars, d_model)
        embed = embed.transpose(1, 2)  # (B, d_model, max_chars)

        # Conv stack
        features = self._conv_stack(embed)  # (B, 256, L')

        # Global average pool
        pooled = features.mean(dim=-1)  # (B, 256)

        # Project to phoneme space
        output = self._output_proj(pooled)  # (B, d_model)
        return output

    # ================================================================
    # Training
    # ================================================================

    def _train_step_impl(self, batch):
        """Single MSE training step.

        Args:
            batch: dict with:
                "input": np.ndarray (B, max_chars) int64 token indices
                "target": np.ndarray (B, d_model) phoneme target vectors

        Returns:
            {'loss': float}
        """
        torch = _get_torch()

        x = batch["input"]
        y = batch["target"]

        if isinstance(x, np.ndarray):
            x = numpy_to_torch(x, device=self._device_str)
        if isinstance(y, np.ndarray):
            y = numpy_to_torch(y, device=self._device_str)
        x = x.long()

        # Forward
        pred = self._forward_impl(x)  # (B, d_model)

        # MSE loss
        loss = torch.nn.functional.mse_loss(pred, y)

        # Backward
        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(
                self._net.parameters(),
                lr=self.config.effective_lr(),
            )

        self._optimizer.zero_grad()
        loss.backward()

        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self._net.parameters(), self.config.grad_clip
            )

        self._optimizer.step()

        return {"loss": float(loss.item())}

    # ================================================================
    # Paired training
    # ================================================================

    def train_pairs(
        self,
        char_sequences: List[str],
        phoneme_vectors: np.ndarray,
        epochs: int = 5,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> List[Dict[str, float]]:
        """Train on paired (character_sequence, phoneme_vector) data.

        Args:
            char_sequences: List of character strings (e.g., words)
            phoneme_vectors: (N, d_model) target phoneme vectors
            epochs: Number of training epochs
            batch_size: Training batch size
            verbose: Print progress

        Returns:
            Training history
        """
        if not self._vocab_built:
            self.build_vocab(char_sequences)

        # Tokenize all
        all_ids = np.array(
            [self.tokenize(c) for c in char_sequences], dtype=np.int64
        )
        n_samples = len(all_ids)
        history = []

        self.train()
        for epoch in range(epochs):
            perm = np.random.permutation(n_samples)
            epoch_losses = []

            for i in range(0, n_samples, batch_size):
                idx = perm[i : i + batch_size]
                losses = self.train_step({
                    "input": all_ids[idx],
                    "target": phoneme_vectors[idx],
                })
                epoch_losses.append(losses["loss"])

            avg_loss = float(np.mean(epoch_losses))
            history.append({"epoch": epoch + 1, "loss": avg_loss})

            if verbose or self.config.log_verbose:
                logger.info(
                    f"[neural_angular_gyrus] Epoch {epoch+1}/{epochs}: "
                    f"loss={avg_loss:.6f}"
                )

        self.eval()
        return history

    # ================================================================
    # Properties
    # ================================================================

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def is_vocab_built(self) -> bool:
        return self._vocab_built

    def __repr__(self) -> str:
        return (
            f"NeuralAngularGyrus(vocab={self._vocab_size}, "
            f"d_model={self.d_model}, max_chars={self.max_chars}, "
            f"device={self._device_str})"
        )
