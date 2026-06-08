"""
language_model.py — NeuralGenerator (v7.3 Phase D)

Small autoregressive character-level Transformer for text generation.

Replaces: Broca trigram Hebb chain + full-sentence retrieval
With:     Transformer decoder (GPT-style), ~5M params, trained on corpus.txt

Architecture:
  char_embedding (vocab_size → 256) + learned positional encoding (max_len=256)
  + 4× TransformerDecoderLayer (d_model=256, nhead=8, ff=512, dropout=0.1)
  + causal self-attention mask
  + output projection (256 → vocab_size)

Training: Next-character prediction (autoregressive LM loss) on corpus.txt
Generation: Temperature-sampled top-k decoding with valence/arousal tokens

Complements (not replaces) the Hebb-based Broca — dual-system architecture.

Usage:
    gen = NeuralGenerator(text_encoder=text_enc)
    gen.build_vocab(corpus)
    gen.pretrain(corpus, epochs=10)
    text = gen.generate("你好", valence=0.5, arousal=0.3)
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
# Additional special tokens for generation
BOS_IDX = 2   # Beginning of sequence
EOS_IDX = 3   # End of sequence
NUM_TEXT_SPECIAL = 4

# Emotion special token indices (placed after text special tokens)
V_POS_IDX = 4
V_NEG_IDX = 5
A_HIGH_IDX = 6
A_LOW_IDX = 7
NUM_EMOTION_TOKENS = 4
NUM_SPECIAL = NUM_TEXT_SPECIAL + NUM_EMOTION_TOKENS  # 8 total


def _create_causal_mask(seq_len: int, device) -> "torch.Tensor":
    """Create causal (triangular) attention mask.

    Args:
        seq_len: Sequence length
        device: Torch device

    Returns:
        (seq_len, seq_len) boolean mask (True = attend)
    """
    torch = _get_torch()
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask.bool()


class NeuralGenerator(NeuralModule):
    """Small autoregressive character-level Transformer language model.

    GPT-style decoder-only architecture with causal self-attention.
    Supports emotion-conditioned generation via special valence/arousal tokens.

    Usage:
        gen = NeuralGenerator(text_encoder=text_enc)
        gen.build_vocab(corpus)
        gen.pretrain(corpus, epochs=10, batch_size=32)
        text = gen.generate("你好", valence=0.5, arousal=0.3, max_len=64)
    """

    def __init__(
        self,
        config: Optional[NNConfig] = None,
        text_encoder: Optional[object] = None,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        max_len: int = 256,
        dropout: float = 0.1,
    ):
        """Initialize the neural generator.

        Args:
            config: NNConfig (device, dtype, etc.)
            text_encoder: TrainableTextEncoder for shared vocab
            d_model: Internal embedding dimension
            n_layers: Number of Transformer decoder layers
            n_heads: Number of attention heads
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        # Set attributes before super().__init__() — _build_network() needs them
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_len = max_len
        self.dropout = dropout
        self._text_encoder = text_encoder
        self._vocab_size = NUM_SPECIAL  # minimum (will grow with build_vocab)

        # Vocab mappings (shared with text_encoder)
        self._char2id: Dict[str, int] = {}
        self._id2char: Dict[int, str] = {}
        self._vocab_built: bool = False

        super().__init__(name="neural_generator", config=config, trainable=True)

    # ================================================================
    # Build network
    # ================================================================

    def _build_network(self):
        torch = _get_torch()

        self._char_embed = torch.nn.Embedding(
            self._vocab_size, self.d_model, padding_idx=PAD_IDX
        )
        self._pos_embed = torch.nn.Embedding(self.max_len, self.d_model)
        torch.nn.init.normal_(self._pos_embed.weight, std=0.02)

        # Transformer decoder stack
        decoder_layer = torch.nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_model * 2,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self._decoder = torch.nn.TransformerDecoder(
            decoder_layer, num_layers=self.n_layers
        )

        # Output projection: d_model → vocab_size
        self._output_proj = torch.nn.Linear(self.d_model, self._vocab_size)

        # Tie input embedding and output projection weights
        self._output_proj.weight = self._char_embed.weight

        self._net = torch.nn.ModuleDict({
            "char_embed": self._char_embed,
            "pos_embed": self._pos_embed,
            "decoder": self._decoder,
            "output_proj": self._output_proj,
        })

    # ================================================================
    # Vocabulary
    # ================================================================

    def build_vocab(self, corpus: List[str]):
        """Build character vocabulary from corpus or share with text_encoder.

        If a text_encoder with a built vocab is available, shares its mapping.
        Otherwise builds from the provided corpus.

        Args:
            corpus: List of text strings
        """
        if self._text_encoder is not None and getattr(
            self._text_encoder, '_vocab_built', False
        ):
            # Share vocab with text encoder
            self._char2id = dict(self._text_encoder._char2id)
            self._id2char = dict(self._text_encoder._id2char)
            self._vocab_size = len(self._char2id) + NUM_EMOTION_TOKENS

            # Add emotion tokens
            self._char2id["[V_POS]"] = V_POS_IDX
            self._char2id["[V_NEG]"] = V_NEG_IDX
            self._char2id["[A_HIGH]"] = A_HIGH_IDX
            self._char2id["[A_LOW]"] = A_LOW_IDX
            for idx, name in [
                (V_POS_IDX, "[V_POS]"), (V_NEG_IDX, "[V_NEG]"),
                (A_HIGH_IDX, "[A_HIGH]"), (A_LOW_IDX, "[A_LOW]"),
            ]:
                self._id2char[idx] = name

            self._vocab_built = True
            self._rebuild_with_new_vocab()
            logger.info(
                f"Shared vocab with text_encoder: {self._vocab_size} tokens "
                f"(+{NUM_EMOTION_TOKENS} emotion)"
            )
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
            "[PAD]": PAD_IDX, "[UNK]": UNK_IDX,
            "[BOS]": BOS_IDX, "[EOS]": EOS_IDX,
            "[V_POS]": V_POS_IDX, "[V_NEG]": V_NEG_IDX,
            "[A_HIGH]": A_HIGH_IDX, "[A_LOW]": A_LOW_IDX,
        }
        for i, c in enumerate(top_chars):
            self._char2id[c] = i + NUM_SPECIAL

        self._id2char = {v: k for k, v in self._char2id.items()}
        self._vocab_size = len(self._char2id)
        self._vocab_built = True
        self._rebuild_with_new_vocab()

        logger.info(
            f"Vocabulary built: {self._vocab_size} tokens "
            f"({NUM_SPECIAL} special + {self._vocab_size - NUM_SPECIAL} chars)"
        )
        return self._vocab_size

    def _rebuild_with_new_vocab(self):
        """Rebuild network layers when vocab size changes."""
        torch = _get_torch()
        old_embed = self._char_embed if hasattr(self, '_char_embed') else None

        self._char_embed = torch.nn.Embedding(
            self._vocab_size, self.d_model, padding_idx=PAD_IDX
        )
        self._output_proj = torch.nn.Linear(self.d_model, self._vocab_size)
        self._output_proj.weight = self._char_embed.weight

        if old_embed is not None:
            # Copy overlapping weights from old embedding
            old_vocab = old_embed.weight.shape[0]
            copy_size = min(old_vocab, self._vocab_size)
            with torch.no_grad():
                self._char_embed.weight[:copy_size] = old_embed.weight[:copy_size]

        self._net["char_embed"] = self._char_embed
        self._net["output_proj"] = self._output_proj

    def tokenize(self, text: str, add_bos: bool = True,
                 add_eos: bool = True) -> List[int]:
        """Convert text to token indices.

        Args:
            text: Input text string
            add_bos: Prepend BOS token
            add_eos: Append EOS token

        Returns:
            List of token indices (padded/truncated to max_len)
        """
        if not self._vocab_built:
            raise RuntimeError("Vocabulary not built. Call build_vocab() first.")

        ids = []
        if add_bos:
            ids.append(BOS_IDX)

        max_chars = self.max_len - (1 if add_bos else 0) - (1 if add_eos else 0)
        for c in text[:max_chars]:
            ids.append(self._char2id.get(c, UNK_IDX))

        if add_eos:
            ids.append(EOS_IDX)

        # Pad
        if len(ids) < self.max_len:
            ids.extend([PAD_IDX] * (self.max_len - len(ids)))

        return ids[:self.max_len]

    def detokenize(self, ids: List[int], skip_special: bool = True) -> str:
        """Convert token indices back to text.

        Args:
            ids: List of token indices
            skip_special: Skip special tokens (PAD, BOS, EOS, emotion)

        Returns:
            Decoded text string
        """
        chars = []
        for idx in ids:
            if skip_special and idx < NUM_SPECIAL:
                if idx == EOS_IDX:
                    break  # Stop at EOS
                continue
            c = self._id2char.get(idx, "")
            if c and not c.startswith("["):
                chars.append(c)
        return "".join(chars)

    # ================================================================
    # Emotion token helpers
    # ================================================================

    def _get_emotion_tokens(self, valence: float, arousal: float) -> List[int]:
        """Get emotion special tokens based on valence/arousal values.

        Args:
            valence: [-1, 1] valence value
            arousal: [0, 1] arousal value

        Returns:
            List of emotion token indices (possibly empty if neutral)
        """
        tokens = []
        if valence > 0.2:
            tokens.append(V_POS_IDX)
        elif valence < -0.2:
            tokens.append(V_NEG_IDX)

        if arousal > 0.5:
            tokens.append(A_HIGH_IDX)
        elif arousal < 0.2:
            tokens.append(A_LOW_IDX)

        return tokens

    # ================================================================
    # Forward
    # ================================================================

    def _forward_impl(self, x):
        """Forward pass — tensor in, tensor out.

        Args:
            x: (B, seq_len) token indices (int64)

        Returns:
            (B, seq_len, vocab_size) logits
        """
        B, L = x.shape
        x = x.long()

        # Embed
        embed = self._char_embed(x)  # (B, L, d_model)
        positions = self._torch.arange(L, device=x.device).unsqueeze(0)
        embed = embed + self._pos_embed(positions)

        # Causal mask
        causal_mask = _create_causal_mask(L, x.device)

        # Pad mask
        pad_mask = (x == PAD_IDX)

        # Decoder forward (using TransformerDecoder with memory=None → self-attention only)
        # TransformerDecoderLayer expects (tgt, memory); pass tgt to both for decoder-only
        decoded = self._decoder(
            tgt=embed,
            memory=embed,  # Self-attention path
            tgt_mask=causal_mask,
            tgt_key_padding_mask=pad_mask,
        )

        # Project to vocab
        logits = self._output_proj(decoded)  # (B, L, vocab_size)
        return logits

    # ================================================================
    # Training
    # ================================================================

    def _train_step_impl(self, batch):
        """Single LM training step (next-char prediction).

        Args:
            batch: dict with:
                "input": np.ndarray (B, seq_len) int64 token indices

        Returns:
            {'loss': float, 'perplexity': float}
        """
        torch = _get_torch()

        x = batch["input"]
        if isinstance(x, np.ndarray):
            x = numpy_to_torch(x, device=self._device_str)
        x = x.long()

        B, L = x.shape

        # Forward
        logits = self._forward_impl(x)  # (B, L, vocab_size)

        # Shift: predict next token
        # logits[:, :-1, :] predicts x[:, 1:]
        logits_shifted = logits[:, :-1, :].contiguous()  # (B, L-1, V)
        targets = x[:, 1:].contiguous()                    # (B, L-1)

        # Mask out padding targets
        valid_mask = (targets != PAD_IDX)

        if valid_mask.sum() == 0:
            return {"loss": 0.0, "perplexity": 1.0}

        loss = torch.nn.functional.cross_entropy(
            logits_shifted.view(-1, self._vocab_size),
            targets.view(-1),
            ignore_index=PAD_IDX,
            reduction="mean",
        )

        # Perplexity
        with torch.no_grad():
            ppl = torch.exp(loss).item()

        # Backward
        if self._optimizer is None:
            self._optimizer = torch.optim.AdamW(
                self._net.parameters(),
                lr=self.config.effective_lr(),
                weight_decay=0.01,
            )

        self._optimizer.zero_grad()
        loss.backward()

        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self._net.parameters(), self.config.grad_clip
            )

        self._optimizer.step()

        return {
            "loss": float(loss.item()),
            "perplexity": float(ppl),
        }

    # ================================================================
    # Generation
    # ================================================================

    def generate(
        self,
        prompt: str = "",
        valence: float = 0.0,
        arousal: float = 0.0,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int = 50,
        seed: Optional[int] = None,
    ) -> str:
        """Generate text autoregressively with emotion conditioning.

        Args:
            prompt: Starting text (empty = generate from scratch with BOS)
            valence: [-1, 1] emotion valence for conditioning
            arousal: [0, 1] emotion arousal for conditioning
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (<1 = sharper, >1 = more random)
            top_k: Top-k sampling (0 = no filtering)
            seed: Random seed for reproducibility

        Returns:
            Generated text string
        """
        torch = _get_torch()
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        device = self._device

        # Build input sequence
        input_ids = [BOS_IDX]

        # Add emotion tokens
        emotion_tokens = self._get_emotion_tokens(valence, arousal)
        input_ids.extend(emotion_tokens)

        # Add prompt tokens
        if prompt:
            for c in prompt[:self.max_len - len(input_ids) - 1]:
                input_ids.append(self._char2id.get(c, UNK_IDX))

        # Truncate if needed
        input_ids = input_ids[:self.max_len - 1]

        self.eval()
        with torch.no_grad():
            while len(input_ids) < self.max_len:
                # Pad current sequence
                seq = input_ids + [PAD_IDX] * (self.max_len - len(input_ids))
                x = torch.tensor([seq], device=device, dtype=torch.long)

                # Forward
                logits = self._forward_impl(x)  # (1, max_len, vocab_size)

                # Get logits at the last non-padding position
                last_pos = len(input_ids) - 1  # 0-indexed
                next_logits = logits[0, last_pos, :]  # (vocab_size,)

                # Temperature scaling
                next_logits = next_logits / max(temperature, 0.01)

                # Top-k filtering
                if top_k > 0 and top_k < self._vocab_size:
                    top_k_vals, _ = torch.topk(next_logits, top_k)
                    threshold = top_k_vals[-1]
                    next_logits[next_logits < threshold] = float("-inf")

                # Sample
                probs = torch.nn.functional.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

                # Stop at EOS
                if next_token == EOS_IDX:
                    break

                input_ids.append(next_token)

                # Stop if generated enough
                if len(input_ids) >= max_new_tokens + len(emotion_tokens) + (
                    1 if prompt else 0
                ):
                    break

        # Decode (skip BOS + emotion tokens + prompt)
        start_idx = 1 + len(emotion_tokens) + (len(prompt) if prompt else 0)
        generated_ids = input_ids[start_idx:]
        return self.detokenize(generated_ids)

    # ================================================================
    # Pre-training
    # ================================================================

    def pretrain(
        self,
        corpus: List[str],
        epochs: int = 10,
        batch_size: int = 32,
        lr: Optional[float] = None,
        verbose: bool = True,
    ) -> List[Dict[str, float]]:
        """Pre-train the language model on a text corpus.

        Args:
            corpus: List of text strings
            epochs: Number of training epochs
            batch_size: Training batch size
            lr: Learning rate override
            verbose: Print progress

        Returns:
            Training history
        """
        torch = _get_torch()

        if not self._vocab_built:
            self.build_vocab(corpus)

        # Prepare optimizer
        effective_lr = lr or self.config.effective_lr()
        self._optimizer = torch.optim.AdamW(
            self._net.parameters(),
            lr=effective_lr,
            weight_decay=0.01,
        )

        # Tokenize corpus
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

        self.train()
        for epoch in range(epochs):
            perm = np.random.permutation(n_samples)
            epoch_losses = []
            epoch_ppls = []

            for i in range(0, n_samples, batch_size):
                idx = perm[i : i + batch_size]
                batch_data = data[idx]
                losses = self.train_step({"input": batch_data})
                epoch_losses.append(losses["loss"])
                epoch_ppls.append(losses.get("perplexity", 1.0))

            avg_loss = float(np.mean(epoch_losses))
            avg_ppl = float(np.mean(epoch_ppls))
            history.append({
                "epoch": epoch + 1,
                "loss": avg_loss,
                "perplexity": avg_ppl,
            })

            if verbose or self.config.log_verbose:
                logger.info(
                    f"[neural_generator] Epoch {epoch+1}/{epochs}: "
                    f"loss={avg_loss:.4f}, ppl={avg_ppl:.1f}"
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

    @property
    def emotion_token_ids(self) -> Dict[str, int]:
        return {
            "V_POS": V_POS_IDX, "V_NEG": V_NEG_IDX,
            "A_HIGH": A_HIGH_IDX, "A_LOW": A_LOW_IDX,
        }

    def __repr__(self) -> str:
        return (
            f"NeuralGenerator(vocab={self._vocab_size}, "
            f"d_model={self.d_model}, layers={self.n_layers}, "
            f"max_len={self.max_len}, device={self._device_str})"
        )
