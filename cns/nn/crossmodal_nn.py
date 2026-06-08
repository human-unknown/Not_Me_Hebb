"""
crossmodal_nn.py — CrossModalNN (v7.2 Phase C)

CLIP-style contrastive learning for text↔vision alignment.

Projects text and visual embeddings into a shared 128-dim space where
matched pairs are close and mismatched pairs are far apart.

Complements (not replaces) the Hebb-based crossmodal.py — together they
form a dual-system architecture for cross-modal binding.

Architecture:
  Text:  TrainableTextEncoder(64) → Linear(64→128) → L2-norm → shared(128,)
  Image: TrainableVisualEncoder(308) → Linear(308→128) → L2-norm → shared(128,)
  Loss:  InfoNCE with temperature τ (symmetric: text→image + image→text)

Usage:
    cm = CrossModalNN(text_encoder=text_enc, visual_encoder=vis_enc)
    loss = cm.train_step({"text": ["猫", "狗", ...], "image": img_batch})
    scores = cm.retrieve_image("一只猫", image_pool)
"""

import logging
from typing import Optional, List, Dict, Union
import numpy as np

from cns.nn.base import NeuralModule
from cns.nn.config import NNConfig
from cns.nn.bridge import _get_torch, numpy_to_torch, torch_to_numpy

logger = logging.getLogger(__name__)


class CrossModalNN(NeuralModule):
    """Contrastive cross-modal learning (CLIP-style).

    Projects text and visual embeddings into a shared space where
    matched (text, image) pairs are close and mismatched pairs are far apart.

    The text and visual encoders are passed externally so they can be
    shared with other modules (e.g., NeuralSemanticStore).

    Usage:
        text_enc = TrainableTextEncoder()
        text_enc.build_vocab(corpus)
        vis_enc = TrainableVisualEncoder()

        cm = CrossModalNN(text_encoder=text_enc, visual_encoder=vis_enc)
        loss_info = cm.train_step({
            "text": ["一只猫", "一条狗", ...],
            "image": image_batch  # (B, 3, 64, 64)
        })
        # loss_info = {"loss": 2.1, "accuracy": 0.25}

        # Retrieval
        indices = cm.retrieve_image("猫", image_pool, top_k=5)
    """

    def __init__(
        self,
        config: Optional[NNConfig] = None,
        text_encoder: Optional[object] = None,
        visual_encoder: Optional[object] = None,
        shared_dim: int = 128,
        temperature: float = 0.07,
        freeze_encoders: bool = True,
    ):
        """Initialize the cross-modal module.

        Args:
            config: NNConfig (device, dtype, etc.)
            text_encoder: TrainableTextEncoder instance (required for encode_text)
            visual_encoder: TrainableVisualEncoder instance (required for encode_image)
            shared_dim: Dimension of the shared embedding space
            temperature: InfoNCE temperature (lower = sharper distribution)
            freeze_encoders: If True, encoder weights are frozen; only projection
                            heads learn. Set False to fine-tune encoders.
        """
        self.shared_dim = shared_dim
        self.temperature = temperature
        self.freeze_encoders = freeze_encoders
        self._text_encoder = text_encoder
        self._visual_encoder = visual_encoder

        # Determine input dims from encoder output dims
        self._text_input_dim = getattr(text_encoder, 'output_dim', 64) \
            if text_encoder is not None else 64
        self._visual_input_dim = getattr(visual_encoder, 'output_dim', 308) \
            if visual_encoder is not None else 308

        super().__init__(name="crossmodal_nn", config=config, trainable=True)

        # Freeze encoders if requested
        if self.freeze_encoders:
            self._freeze_encoder_weights()

    # ================================================================
    # Build network
    # ================================================================

    def _build_network(self):
        torch = _get_torch()

        # Text projection: text_dim → shared_dim
        self._text_proj = torch.nn.Sequential(
            torch.nn.Linear(self._text_input_dim, self.shared_dim),
            torch.nn.LayerNorm(self.shared_dim),
        )

        # Visual projection: visual_dim → shared_dim
        self._visual_proj = torch.nn.Sequential(
            torch.nn.Linear(self._visual_input_dim, self.shared_dim),
            torch.nn.LayerNorm(self.shared_dim),
        )

        # Wrap in ModuleDict
        self._net = torch.nn.ModuleDict({
            "text_proj": self._text_proj,
            "visual_proj": self._visual_proj,
        })

    def _freeze_encoder_weights(self):
        """Freeze the text and visual encoder weights (only projections learn)."""
        torch = _get_torch()
        for enc in [self._text_encoder, self._visual_encoder]:
            if enc is not None and hasattr(enc, '_net') and enc._net is not None:
                for param in enc._net.parameters():
                    param.requires_grad = False
                enc.trainable = False

    def _unfreeze_encoder_weights(self):
        """Un-freeze encoder weights for fine-tuning."""
        for enc in [self._text_encoder, self._visual_encoder]:
            if enc is not None and hasattr(enc, '_net') and enc._net is not None:
                for param in enc._net.parameters():
                    param.requires_grad = True
                enc.trainable = True

    # ================================================================
    # Encoding (shared space)
    # ================================================================

    def encode_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode text(s) to the shared embedding space.

        Args:
            texts: Single text string or list of strings

        Returns:
            (B, shared_dim) L2-normalized float32 vectors
        """
        if isinstance(texts, str):
            texts = [texts]

        if self._text_encoder is None:
            raise RuntimeError(
                "No text_encoder provided. Cannot encode text."
            )

        # Get text embeddings from encoder
        text_embs = self._text_encoder.encode_batch(texts)  # (B, 64)

        # Project to shared space
        torch = _get_torch()
        x = numpy_to_torch(text_embs, device=self._device_str)

        self.eval()
        with torch.no_grad():
            projected = self._text_proj(x)
            projected = torch.nn.functional.normalize(projected, p=2, dim=-1)

        return torch_to_numpy(projected)

    def encode_image(self, images: np.ndarray) -> np.ndarray:
        """Encode image(s) to the shared embedding space.

        Args:
            images: (B, H, W, 3) uint8 or float32, or (B, 3, H, W) float32

        Returns:
            (B, shared_dim) L2-normalized float32 vectors
        """
        if self._visual_encoder is None:
            raise RuntimeError(
                "No visual_encoder provided. Cannot encode image."
            )

        # If images are (B, H, W, 3), preprocess each
        if images.ndim == 4 and images.shape[-1] == 3:
            # (B, H, W, 3) → encode each
            vis_embs = self._visual_encoder.encode_batch(images)
        elif images.ndim == 4 and images.shape[1] == 3:
            # (B, 3, H, W) — already preprocessed
            vis_embs = self._visual_encoder.forward(images)
        else:
            raise ValueError(
                f"Expected 4D image array, got shape {images.shape}"
            )

        # Project to shared space
        torch = _get_torch()
        x = numpy_to_torch(vis_embs, device=self._device_str)

        self.eval()
        with torch.no_grad():
            projected = self._visual_proj(x)
            projected = torch.nn.functional.normalize(projected, p=2, dim=-1)

        return torch_to_numpy(projected)

    # ================================================================
    # Forward (unused — use encode_text / encode_image)
    # ================================================================

    def _forward_impl(self, x):
        """Not used directly. Use encode_text() or encode_image()."""
        return x

    # ================================================================
    # Training (InfoNCE contrastive loss)
    # ================================================================

    def train_step(self, batch: Dict) -> Dict[str, float]:
        """Override to handle text list (not numpy-convertible).

        The parent NeuralModule.train_step converts dict values via
        numpy_to_torch, which filters out List[str]. We handle the
        text encoding here before passing images to the parent.
        """
        if self._net is None or not self.trainable:
            return {"loss": 0.0}
        if not self.config.training_enabled:
            return {"loss": 0.0}

        self._train_mode = True
        self._net.train()

        texts = batch.get("text", [])
        images = batch.get("image", None)

        if images is None or len(texts) == 0:
            self._train_mode = False
            return {"loss": 0.0, "accuracy": 0.0}

        # Pre-encode texts so the parent only sees numpy arrays
        if isinstance(texts, list) and self._text_encoder is not None:
            text_embs = self._text_encoder.encode_batch(texts)
        elif isinstance(texts, np.ndarray):
            text_embs = texts
        else:
            text_embs = np.array(texts, dtype=np.float32)

        # Prepare tensor batch for images only
        tensor_batch = {
            "image": images,
            "text_embs": text_embs,
        }

        tensor_batch = {
            k: numpy_to_torch(v, device=self._device_str)
            for k, v in tensor_batch.items()
            if isinstance(v, np.ndarray)
        }

        losses = self._train_step_impl(tensor_batch)

        self._step_count += 1
        self._total_trained += 1
        self._train_mode = False
        return losses

    def _train_step_impl(self, batch):
        """Single InfoNCE contrastive training step.

        Args:
            batch: dict with:
                "text_embs": torch.Tensor (N, 64) — pre-encoded text embeddings
                "image": torch.Tensor (N, 3, H, W) or (N, C, H, W)

        Returns:
            {'loss': float, 'accuracy': float}
        """
        torch = _get_torch()

        text_tensor = batch.get("text_embs", None)
        images = batch.get("image", None)

        if text_tensor is None or images is None:
            return {"loss": 0.0, "accuracy": 0.0}

        N = text_tensor.shape[0]
        device = self._device

        # --- Encode images (if raw images, encode via visual encoder) ---
        # Images may be (N, H, W, 3) raw uint8/float32 or (N, 3, H, W) preprocessed float32
        # or (N, 308) pre-computed embeddings
        if images.ndim == 4:
            # Determine format by last dim or known embedding dim
            if images.shape[-1] == 3:
                # (N, H, W, 3) — raw images
                vis_np = images.detach().cpu().numpy()
                if self._visual_encoder is not None:
                    vis_embs = self._visual_encoder.encode_batch(vis_np)
                else:
                    vis_embs = vis_np.reshape(vis_np.shape[0], -1)[:, :308]
                vis_tensor = numpy_to_torch(vis_embs, device=self._device_str)
            elif images.shape[-1] == self._visual_input_dim:
                # (N, 308) — pre-computed embeddings
                vis_tensor = images
            elif images.shape[1] == 3:
                # (N, 3, H, W) — preprocessed images, run through visual encoder forward
                vis_np = images.detach().cpu().numpy()
                if self._visual_encoder is not None:
                    vis_embs = self._visual_encoder.forward(vis_np)
                else:
                    vis_embs = vis_np.reshape(vis_np.shape[0], -1)[:, :308]
                vis_tensor = numpy_to_torch(vis_embs, device=self._device_str)
            else:
                # Unknown format, treat as pre-encoded
                vis_tensor = images
        else:
            # Not 4D — treat as pre-encoded
            vis_tensor = images

        # --- Project to shared space ---
        t_emb = self._text_proj(text_tensor)   # (N, shared_dim)
        v_emb = self._visual_proj(vis_tensor)  # (N, shared_dim)

        # L2-normalize
        t_emb = torch.nn.functional.normalize(t_emb, p=2, dim=-1)
        v_emb = torch.nn.functional.normalize(v_emb, p=2, dim=-1)

        # --- InfoNCE loss ---
        # Similarity matrix: (N, N)
        sim = torch.matmul(t_emb, v_emb.T) / self.temperature

        # Labels: diagonal is correct (text_i ↔ image_i)
        labels = torch.arange(N, device=device)

        # Symmetric loss
        loss_t2i = torch.nn.functional.cross_entropy(sim, labels)
        loss_i2t = torch.nn.functional.cross_entropy(sim.T, labels)
        loss = (loss_t2i + loss_i2t) / 2.0

        # Accuracy: fraction where argmax matches label
        with torch.no_grad():
            acc_t2i = (sim.argmax(dim=-1) == labels).float().mean()
            acc_i2t = (sim.T.argmax(dim=-1) == labels).float().mean()
            acc = (acc_t2i + acc_i2t) / 2.0

        # --- Backward ---
        if self._optimizer is None:
            params = list(self._net.parameters())
            if not self.freeze_encoders:
                for enc in [self._text_encoder, self._visual_encoder]:
                    if enc is not None and enc._net is not None:
                        params.extend(enc._net.parameters())
            self._optimizer = torch.optim.AdamW(
                params,
                lr=self.config.effective_lr(),
                weight_decay=0.01,
            )

        self._optimizer.zero_grad()
        loss.backward()

        if self.config.grad_clip > 0:
            all_params = list(self._net.parameters())
            if not self.freeze_encoders:
                for enc in [self._text_encoder, self._visual_encoder]:
                    if enc is not None and enc._net is not None:
                        all_params.extend(enc._net.parameters())
            torch.nn.utils.clip_grad_norm_(all_params, self.config.grad_clip)

        self._optimizer.step()

        return {
            "loss": float(loss.item()),
            "accuracy": float(acc.item()),
        }

    # ================================================================
    # Retrieval
    # ================================================================

    def retrieve_image(
        self,
        query_text: str,
        image_pool: np.ndarray,
        top_k: int = 5,
    ) -> List[int]:
        """Text → Image retrieval: find best matching images.

        Args:
            query_text: Text query string
            image_pool: (M, H, W, 3) or (M, 3, H, W) image array
            top_k: Number of top matches

        Returns:
            List of image indices sorted by similarity (best first)
        """
        t_emb = self.encode_text(query_text)  # (1, shared_dim)

        # Encode image pool
        v_embs = self.encode_image(image_pool)  # (M, shared_dim)

        # Cosine similarity
        sims = np.dot(v_embs, t_emb[0])  # (M,)
        top_indices = np.argsort(-sims)[:top_k]
        return list(top_indices)

    def retrieve_text(
        self,
        query_image: np.ndarray,
        text_pool: List[str],
        top_k: int = 5,
    ) -> List[int]:
        """Image → Text retrieval: find best matching texts.

        Args:
            query_image: (H, W, 3) or (3, H, W) single image
            text_pool: List of M text strings
            top_k: Number of top matches

        Returns:
            List of text indices sorted by similarity (best first)
        """
        # Encode query image
        if query_image.ndim == 3:
            query_image = query_image[np.newaxis, ...]
        v_emb = self.encode_image(query_image)  # (1, shared_dim)

        t_embs = self.encode_text(text_pool)  # (M, shared_dim)

        sims = np.dot(t_embs, v_emb[0])  # (M,)
        top_indices = np.argsort(-sims)[:top_k]
        return list(top_indices)

    # ================================================================
    # Utilities
    # ================================================================

    def compute_similarity(
        self, text: str, image: np.ndarray
    ) -> float:
        """Compute cosine similarity between a text and image in shared space.

        Args:
            text: Text string
            image: (H, W, 3) single image

        Returns:
            Cosine similarity in [-1, 1]
        """
        t_emb = self.encode_text(text)  # (1, shared_dim)
        if image.ndim == 3:
            image = image[np.newaxis, ...]
        v_emb = self.encode_image(image)  # (1, shared_dim)
        return float(np.dot(t_emb[0], v_emb[0]))

    def compute_similarity_matrix(
        self, texts: List[str], images: np.ndarray
    ) -> np.ndarray:
        """Compute text×image similarity matrix.

        Args:
            texts: List of N text strings
            images: (M, H, W, 3) or (M, 3, H, W)

        Returns:
            (N, M) similarity matrix
        """
        t_embs = self.encode_text(texts)    # (N, shared_dim)
        v_embs = self.encode_image(images)  # (M, shared_dim)
        return np.dot(t_embs, v_embs.T)     # (N, M)

    # ================================================================
    # Save / Load
    # ================================================================

    def save(self, path: Optional[str] = None) -> str:
        """Save model weights and config."""
        return super().save(path)

    def load(self, path: Optional[str] = None) -> bool:
        """Load model weights."""
        ok = super().load(path)
        if ok and self.freeze_encoders:
            self._freeze_encoder_weights()
        return ok

    @property
    def has_text_encoder(self) -> bool:
        return self._text_encoder is not None

    @property
    def has_visual_encoder(self) -> bool:
        return self._visual_encoder is not None

    def __repr__(self) -> str:
        return (
            f"CrossModalNN(text_dim={self._text_input_dim}→"
            f"visual_dim={self._visual_input_dim}→"
            f"{self.shared_dim}, τ={self.temperature}, "
            f"freeze_encoders={self.freeze_encoders}, "
            f"device={self._device_str})"
        )
