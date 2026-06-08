"""
visual_encoder.py — TrainableVisualEncoder (v7.0 Phase B)

Small CNN with sub-pathway projection heads for visual feature extraction.

Replaces: Gabor filter bank → V1→V2→V4→IT hand-crafted pipeline
With:     Shared CNN backbone + 7 parallel sub-pathway heads

Architecture:
  Input image (H×W×3) → resize 64×64
  → Shared CNN backbone (4 conv blocks: 3→32→64→128→256)
  → AdaptiveAvgPool2d → (256,) global feature
  → 7 parallel projection heads → concatenate → (308,)

Sub-pathways (matching VisualEncoder.SUBPATHWAYS):
  m_pathway:  256→96   (motion/spatial — M pathway)
  p_pathway:  256→112  (shape/detail — P pathway)
  k_pathway:  256→48   (color — K pathway)
  it:         256→16   (object recognition — IT)
  sc:         256→16   (saliency — Superior Colliculus)
  pulvinar:   256→12   (thalamic shortcut)
  binding:    256→8    (FPN binding signal)

Training: Autoencoder reconstruction (encode→decode→MSE)
"""

import logging
from typing import Optional, List, Dict
import numpy as np

from cns.nn.base import NeuralModule
from cns.nn.config import NNConfig
from cns.nn.interfaces import VisualEncoder
from cns.nn.bridge import _get_torch, numpy_to_torch, torch_to_numpy

logger = logging.getLogger(__name__)


class TrainableVisualEncoder(VisualEncoder):
    """Small CNN with 7 sub-pathway projection heads.

    Produces 308-dim visual feature vectors matching D=516 visual channels.

    Usage:
        encoder = TrainableVisualEncoder()
        vec = encoder.encode(image_array)  # → (308,) float32
    """

    def __init__(
        self,
        config: Optional[NNConfig] = None,
        trainable: bool = True,
        input_size: int = 64,
        base_channels: int = 32,
    ):
        """Initialize the visual encoder.

        Args:
            config: NNConfig (device, dtype, etc.)
            trainable: Whether weights are trainable
            input_size: Input image size (square, resized to this)
            base_channels: Base channel count for conv blocks
        """
        # Set attributes BEFORE super().__init__() — _build_network() needs them
        self.input_size = input_size
        self.base_channels = base_channels
        super().__init__(config=config, trainable=trainable)

    # ================================================================
    # Build network
    # ================================================================

    def _build_network(self):
        torch = _get_torch()
        ch = self.base_channels

        # --- Shared backbone ---
        self._backbone = torch.nn.Sequential(
            # Block 1: 3 → ch
            torch.nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(ch),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),  # 64→32

            # Block 2: ch → ch*2
            torch.nn.Conv2d(ch, ch * 2, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(ch * 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),  # 32→16

            # Block 3: ch*2 → ch*4
            torch.nn.Conv2d(ch * 2, ch * 4, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(ch * 4),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),  # 16→8

            # Block 4: ch*4 → ch*8
            torch.nn.Conv2d(ch * 4, ch * 8, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(ch * 8),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),  # 8→4
        )

        self._backbone_dim = ch * 8  # 256 when ch=32
        self._pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        # --- 7 sub-pathway projection heads ---
        self._heads = torch.nn.ModuleDict({
            "m_pathway": torch.nn.Sequential(
                torch.nn.Linear(self._backbone_dim, 96),
                torch.nn.LayerNorm(96),
            ),
            "p_pathway": torch.nn.Sequential(
                torch.nn.Linear(self._backbone_dim, 112),
                torch.nn.LayerNorm(112),
            ),
            "k_pathway": torch.nn.Sequential(
                torch.nn.Linear(self._backbone_dim, 48),
                torch.nn.LayerNorm(48),
            ),
            "it": torch.nn.Sequential(
                torch.nn.Linear(self._backbone_dim, 16),
                torch.nn.LayerNorm(16),
            ),
            "sc": torch.nn.Sequential(
                torch.nn.Linear(self._backbone_dim, 16),
                torch.nn.LayerNorm(16),
            ),
            "pulvinar": torch.nn.Sequential(
                torch.nn.Linear(self._backbone_dim, 12),
                torch.nn.LayerNorm(12),
            ),
            "binding": torch.nn.Sequential(
                torch.nn.Linear(self._backbone_dim, 8),
                torch.nn.LayerNorm(8),
            ),
        })

        # --- Decoder for autoencoder training ---
        self._decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                self._backbone_dim, ch * 4,
                kernel_size=4, stride=2, padding=1
            ),  # 4→8
            torch.nn.BatchNorm2d(ch * 4),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(
                ch * 4, ch * 2,
                kernel_size=4, stride=2, padding=1
            ),  # 8→16
            torch.nn.BatchNorm2d(ch * 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(
                ch * 2, ch,
                kernel_size=4, stride=2, padding=1
            ),  # 16→32
            torch.nn.BatchNorm2d(ch),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(
                ch, 3,
                kernel_size=4, stride=2, padding=1
            ),  # 32→64
            torch.nn.Sigmoid(),
        )

        # Wrap in ModuleDict for clean state_dict
        self._net = torch.nn.ModuleDict({
            "backbone": self._backbone,
            "heads": self._heads,
            "decoder": self._decoder,
        })

    # ================================================================
    # Image preprocessing
    # ================================================================

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for encoding.

        Args:
            image: (H, W, 3) uint8 [0-255] or float32 [0-1]

        Returns:
            (1, 3, input_size, input_size) float32 [0,1]
        """
        # Handle grayscale
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

        # Normalize to [0, 1]
        if image.dtype == np.uint8 or image.max() > 1.0:
            image = image.astype(np.float32) / 255.0

        image = image.astype(np.float32)
        image = np.clip(image, 0.0, 1.0)

        # Resize (simple numpy resize — downsample/upsample via slicing for
        # speed; PIL import avoided to keep dependency light)
        H, W = image.shape[:2]
        if H != self.input_size or W != self.input_size:
            # Use simple cropping/centering for speed
            if H > self.input_size:
                start_h = (H - self.input_size) // 2
                image = image[start_h:start_h + self.input_size, :, :]
            if W > self.input_size:
                start_w = (W - self.input_size) // 2
                image = image[:, start_w:start_w + self.input_size, :]
            H, W = image.shape[:2]
            # Pad if smaller
            if H < self.input_size or W < self.input_size:
                pad_h = (self.input_size - H) // 2
                pad_w = (self.input_size - W) // 2
                padded = np.zeros(
                    (self.input_size, self.input_size, 3), dtype=np.float32
                )
                padded[
                    pad_h:pad_h + H, pad_w:pad_w + W, :
                ] = image[:self.input_size, :self.input_size, :]
                image = padded
            image = image[:self.input_size, :self.input_size, :]

        # HWC → CHW
        image = np.transpose(image, (2, 0, 1))  # (3, H, W)
        image = np.expand_dims(image, axis=0)   # (1, 3, H, W)
        return image.astype(np.float32)

    # ================================================================
    # Core interface
    # ================================================================

    def encode(self, image: np.ndarray) -> np.ndarray:
        """Encode a single image to a 308-dim visual feature vector.

        Args:
            image: (H, W, 3) uint8 or float32

        Returns:
            (308,) float32 vector
        """
        preprocessed = self.preprocess_image(image)
        result = self.forward(preprocessed)
        return result[0]  # (308,)

    def encode_batch(self, images: np.ndarray) -> np.ndarray:
        """Encode a batch of images.

        Args:
            images: (B, H, W, 3) uint8 or float32

        Returns:
            (B, 308) float32
        """
        batch_tensors = []
        for i in range(len(images)):
            batch_tensors.append(self.preprocess_image(images[i]))
        batch = np.concatenate(batch_tensors, axis=0)
        return self.forward(batch)

    # ================================================================
    # Forward implementation
    # ================================================================

    def _forward_impl(self, x):
        """Forward pass — tensor in, tensor out.

        Args:
            x: (B, 3, H, W) float32 tensor [0,1]

        Returns:
            (B, 308) float32 tensor
        """
        # Backbone
        features = self._backbone(x)  # (B, backbone_dim, H', W')
        pooled = self._pool(features)  # (B, backbone_dim, 1, 1)
        global_feat = pooled.view(pooled.shape[0], -1)  # (B, backbone_dim)

        # Sub-pathway heads
        head_outputs = []
        for key in ["m_pathway", "p_pathway", "k_pathway",
                     "it", "sc", "pulvinar", "binding"]:
            head_outputs.append(self._heads[key](global_feat))

        return self._torch.cat(head_outputs, dim=-1)  # (B, 308)

    # ================================================================
    # Training (autoencoder reconstruction)
    # ================================================================

    def _train_step_impl(self, batch):
        """Single autoencoder training step.

        Args:
            batch: dict with 'input' (B, 3, H, W) float32 images [0,1]

        Returns:
            {'loss': float}
        """
        torch = _get_torch()

        x = batch["input"]

        # Forward through backbone (but not heads)
        features = self._backbone(x)  # (B, backbone_dim, 4, 4)

        # Decode
        reconstructed = self._decoder(features)  # (B, 3, 64, 64)

        # Ensure target matches reconstruction size
        if x.shape[-2:] != reconstructed.shape[-2:]:
            x_resized = torch.nn.functional.interpolate(
                x, size=reconstructed.shape[-2:], mode="bilinear",
                align_corners=False
            )
        else:
            x_resized = x

        # MSE reconstruction loss
        loss = torch.nn.functional.mse_loss(reconstructed, x_resized)

        # Backward
        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(
                self._net.parameters(),
                lr=self.config.effective_lr(self.config.visual_lr),
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
    # Utilities
    # ================================================================

    def get_subpathway_outputs(
        self, image: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Get individual sub-pathway outputs for inspection.

        Args:
            image: (H, W, 3) uint8 or float32

        Returns:
            Dict mapping sub-pathway name → numpy array
        """
        torch = _get_torch()
        preprocessed = self.preprocess_image(image)
        x = numpy_to_torch(preprocessed, device=self._device_str)

        self.eval()
        with torch.no_grad():
            features = self._backbone(x)
            pooled = self._pool(features)
            global_feat = pooled.view(pooled.shape[0], -1)

            outputs = {}
            for key in ["m_pathway", "p_pathway", "k_pathway",
                         "it", "sc", "pulvinar", "binding"]:
                outputs[key] = torch_to_numpy(
                    self._heads[key](global_feat)
                )[0]

        return outputs

    def __repr__(self) -> str:
        return (
            f"TrainableVisualEncoder(size={self.input_size}, "
            f"backbone_dim={self._backbone_dim}, "
            f"device={self._device_str}, trainable={self.trainable})"
        )
