"""
audio_encoder.py — TrainableAudioEncoder (v7.0 Phase B)

Small CNN on Mel spectrogram for auditory feature extraction.

Replaces: Mel spectrum → CochlearNuc→SOC→IC→AuditoryCortex hand-crafted pipeline
With:     Mel spectrogram → shared CNN backbone + 4 sub-module heads

Architecture:
  Raw audio → Mel spectrogram (n_mels=64, fixed_time_bins=128)
  → Shared CNN backbone (3 conv blocks: 1→32→64→128)
  → AdaptiveAvgPool2d → (128,) global feature
  → 4 parallel projection heads → concatenate → (96,)

Sub-modules (matching AudioEncoder.SUBMODULES):
  cochlear_nucleus: 128→32  (tonotopic spectrum)
  soc:              128→24  (binaural spatial — ITD + ILD)
  ic:               128→24  (integrated frequency×space×time)
  auditory_cortex:  128→16  (auditory objects/scene)

Training: Mel spectrogram reconstruction (encode→decode→MSE)

Mel computation uses scipy.signal (fallback if torchaudio unavailable).
"""

import logging
from typing import Optional, List, Dict
import numpy as np

from cns.nn.base import NeuralModule
from cns.nn.config import NNConfig
from cns.nn.interfaces import AudioEncoder
from cns.nn.bridge import _get_torch, numpy_to_torch, torch_to_numpy

logger = logging.getLogger(__name__)


def _compute_mel_numpy(
    audio: np.ndarray,
    sample_rate: int,
    n_mels: int = 64,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> np.ndarray:
    """Compute mel spectrogram using scipy (no torchaudio needed).

    Args:
        audio: (samples,) float32 waveform
        sample_rate: Sample rate in Hz
        n_mels: Number of mel bands
        n_fft: FFT window size
        hop_length: Hop length between frames

    Returns:
        (n_mels, n_frames) mel spectrogram (power, not dB)
    """
    try:
        from scipy.signal import spectrogram
        from scipy.fft import rfft

        # Compute spectrogram
        _, _, Sxx = spectrogram(
            audio,
            fs=sample_rate,
            nperseg=n_fft,
            noverlap=n_fft - hop_length,
            nfft=n_fft,
            scaling="spectrum",
        )
        # Sxx: (n_freq_bins, n_frames), n_freq_bins = n_fft//2 + 1

        # Simple mel filterbank
        n_freq_bins = Sxx.shape[0]
        mel_filters = _create_mel_filterbank(
            n_mels, n_freq_bins, sample_rate, n_fft
        )

        # Apply mel filterbank
        mel_spec = mel_filters @ Sxx  # (n_mels, n_frames)

        # Add small epsilon to avoid log(0)
        mel_spec = np.maximum(mel_spec, 1e-10)

        return mel_spec.astype(np.float32)

    except ImportError:
        # Fallback: simple STFT-based approximation
        logger.warning(
            "scipy not available for mel computation; "
            "using simple magnitude spectrum"
        )
        n_fft_real = n_fft
        n_frames = max(1, (len(audio) - n_fft_real) // hop_length + 1)
        spec = np.zeros((n_mels, n_frames), dtype=np.float32)
        for i in range(n_frames):
            start = i * hop_length
            frame = audio[start:start + n_fft_real]
            if len(frame) < n_fft_real:
                frame = np.pad(frame, (0, n_fft_real - len(frame)))
            frame = frame * np.hanning(len(frame))
            mag = np.abs(np.fft.rfft(frame, n=n_fft_real))
            # Downsample to n_mels via averaging
            mag_down = np.array([
                mag[j * len(mag) // n_mels:(j + 1) * len(mag) // n_mels].mean()
                for j in range(n_mels)
            ])
            spec[:, i] = mag_down
        return np.maximum(spec, 1e-10).astype(np.float32)


def _create_mel_filterbank(
    n_mels: int, n_freq_bins: int, sample_rate: int, n_fft: int
) -> np.ndarray:
    """Create a mel filterbank matrix.

    Args:
        n_mels: Number of mel bands
        n_freq_bins: Number of frequency bins
        sample_rate: Sample rate in Hz
        n_fft: FFT size

    Returns:
        (n_mels, n_freq_bins) filterbank matrix
    """
    # Mel scale conversion
    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    low_mel = hz_to_mel(0)
    high_mel = hz_to_mel(sample_rate / 2)

    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    # Convert to FFT bin indices
    bin_indices = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    bin_indices = np.clip(bin_indices, 0, n_freq_bins - 1)

    # Create filterbank
    filters = np.zeros((n_mels, n_freq_bins), dtype=np.float32)
    for i in range(n_mels):
        start, center, end = bin_indices[i], bin_indices[i + 1], bin_indices[i + 2]
        if start >= end:
            if center < n_freq_bins:
                filters[i, center] = 1.0
            continue
        # Rising ramp
        if center > start:
            filters[i, start:center] = np.linspace(0, 1, center - start)
        # Falling ramp
        if end > center:
            filters[i, center:end] = np.linspace(1, 0, end - center)

    return filters


class TrainableAudioEncoder(AudioEncoder):
    """Small CNN on Mel spectrogram with 4 sub-module heads.

    Produces 96-dim auditory feature vectors matching D=516 audio channels.

    Usage:
        encoder = TrainableAudioEncoder()
        vec = encoder.encode(audio_waveform, sample_rate=16000)  # → (96,) float32
    """

    def __init__(
        self,
        config: Optional[NNConfig] = None,
        trainable: bool = True,
        n_mels: int = 64,
        n_fft: int = 1024,
        hop_length: int = 256,
        fixed_time_bins: int = 128,
    ):
        """Initialize the audio encoder.

        Args:
            config: NNConfig (device, dtype, etc.)
            trainable: Whether weights are trainable
            n_mels: Number of mel frequency bands
            n_fft: FFT window size
            hop_length: Hop length between frames
            fixed_time_bins: Time bins after resizing (fixed input to CNN)
        """
        super().__init__(config=config, trainable=trainable)
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fixed_time_bins = fixed_time_bins

    # ================================================================
    # Build network
    # ================================================================

    def _build_network(self):
        torch = _get_torch()

        # --- Shared CNN backbone ---
        self._backbone = torch.nn.Sequential(
            # Block 1: 1 → 32
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d((2, 2)),  # (64,128)→(32,64)

            # Block 2: 32 → 64
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d((2, 2)),  # (32,64)→(16,32)

            # Block 3: 64 → 128
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d((2, 2)),  # (16,32)→(8,16)
        )

        self._backbone_dim = 128
        self._pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        # --- 4 sub-module projection heads ---
        self._heads = torch.nn.ModuleDict({
            "cochlear_nucleus": torch.nn.Sequential(
                torch.nn.Linear(self._backbone_dim, 32),
                torch.nn.LayerNorm(32),
            ),
            "soc": torch.nn.Sequential(
                torch.nn.Linear(self._backbone_dim, 24),
                torch.nn.LayerNorm(24),
            ),
            "ic": torch.nn.Sequential(
                torch.nn.Linear(self._backbone_dim, 24),
                torch.nn.LayerNorm(24),
            ),
            "auditory_cortex": torch.nn.Sequential(
                torch.nn.Linear(self._backbone_dim, 16),
                torch.nn.LayerNorm(16),
            ),
        })

        # --- Decoder for autoencoder training ---
        self._decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                128, 64, kernel_size=4, stride=2, padding=1
            ),  # (8,16)→(16,32)
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(
                64, 32, kernel_size=4, stride=2, padding=1
            ),  # (16,32)→(32,64)
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(
                32, 1, kernel_size=4, stride=2, padding=1
            ),  # (32,64)→(64,128)
            torch.nn.Sigmoid(),
        )

        # Wrap in ModuleDict
        self._net = torch.nn.ModuleDict({
            "backbone": self._backbone,
            "heads": self._heads,
            "decoder": self._decoder,
        })

    # ================================================================
    # Mel spectrogram computation
    # ================================================================

    def compute_mel(
        self, audio: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """Compute mel spectrogram from raw audio waveform.

        Args:
            audio: (samples,) float32 waveform (mono)
            sample_rate: Sample rate in Hz

        Returns:
            (1, 1, n_mels, fixed_time_bins) float32 mel spectrogram
        """
        # Ensure mono
        if audio.ndim > 1:
            audio = audio.mean(axis=-1) if audio.shape[-1] <= 2 else audio[:, 0]

        audio = audio.astype(np.float32).ravel()

        # Normalize
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val

        # Compute mel spectrogram
        mel_spec = _compute_mel_numpy(
            audio,
            sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        # mel_spec: (n_mels, n_frames)

        # Resize time axis to fixed length
        n_frames = mel_spec.shape[1]
        if n_frames > self.fixed_time_bins:
            # Center crop
            start = (n_frames - self.fixed_time_bins) // 2
            mel_spec = mel_spec[:, start:start + self.fixed_time_bins]
        elif n_frames < self.fixed_time_bins:
            # Pad with zeros
            padded = np.zeros(
                (self.n_mels, self.fixed_time_bins), dtype=np.float32
            )
            start = (self.fixed_time_bins - n_frames) // 2
            padded[:, start:start + n_frames] = mel_spec
            mel_spec = padded

        # Shape for CNN: (1, 1, n_mels, time)
        mel_spec = mel_spec[np.newaxis, np.newaxis, :, :]  # (1, 1, 64, 128)
        return mel_spec.astype(np.float32)

    # ================================================================
    # Core interface
    # ================================================================

    def encode(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> np.ndarray:
        """Encode an audio waveform to a 96-dim auditory feature vector.

        Args:
            audio: (samples,) float32 waveform (mono)
            sample_rate: Sample rate in Hz

        Returns:
            (96,) float32 vector
        """
        mel = self.compute_mel(audio, sample_rate)
        result = self.forward(mel)
        return result[0]  # (96,)

    def encode_batch(
        self,
        audios: List[np.ndarray],
        sample_rates: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Encode a batch of audio clips.

        Args:
            audios: List of (samples,) audio arrays
            sample_rates: List of sample rates (default: all 16000)

        Returns:
            (B, 96) float32
        """
        if sample_rates is None:
            sample_rates = [16000] * len(audios)

        batch_mels = []
        for audio, sr in zip(audios, sample_rates):
            mel = self.compute_mel(audio, sr)
            batch_mels.append(mel)

        batch = np.concatenate(batch_mels, axis=0)
        return self.forward(batch)

    # ================================================================
    # Forward implementation
    # ================================================================

    def _forward_impl(self, x):
        """Forward pass — tensor in, tensor out.

        Args:
            x: (B, 1, n_mels, fixed_time_bins) float32 tensor

        Returns:
            (B, 96) float32 tensor
        """
        # Backbone
        features = self._backbone(x)  # (B, backbone_dim, H', W')
        pooled = self._pool(features)  # (B, backbone_dim, 1, 1)
        global_feat = pooled.view(pooled.shape[0], -1)  # (B, backbone_dim)

        # Sub-module heads
        head_outputs = []
        for key in ["cochlear_nucleus", "soc", "ic", "auditory_cortex"]:
            head_outputs.append(self._heads[key](global_feat))

        return self._torch.cat(head_outputs, dim=-1)  # (B, 96)

    # ================================================================
    # Training (mel spectrogram reconstruction)
    # ================================================================

    def _train_step_impl(self, batch):
        """Single autoencoder training step.

        Args:
            batch: dict with 'input' (B, 1, n_mels, time) mel spectrograms

        Returns:
            {'loss': float}
        """
        torch = _get_torch()

        x = batch["input"]

        # Forward through backbone
        features = self._backbone(x)
        reconstructed = self._decoder(features)

        # Ensure target matches reconstruction size
        if x.shape[-2:] != reconstructed.shape[-2:]:
            x_resized = torch.nn.functional.interpolate(
                x, size=reconstructed.shape[-2:],
                mode="bilinear", align_corners=False
            )
        else:
            x_resized = x

        loss = torch.nn.functional.mse_loss(reconstructed, x_resized)

        # Backward
        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(
                self._net.parameters(),
                lr=self.config.effective_lr(self.config.audio_lr),
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

    def get_submodule_outputs(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> Dict[str, np.ndarray]:
        """Get individual sub-module outputs for inspection.

        Args:
            audio: (samples,) float32 waveform
            sample_rate: Sample rate in Hz

        Returns:
            Dict mapping sub-module name → numpy array
        """
        torch = _get_torch()
        mel = self.compute_mel(audio, sample_rate)
        x = numpy_to_torch(mel, device=self._device_str)

        self.eval()
        with torch.no_grad():
            features = self._backbone(x)
            pooled = self._pool(features)
            global_feat = pooled.view(pooled.shape[0], -1)

            outputs = {}
            for key in ["cochlear_nucleus", "soc", "ic", "auditory_cortex"]:
                outputs[key] = torch_to_numpy(
                    self._heads[key](global_feat)
                )[0]

        return outputs

    def __repr__(self) -> str:
        return (
            f"TrainableAudioEncoder(n_mels={self.n_mels}, "
            f"time_bins={self.fixed_time_bins}, "
            f"backbone_dim={self._backbone_dim}, "
            f"device={self._device_str}, trainable={self.trainable})"
        )
