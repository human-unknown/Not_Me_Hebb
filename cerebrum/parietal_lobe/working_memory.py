"""
working_memory.py — 视空间模板 (Visuospatial Sketchpad) [v6.0]

Brain area mapping:
  - Right inferior parietal cortex: visual temporary storage
  - Intraparietal sulcus (IPS): spatial position manipulation
  - Right ventrolateral PFC: active rehearsal

Function:
  - Temporary storage of visual/spatial information (max 4 chunks, ~2sec decay)
  - Spatial information operations (mental rotation, position comparison)
  - Active rehearsal maintenance (requires attentional resources)

Right-hemisphere dominant — symmetric with left-hemisphere phonological loop.
Part of Baddeley's multicomponent working memory model.

Reference:
  - Baddeley, A. D., & Hitch, G. (1974). Working memory.
  - D'Esposito, M., & Postle, B. R. (2015). The cognitive neuroscience of WM.
"""

import numpy as np
from typing import Optional, List, Tuple


class VisuospatialSketchpad:
    """Visuospatial sketchpad — temporary storage/manipulation of visual & spatial info.

    Capacity: ~4 visual chunks (fewer than phonological loop's 7±2)
    Decay: ~2 seconds without active rehearsal
    Operations: mental rotation, spatial relationship comparison
    """

    def __init__(self, max_chunks: int = 4, decay_rate: float = 0.3):
        self.max_chunks = max_chunks
        self.decay_rate = decay_rate

        # Visual store: [(visual_vector, spatial_tag, timestamp)]
        # visual_vector: (64,) visual features
        # spatial_tag: (32,) spatial position info
        self.visual_store: List[Tuple[np.ndarray, np.ndarray, float]] = []

        # Spatial workspace: (32,) current attentional focus position
        self.spatial_focus: np.ndarray = np.zeros(32, dtype=np.float32)

        # Stats
        self.n_operations: int = 0  # Spatial operations performed

    # ----- Storage -----

    def sketch(self, visual_input: np.ndarray,
               spatial_info: np.ndarray = None,
               timestamp: float = 0.0):
        """Store a visual chunk temporarily.

        Args:
            visual_input: (64,) visual feature vector
            spatial_info: (32,) spatial position info, None = default
            timestamp: timestamp for decay calculation
        """
        if len(visual_input) > 64:
            visual_input = visual_input[:64]

        if spatial_info is None:
            spatial_info = np.zeros(32, dtype=np.float32)
        elif len(spatial_info) > 32:
            spatial_info = spatial_info[:32]

        vis_vec = np.asarray(visual_input, dtype=np.float32).copy()
        sp_vec = np.asarray(spatial_info, dtype=np.float32).copy()

        # Capacity management: overflow → replace oldest chunk
        if len(self.visual_store) >= self.max_chunks:
            self.visual_store.pop(0)

        self.visual_store.append((vis_vec, sp_vec, timestamp))

    def refresh(self, timestamp: float):
        """Decay expired chunks and keep <= max_chunks.

        Simulates passive decay of chunks — information lost without rehearsal.
        """
        if not self.visual_store:
            return

        kept = []
        for vis, sp, ts in self.visual_store:
            age = timestamp - ts
            # Exponential decay: after ~2 sec, activation drops below 0.37
            if age < 5.0 and np.random.random() > (
                1.0 - np.exp(-self.decay_rate * age)):
                kept.append((vis, sp, ts))
        self.visual_store = kept

    # ----- Operations -----

    def mental_rotation(self, degrees: float,
                       chunk_index: int = 0) -> Optional[np.ndarray]:
        """Mental rotation: rotate a specified chunk's spatial representation.

        Simulates spatial manipulation in working memory.
        Rotation implemented via cyclic shift of spatial tag.

        Args:
            degrees: rotation angle
            chunk_index: index of chunk to rotate

        Returns:
            Rotated spatial vector, or None (chunk doesn't exist)
        """
        if chunk_index >= len(self.visual_store):
            return None

        vis, sp, ts = self.visual_store[chunk_index]
        shift = int((degrees / 360.0) * len(sp))
        rotated = np.roll(sp.copy(), shift)
        self.n_operations += 1
        return rotated

    def spatial_compare(self, target: np.ndarray) -> float:
        """Compare current spatial focus with target position.

        Args:
            target: (32,) target spatial position

        Returns:
            Cosine similarity [-1, 1]
        """
        if len(target) > 32:
            target = target[:32]
        denom = (np.linalg.norm(self.spatial_focus)
                 * np.linalg.norm(target) + 1e-8)
        return float(np.dot(self.spatial_focus, target) / denom)

    # ----- Query -----

    def get_active_visuals(self) -> List[np.ndarray]:
        """Return currently active visual chunks (sorted by recency)."""
        return [vis for vis, sp, ts in
                sorted(self.visual_store, key=lambda x: x[2], reverse=True)]

    @property
    def n_chunks(self) -> int:
        return len(self.visual_store)

    @property
    def is_full(self) -> bool:
        return self.n_chunks >= self.max_chunks

    def get_state(self) -> dict:
        return {
            'n_chunks': self.n_chunks,
            'max_chunks': self.max_chunks,
            'n_operations': self.n_operations,
            'spatial_focus_norm': float(np.linalg.norm(self.spatial_focus)),
        }
