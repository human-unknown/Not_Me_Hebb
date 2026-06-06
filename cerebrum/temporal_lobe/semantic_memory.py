"""
semantic_memory.py — 语义记忆系统 (Semantic Memory System)

Brain area mapping:
  - Anterior Temporal Lobe: cross-modal concept integration, semantic hub
  - Angular Gyrus (BA39): distributed storage of semantic knowledge
  - Inferior Parietal + Inferior Frontal: semantic retrieval pathways

Function:
  - Stores "knowing that" knowledge (vs episodic "remembering when")
  - Extracts gist knowledge from repeated episodic experiences
  - Slow learning rate → requires multiple repetitions to consolidate
  - Slow decay rate → once formed, hard to forget
  - Large capacity → can store thousands of concepts/facts

Key differences from hippocampal episodic memory:
  ┌──────────────┬─────────────────────┬─────────────────────┐
  │              │ Episodic (Hippo)    │ Semantic (Cortical) │
  ├──────────────┼─────────────────────┼─────────────────────┤
  │ Learn rate   │ Fast (lr=0.05)      │ Slow (lr=0.01)      │
  │ Decay rate   │ Fast (decay=0.02)   │ Slow (decay=0.003)  │
  │ Capacity     │ 512 clusters        │ 1024 clusters       │
  │ Storage      │ Full 516-dim        │ 64-dim gist vector  │
  │ Threshold    │ Medium (0.70)       │ Low (0.45)          │
  │ Experience   │ "I remember..."     │ "I know..."         │
  └──────────────┴─────────────────────┴─────────────────────┘

Reference:
  - Binder & Desai (2011). The neurobiology of semantic memory.
  - Patterson et al. (2007). Where do you know what you know?
"""

import numpy as np
from typing import Optional, List, Tuple
from cns.data_types import D, Theta, Cluster
from cerebrum.limbic_system.hippocampus import (
    ClusterNetwork, _masked_cosine, _auto_mask,
)


# Semantic memory specific parameters
SEMANTIC_THETA_DEFAULTS = {
    'cluster_threshold': 0.45,   # Low threshold — similar concepts merge (wider than episodic)
    'learn_rate_l0': 0.01,       # Slow learning — needs repeated exposure (5x slower than episodic)
    'decay_rate': 0.003,         # Slow forgetting — knowledge once formed persists
}


class SemanticMemory:
    """Cortical semantic memory — independent ClusterNetwork for "knowing" type knowledge.

    Uses the same ClusterNetwork recipe as hippocampal episodic memory,
    but with different parameter tuning: slow learn, slow decay, low threshold, large capacity.

    Storage format (D=516 dim centroid):
      centroid[:64]   = text gist    — concept/fact semantic encoding
      centroid[64:72] = body_snap    — body state snapshot at encoding time
      centroid[72:80] = emotion      — valence/arousal/emotional tags
      centroid[80:]   = reserved     — cross-modal links (vision/audio/pain simplified to 0)
    """

    def __init__(self, max_clusters: int = 1024):
        theta = Theta(**SEMANTIC_THETA_DEFAULTS)
        self.net = ClusterNetwork(theta, hash_offset=0)
        self.max_clusters = max_clusters
        self.n_facts: int = 0              # Cumulative facts/concepts stored
        self.consolidation_count: int = 0   # Number of system consolidations performed

    # ----- Gist Extraction -----

    def extract_gist(self, episodic_centroid: np.ndarray) -> np.ndarray:
        """Extract gist (64-dim semantic) from full episodic memory (516-dim).

        Keeps only: text[0:64] + body_snap[64:72] + emotion[72:80]
        Discards: visual detail[64:372], audio detail[372:468], pain detail[468:516]

        This simulates the process in memory consolidation where concrete details
        fade while the gist is preserved.
        """
        gist = np.zeros(D, dtype=np.float32)
        # Text gist
        gist[:64] = episodic_centroid[:64]
        # Body snapshot (compressed to 8 dims)
        if len(episodic_centroid) > 64:
            body_slice = episodic_centroid[64:72]
            gist[64:72] = body_slice[:8]
        # Emotion snapshot
        if len(episodic_centroid) > 80:
            gist[72:80] = episodic_centroid[72:80]
        # Remaining dims stay zero (channel detail not encoded)
        return gist

    # ----- Learn / Store -----

    def learn_fact(self, fact_vec: np.ndarray, weight: float = 1.0) -> Cluster:
        """Learn a fact/concept.

        Uses low learning rate — a single exposure is NOT enough to form
        stable semantic knowledge. Requires repeated exposure or high weight.

        Args:
            fact_vec: (D,) full-dimension vector (already gist-extracted or direct)
            weight: learning weight (>1 = reinforce, <1 = weaken)

        Returns:
            Created or updated Cluster
        """
        orig_lr = self.net.theta.learn_rate_l0
        self.net.theta.learn_rate_l0 = orig_lr * weight
        c = self.net.learn(fact_vec)
        self.net.theta.learn_rate_l0 = orig_lr
        self.n_facts += 1
        return c

    # ----- Query -----

    def query(self, query_vec: np.ndarray,
              top_k: int = 5) -> List[Tuple[Cluster, float]]:
        """Query semantic knowledge.

        Args:
            query_vec: (D,) query vector (typically only text[0:64] segment active)
            top_k: return top k matches

        Returns:
            [(cluster, similarity), ...] sorted by similarity descending
        """
        if self.net.n_clusters == 0:
            return []

        if len(query_vec) < D:
            padded = np.zeros(D, dtype=np.float32)
            padded[:len(query_vec)] = query_vec
            query_vec = padded

        mask = _auto_mask(query_vec)
        h = self.net.hash_features(query_vec)

        # Scan all clusters (cross-bucket query — concepts may be in different buckets)
        scored = []
        for c in self.net.clusters:
            sim = _masked_cosine(h, c.centroid, mask)
            if sim > 0.15:  # Very low threshold — any weak association is returned
                scored.append((c, float(sim)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def knows_about(self, query_vec: np.ndarray) -> float:
        """Return Agent's "familiarity" with a topic [0, 1].

        0 = completely unknown, 1 = very familiar (multiple strong associations).
        """
        results = self.query(query_vec, top_k=3)
        if not results:
            return 0.0
        # Familiarity = weighted sum of top-3 similarities
        familiarity = sum(sim * (0.7 ** i) for i, (_, sim)
                         in enumerate(results))
        return float(np.clip(familiarity / 1.5, 0.0, 1.0))

    # ----- Systems Consolidation: Episodic → Semantic Extraction -----

    def consolidate_from_episodic(self,
                                   episodic_net: ClusterNetwork,
                                   n_top: int = 30,
                                   min_activation: float = 0.1,
                                   body_vec=None,
                                   valence: float = 0.0,
                                   arousal: float = 0.0
                                   ) -> dict:
        """Extract gist from hippocampal episodic memory → store in semantic memory.

        This is the core operation of systems consolidation:
        converting concrete experiences into abstract knowledge.
        Only processes the top N most active episodic clusters to control cost.

        Args:
            episodic_net: Hippocampal ClusterNetwork
            n_top: max episodic clusters to process
            min_activation: minimum activation threshold
            body_vec: current body vector (for emotional gating)
            valence: current valence
            arousal: current arousal

        Returns:
            dict: {n_processed, n_new, n_updated, n_skipped}
        """
        if episodic_net.n_clusters == 0:
            return {'n_processed': 0, 'n_new': 0, 'n_updated': 0, 'n_skipped': 0}

        # Select most active clusters
        candidates = sorted(
            [c for c in episodic_net.clusters
             if c.activation > min_activation],
            key=lambda c: c.activation, reverse=True,
        )[:n_top]

        stats = {'n_processed': 0, 'n_new': 0, 'n_updated': 0, 'n_skipped': 0}

        for c in candidates:
            stats['n_processed'] += 1
            # Emotional gating: high-arousal episodes → more likely to be semanticized
            # Amygdala emotional tagging: strongly emotional memories more easily become long-term knowledge
            emotion_salience = 0.3 + 0.7 * abs(valence) * arousal
            if emotion_salience < 0.2 and stats['n_processed'] > 10:
                stats['n_skipped'] += 1
                continue

            # Extract gist
            gist = self.extract_gist(c.centroid)

            # Inject current emotional/body context
            if body_vec is not None and len(body_vec.b) >= 8:
                gist[64:72] = body_vec.b[:8].astype(np.float32)
            gist[72] = valence
            gist[73] = arousal

            # Check if similar knowledge already exists
            existing = self.net.recall(gist)
            if existing is not None:
                # Update existing knowledge (slow, semantic memory needs multiple confirmations)
                orig_lr = self.net.theta.learn_rate_l0
                # More established knowledge is harder to change
                stability = min(1.0, existing.count / 20.0)
                self.net.theta.learn_rate_l0 = orig_lr * (1.0 - stability * 0.8)
                self.net.learn(gist)
                self.net.theta.learn_rate_l0 = orig_lr
                stats['n_updated'] += 1
            else:
                # New knowledge — higher weight on first extraction
                self.learn_fact(gist, weight=1.5)
                stats['n_new'] += 1

        self.consolidation_count += 1
        return stats

    # ----- Stats -----

    @property
    def n_clusters(self) -> int:
        return self.net.n_clusters

    @property
    def total_activation(self) -> float:
        return self.net.total_activation

    def get_state(self) -> dict:
        """Return semantic memory state summary."""
        top = self.net.get_top_clusters(5)
        return {
            'n_clusters': self.n_clusters,
            'n_facts': self.n_facts,
            'consolidation_count': self.consolidation_count,
            'total_activation': self.total_activation,
            'top_clusters': [
                {'activation': float(c.activation),
                 'count': c.count, 'age': c.age}
                for c in top
            ],
        }
