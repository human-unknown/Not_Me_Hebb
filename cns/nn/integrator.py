"""
integrator.py — NNBridge: Agent ↔ Neural Network Integration Layer (v7.5 Phase F)

Serves as the single integration point between Agent and all NN modules (Phases A-E).
Agent holds one NNBridge instance and delegates NN concerns at well-defined hook points.

Design:
  - Lazy initialization: NN modules created only on first use (fast startup)
  - All NN errors caught — NN failures NEVER crash Agent
  - Dual-system: supplements, not replaces, Hebb modules
  - VTA/LC modulation: reads modulation signals, applies to NN params
  - Sleep consolidation: NREM → NN gradient updates, REM → emotional decay

Hook points in Agent:
  1. enhance_sensory(s)   — after sensory processing, before L0 learn
  2. record_step(F, ...)  — after FEP recording
  3. VTA modulation       — after VTA computes RPE
  4. LC modulation        — after LC computes NE
  5. sleep consolidation  — during NREM N3 / REM sleep

Usage:
    bridge = NNBridge(config=NNConfig(nn_enabled=True), agent=agent)
    bridge.enhance_sensory(s)
    bridge.record_step(F_total=0.3, valence=0.5, ...)
"""

import os
import logging
from typing import Optional, Dict, List, Any
import numpy as np

from cns.nn.config import NNConfig, DEFAULT_NN_CONFIG
from cns.nn.trainer import Trainer
from cns.nn.metrics import ExperienceTracker

logger = logging.getLogger(__name__)


class NNBridge:
    """Integration bridge between Agent and all NN modules.

    Lazy-initializes NN modules, Trainer, and ExperienceTracker.
    Provides hook methods that Agent calls at specific points in step().

    Attributes:
        config: NNConfig instance
        is_enabled: whether NN integration is active
        _current_lr_mult: latest NN learning rate multiplier from VTA
        _current_temperature: latest temperature from LC for generator
        _current_dropout: latest dropout from LC for comprehender
    """

    def __init__(self, config: Optional[NNConfig] = None, agent=None):
        """Initialize NNBridge.

        Modules are NOT created until first use (lazy init).
        This keeps Agent startup fast when NN is disabled.

        Args:
            config: NNConfig (None = use DEFAULT_NN_CONFIG)
            agent: Parent Agent instance (for accessing Hebb network, body, etc.)
        """
        self.config = config or DEFAULT_NN_CONFIG
        self.agent = agent
        self._enabled = self.config.nn_enabled
        self._modules: Dict[str, Any] = {}
        self._trainer: Optional[Trainer] = None
        self._tracker: Optional[ExperienceTracker] = None
        self._initialized = False

        # Modulation state (set by VTA/LC hooks in agent.step())
        self._current_lr_mult: float = 1.0
        self._current_temperature: float = 0.8
        self._current_dropout: float = 0.1

        # Blend ratio for sensory enhancement (grows with training)
        self._blend_ratio: float = 0.1
        self._total_train_steps: int = 0

        # v7.7: Dialogue buffer for batch training
        self._dialogue_buffer: list[tuple[str, str]] = []
        self._batch_size: int = 8  # Trigger batch train every N turns

    # ================================================================
    # Initialization
    # ================================================================

    def _ensure_init(self):
        """Lazy-initialize all NN modules. Called on first use."""
        if self._initialized or not self._enabled:
            return

        try:
            from cns.nn.text_encoder import TrainableTextEncoder
            from cns.nn.visual_encoder import TrainableVisualEncoder
            from cns.nn.audio_encoder import TrainableAudioEncoder
            from cns.nn.language_model import NeuralGenerator
            from cns.nn.comprehender import NeuralComprehender
            from cns.nn.angular_gyrus_nn import NeuralAngularGyrus
            from cns.nn.semantic_store import NeuralSemanticStore
            from cns.nn.crossmodal_nn import CrossModalNN

            # Phase B: Perception encoders
            self._modules['text_encoder'] = TrainableTextEncoder(
                config=self.config, trainable=True)
            self._modules['visual_encoder'] = TrainableVisualEncoder(
                config=self.config, trainable=True)
            self._modules['audio_encoder'] = TrainableAudioEncoder(
                config=self.config, trainable=True)

            # Phase C: Memory layer
            self._modules['semantic_store'] = NeuralSemanticStore(
                config=self.config,
                text_encoder=self._modules['text_encoder'])
            self._modules['crossmodal'] = CrossModalNN(
                config=self.config,
                text_encoder=self._modules['text_encoder'],
                visual_encoder=self._modules['visual_encoder'])

            # Phase D: Language system
            self._modules['generator'] = NeuralGenerator(
                config=self.config,
                text_encoder=self._modules['text_encoder'])
            self._modules['comprehender'] = NeuralComprehender(
                config=self.config,
                text_encoder=self._modules['text_encoder'],
                memory_store=self._modules['semantic_store'])
            self._modules['angular_gyrus_nn'] = NeuralAngularGyrus(
                config=self.config,
                text_encoder=self._modules['text_encoder'])

            # Phase E: Trainer + ExperienceTracker
            self._trainer = Trainer(config=self.config)
            for name, mod in self._modules.items():
                if hasattr(mod, '_net') and mod._net is not None:
                    self._trainer.register(mod)

            self._tracker = ExperienceTracker(config=self.config)

            self._initialized = True
            logger.info("[NNBridge] All NN modules initialized")

        except Exception as e:
            logger.warning(f"[NNBridge] Initialization failed: {e}")
            self._enabled = False  # Graceful fallback

    # ================================================================
    # Hook 1: Sensory Enhancement
    # ================================================================

    def enhance_sensory(self, s: np.ndarray) -> np.ndarray:
        """Optionally enhance sensory with NN encoders.

        Only active when nn_sensory_enhance=True in config.
        Blend ratio starts at 0.1 and grows with cumulative training steps
        (capped at 0.5 — Hebb still dominates).

        Args:
            s: Current sensory vector (D=516)

        Returns:
            Enhanced sensory vector (same shape)
        """
        if not self._enabled:
            return s

        self._ensure_init()
        if not self._initialized:
            return s

        if not self.config.nn_sensory_enhance:
            return s

        try:
            s_out = s.copy()

            # Text segment [0:64] — use NN text encoder if text is non-zero
            text_vec = s[0:64].copy().reshape(1, -1).astype(np.float32)
            if np.linalg.norm(text_vec) > 0.01:
                try:
                    nn_text = self._modules['text_encoder'].forward(
                        text_vec.reshape(1, 64))
                    if nn_text is not None and nn_text.size == 64:
                        nn_text = nn_text.ravel().astype(np.float32)
                        s_out[0:64] = (
                            (1.0 - self._blend_ratio) * s_out[0:64]
                            + self._blend_ratio * nn_text
                        )
                except Exception:
                    pass  # Text encoder not yet trained — skip

            return s_out.astype(np.float32)

        except Exception:
            # NN enhancement failure → return original sensory
            return s

    # ================================================================
    # Hook 2: Metrics Recording
    # ================================================================

    def record_step(self, **metrics):
        """Record per-step metrics to ExperienceTracker.

        Called after FEP computation in agent.step().
        Args are flexible — any metric key-value pairs.

        Common keys: F_total, F_body, F_social, F_cognitive, F_accuracy,
        valence, arousal, step_count
        """
        if not self._enabled:
            return

        self._ensure_init()
        if not self._initialized:
            return

        try:
            self._tracker.record_step(**metrics)
        except Exception:
            pass

    def record_dialogue(self, user_text: str, response: str, **metrics):
        """Record a dialogue turn.

        Args:
            user_text: User input
            response: Agent response
            **metrics: Additional metrics
        """
        if not self._enabled:
            return

        self._ensure_init()
        if not self._initialized:
            return

        try:
            self._tracker.record_dialogue_turn(user_text, response, metrics)
        except Exception:
            pass

    def train_on_dialogue_turn(self, user_text: str, response: str):
        """v7.6: Feed a dialogue turn as NN training sample.

        This is the FIRST actual NN training activation point —
        called after each dialogue response. Uses low online LR (1e-4)
        to gently adapt NN modules without catastrophic forgetting.

        Args:
            user_text: User's input text
            response: Agent's response text
        """
        if not self._enabled:
            return

        self._ensure_init()
        if not self._initialized:
            return

        try:
            combined = (user_text + " " + response).strip()
            if len(combined) < 4:
                return

            # v7.7: Accumulate in buffer → batch train when full
            self._dialogue_buffer.append((user_text, response))
            if len(self._dialogue_buffer) >= self._batch_size:
                self._train_on_buffer()

            # Single-sample online train (always — supplements batch)
            text_mod = self._modules.get('text_encoder')
            if text_mod is not None and text_mod.trainable:
                try:
                    text_mod.train_step({'text': combined})
                except Exception:
                    pass

            gen_mod = self._modules.get('generator')
            if gen_mod is not None and gen_mod.trainable and len(response) >= 2:
                try:
                    gen_mod.train_step({'text': response})
                except Exception:
                    pass

            comp_mod = self._modules.get('comprehender')
            if comp_mod is not None and comp_mod.trainable:
                try:
                    comp_mod.train_step({
                        'input_text': user_text,
                        'target_text': response,
                    })
                except Exception:
                    pass

            self._total_train_steps += 1
            self._blend_ratio = min(0.5, 0.1 + 0.005 * self._total_train_steps)

        except Exception:
            pass  # NN training failure → silent, never block dialogue

    def _train_on_buffer(self):
        """v7.7: Batch train on accumulated dialogue history.

        Called automatically when _dialogue_buffer reaches _batch_size.
        Runs 2-3 gradient steps on all samples — consolidates learning
        from recent dialogue turns into NN weights.
        """
        if not self._trainer or len(self._dialogue_buffer) < 2:
            return

        try:
            # Combine buffer text for text encoder
            all_text = " ".join(
                u + " " + r for u, r in self._dialogue_buffer)
            text_mod = self._modules.get('text_encoder')
            if text_mod is not None and text_mod.trainable:
                for _ in range(2):  # 2 epochs on buffer
                    try:
                        text_mod.train_step({'text': all_text})
                    except Exception:
                        pass

            # Train generator on each response in buffer
            gen_mod = self._modules.get('generator')
            if gen_mod is not None and gen_mod.trainable:
                for _, resp in self._dialogue_buffer:
                    if len(resp) >= 2:
                        try:
                            gen_mod.train_step({'text': resp})
                        except Exception:
                            pass

            logger.info("[NNBridge] Batch trained on %d dialogue turns",
                       len(self._dialogue_buffer))
        except Exception:
            pass
        finally:
            self._dialogue_buffer = []  # Clear buffer after training

    # ================================================================
    # Hook 3: VTA → NN Learning Rate Modulation
    # ================================================================

    def get_nn_lr_modulation(self, rpe: float, da: float) -> float:
        """Compute NN learning rate multiplier from VTA signals.

        Damped version of Hebb LR modulation — NN learns more conservatively.

        Args:
            rpe: Reward prediction error from VTA
            da: Total dopamine level from VTA

        Returns:
            NN learning rate multiplier [0.2, 2.0]
        """
        if not self._enabled:
            return 1.0

        # Same mechanism as VTA but damped by 0.7x
        da_baseline = 0.30
        da_deviation = da - da_baseline

        if da_deviation > 0:
            lr_mult = 1.0 + 1.4 * (da_deviation / 0.7)  # [1.0, ~2.0]
        else:
            lr_mult = 1.0 - 0.8 * (abs(da_deviation) / 0.25)  # [~0.2, 1.0]

        nn_mult = float(np.clip(lr_mult * 0.7, 0.2, 2.0))
        self._current_lr_mult = nn_mult
        return nn_mult

    # ================================================================
    # Hook 4: LC → NN Explore/Exploit (temperature, dropout)
    # ================================================================

    def get_nn_explore_params(
        self,
        tonic_ne: float,
        total_ne: float,
        exploration_bias: float,
    ) -> Dict[str, float]:
        """Compute NN temperature and dropout from LC NE signals.

        High NE (exploit) → low temperature (deterministic generation)
        Low NE (explore) → high temperature (diverse generation)
        Low Yerkes-Dodson → high dropout (uncertainty)

        Args:
            tonic_ne: Tonic NE level from LC
            total_ne: Total NE level from LC
            exploration_bias: Explore/exploit bias from LC [-1, 1]

        Returns:
            {'temperature': float, 'dropout': float}
        """
        if not self._enabled:
            return {'temperature': 0.8, 'dropout': 0.1}

        # Temperature: exploit (positive bias) → low temp, explore → high temp
        temperature = float(np.clip(
            0.5 + 0.8 * (1.0 - abs(exploration_bias)),
            0.3, 1.5))

        # Dropout: based on distance from optimal NE (Yerkes-Dodson)
        optimal_ne = 0.40
        yd_distance = abs(total_ne - optimal_ne)
        dropout = float(np.clip(
            0.1 + 0.3 * yd_distance / 0.3,
            0.05, 0.5))

        self._current_temperature = temperature
        self._current_dropout = dropout

        # Apply to NN modules if initialized
        if self._initialized:
            try:
                gen = self._modules.get('generator')
                if gen is not None and hasattr(gen, '_net') and gen._net is not None:
                    # Update generator temperature (used during generation)
                    if hasattr(gen, 'temperature'):
                        gen.temperature = temperature
                    # Update dropout in the network
                    if hasattr(gen._net, 'dropout'):
                        gen._net.dropout = dropout
            except Exception:
                pass

            try:
                comp = self._modules.get('comprehender')
                if comp is not None and hasattr(comp, '_net') and comp._net is not None:
                    if hasattr(comp._net, 'dropout'):
                        comp._net.dropout = dropout
            except Exception:
                pass

        return {'temperature': temperature, 'dropout': dropout}

    # ================================================================
    # Hook 5: Sleep Consolidation
    # ================================================================

    def sleep_nrem_consolidation(self):
        """Run NN gradient updates on replayed patterns during NREM.

        Uses top Hebb clusters as replay patterns for NN modules.
        This bridges Hebb episodic memory → NN semantic learning.
        Only runs 1-2 gradient steps to keep sleep processing fast.
        """
        if not self._enabled:
            return

        self._ensure_init()
        if not self._initialized:
            return

        if self._trainer is None:
            return

        try:
            # Collect replay patterns from top Hebb clusters
            if self.agent is None or not hasattr(self.agent, 'net'):
                return

            hebb_net = self.agent.net
            if hebb_net.n_clusters == 0:
                return

            # Top 5 most active clusters → replay patterns
            sorted_clusters = sorted(
                hebb_net.clusters,
                key=lambda c: c.activation,
                reverse=True,
            )
            top_n = min(5, len(sorted_clusters))

            for i, cluster in enumerate(sorted_clusters[:top_n]):
                centroid = cluster.centroid.copy()

                # Text segment → text encoder replay
                text_vec = centroid[0:64].astype(np.float32).reshape(1, -1)
                if np.linalg.norm(text_vec) > 0.01:
                    try:
                        text_mod = self._modules.get('text_encoder')
                        if text_mod is not None and text_mod.trainable:
                            text_mod.train_step({
                                'input': text_vec.reshape(1, 64),
                            })
                    except Exception:
                        pass

                # Generator replay: use text as context for next-char prediction
                try:
                    gen_mod = self._modules.get('generator')
                    if gen_mod is not None and gen_mod.trainable:
                        gen_mod.train_step({
                            'input': text_vec.reshape(1, 64),
                        })
                except Exception:
                    pass

            # Increment blend ratio as NN gains training
            self._total_train_steps += 1
            self._blend_ratio = min(0.5, 0.1 + 0.005 * self._total_train_steps)

        except Exception:
            pass  # NN sleep consolidation failure → silent continue

    def sleep_rem_consolidation(self):
        """Decay emotional bias in NN generator during REM.

        Implements "emotional detoxification" of REM sleep:
        High-valence associations are slightly dampened to prevent
        emotional overfitting of the generator.
        """
        if not self._enabled:
            return

        self._ensure_init()
        if not self._initialized:
            return

        try:
            gen = self._modules.get('generator')
            if gen is None:
                return

            # Gentle decay of emotional token bias (if the generator tracks it)
            if hasattr(gen, '_emotion_bias'):
                gen._emotion_bias *= 0.95  # 5% decay per REM cycle
                gen._emotion_bias = float(np.clip(gen._emotion_bias, -1.0, 1.0))

            # Also decay the comprehender's N400 EMA (emotional reset)
            comp = self._modules.get('comprehender')
            if comp is not None and hasattr(comp, '_n400_ema'):
                comp._n400_ema *= 0.9

        except Exception:
            pass

    # ================================================================
    # Checkpoint (Save/Load)
    # ================================================================

    def save_checkpoint(self, directory: str) -> bool:
        """Save all NN modules + trainer state.

        Args:
            directory: Target directory

        Returns:
            True if saved successfully
        """
        if not self._enabled:
            return False

        self._ensure_init()
        if not self._initialized:
            return False

        try:
            if self._trainer is not None:
                self._trainer.save_checkpoint(directory)

            # Also save tracker
            if self._tracker is not None:
                tracker_path = os.path.join(directory, "tracker.json")
                self._tracker.to_json(tracker_path)

            logger.info(f"[NNBridge] Checkpoint saved to {directory}")
            return True
        except Exception as e:
            logger.warning(f"[NNBridge] Save failed: {e}")
            return False

    def load_checkpoint(self, directory: str) -> bool:
        """Load all NN modules + trainer state.

        Args:
            directory: Source directory

        Returns:
            True if loaded successfully
        """
        if not self._enabled:
            return False

        self._ensure_init()
        if not self._initialized:
            return False

        try:
            if self._trainer is not None:
                self._trainer.load_checkpoint(directory)

            # Load tracker
            tracker_path = os.path.join(directory, "tracker.json")
            if os.path.exists(tracker_path) and self._tracker is not None:
                from cns.nn.metrics import ExperienceTracker
                self._tracker = ExperienceTracker.from_json(
                    tracker_path, config=self.config)

            logger.info(f"[NNBridge] Checkpoint loaded from {directory}")
            return True
        except Exception as e:
            logger.warning(f"[NNBridge] Load failed: {e}")
            return False

    # ================================================================
    # Status (for Web UI)
    # ================================================================

    def get_status(self) -> Dict[str, Any]:
        """Return brief status dict for Web UI _build_status()."""
        if not self._enabled:
            return {'enabled': False}

        status = {
            'enabled': True,
            'initialized': self._initialized,
            'blend_ratio': self._blend_ratio,
            'total_train_steps': self._total_train_steps,
            'current_lr_mult': self._current_lr_mult,
            'current_temperature': self._current_temperature,
            'current_dropout': self._current_dropout,
        }

        if self._initialized:
            status['modules'] = list(self._modules.keys())
            if self._tracker is not None:
                status['tracker'] = {
                    'n_records': self._tracker.n_records,
                    'n_turns': self._tracker.turn_count,
                    'vocab_size': self._tracker.vocab_size,
                }

        return status

    def get_training_summary(self) -> Dict[str, Any]:
        """Return training summary for Web UI."""
        if not self._initialized or self._trainer is None:
            return {}

        return self._trainer.get_training_history()

    # ================================================================
    # Properties
    # ================================================================

    @property
    def is_enabled(self) -> bool:
        return self._enabled and self._initialized

    @property
    def tracker(self) -> Optional[ExperienceTracker]:
        return self._tracker

    @property
    def trainer(self) -> Optional[Trainer]:
        return self._trainer

    def __repr__(self) -> str:
        return (
            f"NNBridge(enabled={self._enabled}, "
            f"initialized={self._initialized}, "
            f"modules={list(self._modules.keys()) if self._initialized else []}, "
            f"blend={self._blend_ratio:.2f})"
        )
