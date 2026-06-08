"""
trainer.py — Unified Training Orchestrator (v7.4 Phase E)

Replaces duplicated pretrain() boilerplate across NeuralModules with
a single unified API for batch pretraining, online fine-tuning,
LR scheduling, and checkpoint management.

Design:
  - register(module) → register any NeuralModule for training
  - pretrain(module_name, corpus) → run multi-epoch pretraining
  - online_finetune(module_name, batch) → single-step low-LR update
  - save_checkpoint(dir) / load_checkpoint(dir) → unified persistence

Constraints:
  - No modifications to Agent, FEP, body, or sleep systems
  - Does NOT replace module.pretrain() — Trainer calls module.train_step()
    internally, so individual pretrain() methods remain available
  - All modules must share the same NNConfig (device, dtype)
"""

import os
import json
import logging
from typing import Optional, Dict, List, Any, Callable
import numpy as np

from cns.nn.config import NNConfig, DEFAULT_NN_CONFIG
from cns.nn.base import NeuralModule

logger = logging.getLogger(__name__)


class Trainer:
    """Unified training orchestrator for all NeuralModules.

    Handles batch pretraining, online fine-tuning, LR scheduling,
    and unified checkpoint management across registered modules.

    Usage:
        trainer = Trainer(config=nn_config)
        trainer.register(text_encoder)
        trainer.register(generator)

        # Pre-train on corpus
        trainer.pretrain("text_encoder", corpus, epochs=5)
        trainer.pretrain("neural_generator", corpus, epochs=10)

        # Online fine-tune during dialogue
        trainer.online_finetune("neural_generator", {"input": user_tokens})

        # Save/load all modules
        trainer.save_checkpoint(".notme/checkpoints/")
        trainer.load_checkpoint(".notme/checkpoints/")
    """

    def __init__(self, config: Optional[NNConfig] = None):
        """Initialize Trainer.

        Args:
            config: NNConfig (None = use DEFAULT_NN_CONFIG)
        """
        self.config = config or DEFAULT_NN_CONFIG
        self._modules: Dict[str, NeuralModule] = {}
        self.history: Dict[str, List[Dict[str, float]]] = {}
        self._online_optimizers: Dict[str, Any] = {}
        self._online_schedulers: Dict[str, Any] = {}
        self._checkpoint_dir: Optional[str] = None

        # Internal state
        self._total_pretrain_steps: int = 0
        self._total_online_steps: int = 0
        self._version: str = "7.4"

    # ================================================================
    # Module Registration
    # ================================================================

    def register(self, module: NeuralModule):
        """Register a NeuralModule for training.

        Args:
            module: Any NeuralModule subclass instance (must be trainable)
        """
        if not isinstance(module, NeuralModule):
            raise TypeError(
                f"Expected NeuralModule, got {type(module).__name__}"
            )
        if module.name in self._modules:
            logger.warning(
                f"Module '{module.name}' already registered — overwriting"
            )
        self._modules[module.name] = module
        if module.name not in self.history:
            self.history[module.name] = []
        logger.debug(f"[Trainer] Registered module: {module.name}")

    def unregister(self, module_name: str):
        """Remove a module from the trainer.

        Args:
            module_name: Name of the module to remove
        """
        self._modules.pop(module_name, None)
        self.history.pop(module_name, None)
        self._online_optimizers.pop(module_name, None)
        self._online_schedulers.pop(module_name, None)

    @property
    def modules(self) -> Dict[str, NeuralModule]:
        """Return registered modules (read-only view)."""
        return dict(self._modules)

    @property
    def module_names(self) -> List[str]:
        """Return list of registered module names."""
        return list(self._modules.keys())

    # ================================================================
    # Pre-training
    # ================================================================

    def pretrain(
        self,
        module_name: str,
        corpus: List[str],
        epochs: Optional[int] = None,
        batch_size: int = 32,
        lr: Optional[float] = None,
        callback: Optional[Callable[[int, int, Dict[str, float]], None]] = None,
        verbose: bool = True,
    ) -> List[Dict[str, float]]:
        """Pre-train one registered module on a text corpus.

        Handles shuffling, batching, progress logging, and LR scheduling.
        Delegates to module.train_step() for actual training logic.

        Args:
            module_name: Name of registered module to train
            corpus: List of text strings (training data)
            epochs: Number of epochs (None = use config.pretrain_epochs)
            batch_size: Training batch size
            lr: Learning rate override (None = use config default)
            callback: Optional callback(epoch, batch_idx, losses) per batch
            verbose: Print progress to logger

        Returns:
            Training history: [{'epoch': N, 'loss': float, ...}, ...]
        """
        if module_name not in self._modules:
            raise KeyError(
                f"Module '{module_name}' not registered. "
                f"Available: {list(self._modules.keys())}"
            )

        module = self._modules[module_name]
        if not module.trainable:
            logger.warning(
                f"Module '{module_name}' is not trainable — skipping"
            )
            return []

        torch = self._get_torch()
        epochs = epochs if epochs is not None else self.config.pretrain_epochs
        effective_lr = lr if lr is not None else self.config.effective_lr()

        # Build vocab if module supports it and not yet built
        if hasattr(module, '_vocab_built') and not getattr(module, '_vocab_built', False):
            if hasattr(module, 'build_vocab'):
                module.build_vocab(corpus)

        # Tokenize corpus using the module's tokenizer if available
        data = self._tokenize_corpus(module, corpus)
        if len(data) == 0:
            logger.warning(f"No valid training data for module '{module_name}'")
            return []

        # Set up optimizer for pretraining
        if module._optimizer is None:
            module._optimizer = torch.optim.AdamW(
                module._net.parameters(),
                lr=effective_lr,
                weight_decay=0.01,
            )

        # LR scheduler
        batches_per_epoch = max(1, len(data) // batch_size)
        scheduler = self._create_lr_scheduler(
            module._optimizer, epochs, batches_per_epoch
        )

        n_samples = len(data)
        history = []

        module.train()
        for epoch in range(epochs):
            perm = np.random.permutation(n_samples)
            epoch_losses = []
            epoch_extras: Dict[str, List[float]] = {}

            for i in range(0, n_samples, batch_size):
                idx = perm[i : i + batch_size]
                batch_data = data[idx]
                losses = module.train_step({"input": batch_data})
                epoch_losses.append(losses.get("loss", 0.0))

                # Collect extra metrics
                for k, v in losses.items():
                    if k != "loss":
                        epoch_extras.setdefault(k, []).append(v)

                if callback:
                    callback(epoch + 1, i // batch_size, losses)

            # Step scheduler
            if scheduler is not None:
                if hasattr(scheduler, 'get_last_lr'):
                    current_lr = scheduler.get_last_lr()[0]
                scheduler.step()

            avg_loss = float(np.mean(epoch_losses))
            epoch_record = {
                "epoch": epoch + 1,
                "loss": avg_loss,
            }
            for k, values in epoch_extras.items():
                epoch_record[k] = float(np.mean(values))

            history.append(epoch_record)

            if verbose or self.config.log_verbose:
                extras_str = " ".join(
                    f"{k}={epoch_record[k]:.4f}"
                    for k in epoch_extras
                )
                logger.info(
                    f"[{module_name}] Epoch {epoch + 1}/{epochs} "
                    f"loss={avg_loss:.4f} {extras_str}"
                )

            # Checkpoint saving
            if (
                self.config.checkpoint_interval > 0
                and self._checkpoint_dir
                and (epoch + 1) % self.config.checkpoint_interval == 0
            ):
                self.save_checkpoint(self._checkpoint_dir)

        module.eval()
        self.history[module_name].extend(history)
        self._total_pretrain_steps += epochs * batches_per_epoch

        return history

    def pretrain_all(
        self,
        corpus: List[str],
        configs: Optional[Dict[str, Dict[str, Any]]] = None,
        callback: Optional[Callable] = None,
        verbose: bool = True,
    ) -> Dict[str, List[Dict[str, float]]]:
        """Pre-train all registered modules in sequence.

        Args:
            corpus: List of text strings (shared training data)
            configs: Per-module config overrides:
                {module_name: {epochs, batch_size, lr}}
            callback: Optional callback per batch
            verbose: Print progress

        Returns:
            {module_name: training_history}
        """
        configs = configs or {}
        results = {}

        for name in self._modules:
            cfg = configs.get(name, {})
            if verbose:
                logger.info(
                    f"[Trainer] Starting pretrain: {name} "
                    f"(epochs={cfg.get('epochs', self.config.pretrain_epochs)})"
                )
            history = self.pretrain(
                module_name=name,
                corpus=corpus,
                epochs=cfg.get("epochs"),
                batch_size=cfg.get("batch_size", 32),
                lr=cfg.get("lr"),
                callback=callback,
                verbose=verbose,
            )
            results[name] = history

        return results

    # ================================================================
    # Online Fine-tuning
    # ================================================================

    def online_finetune(
        self,
        module_name: str,
        batch: Dict[str, Any],
    ) -> Dict[str, float]:
        """Single-step online fine-tuning during dialogue.

        Uses a separate lower-LR optimizer to avoid catastrophic forgetting.
        Optimizer is lazily created on first call and persisted across turns.

        Args:
            module_name: Name of registered module to fine-tune
            batch: Training batch dict (same format as train_step)

        Returns:
            Loss dict {'loss': float, ...}
        """
        if module_name not in self._modules:
            raise KeyError(
                f"Module '{module_name}' not registered."
            )

        module = self._modules[module_name]
        if not module.trainable:
            return {"loss": 0.0}

        if not self.config.training_enabled:
            return {"loss": 0.0}

        torch = self._get_torch()

        # Create online optimizer (separate, lower LR)
        if module_name not in self._online_optimizers:
            online_lr = self.config.online_lr
            self._online_optimizers[module_name] = torch.optim.AdamW(
                module._net.parameters(),
                lr=online_lr,
                weight_decay=0.01,
            )

        # Save original optimizer, swap in online optimizer
        original_optimizer = module._optimizer
        module._optimizer = self._online_optimizers[module_name]

        try:
            # Ensure input is properly formatted
            formatted_batch = {}
            for k, v in batch.items():
                if isinstance(v, np.ndarray):
                    formatted_batch[k] = v
                elif isinstance(v, list) and k == "input":
                    # Tokenize list of texts using module's tokenizer
                    if hasattr(module, 'tokenize'):
                        tokens = [module.tokenize(t) for t in v if t]
                        if tokens:
                            formatted_batch[k] = np.array(tokens, dtype=np.int64)
                    else:
                        formatted_batch[k] = np.array(v)
                else:
                    formatted_batch[k] = v

            losses = module.train_step(formatted_batch)
            self._total_online_steps += 1
            return losses
        finally:
            # Restore original optimizer
            module._optimizer = original_optimizer

    def get_online_optimizer_lr(self, module_name: str) -> float:
        """Get the learning rate of the online optimizer.

        Args:
            module_name: Name of registered module

        Returns:
            Current LR of the online optimizer, or -1 if not created
        """
        if module_name not in self._online_optimizers:
            return -1.0
        opt = self._online_optimizers[module_name]
        for param_group in opt.param_groups:
            return float(param_group.get("lr", -1.0))
        return -1.0

    # ================================================================
    # Checkpoint Management
    # ================================================================

    def save_checkpoint(self, directory: str) -> str:
        """Save all registered modules + trainer state to directory.

        Creates:
            directory/
              {module_name}.pt          — per-module weights (via module.save())
              trainer_state.json        — trainer metadata

        Args:
            directory: Target directory path

        Returns:
            Path to the checkpoint directory
        """
        os.makedirs(directory, exist_ok=True)
        self._checkpoint_dir = directory

        # Save each module
        saved_paths = []
        for name, module in self._modules.items():
            path = os.path.join(directory, f"{name}.pt")
            module.save(path)
            saved_paths.append(path)

        # Save trainer state
        state = {
            "version": self._version,
            "total_pretrain_steps": self._total_pretrain_steps,
            "total_online_steps": self._total_online_steps,
            "module_names": list(self._modules.keys()),
            "history": {
                name: records
                for name, records in self.history.items()
            },
        }
        state_path = os.path.join(directory, "trainer_state.json")
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False, default=str)

        logger.info(
            f"[Trainer] Checkpoint saved to {directory} "
            f"({len(saved_paths)} modules)"
        )
        return directory

    def load_checkpoint(self, directory: str) -> bool:
        """Load all registered modules + trainer state from directory.

        Args:
            directory: Checkpoint directory path

        Returns:
            True if loaded successfully, False otherwise
        """
        if not os.path.isdir(directory):
            logger.warning(f"[Trainer] Checkpoint directory not found: {directory}")
            return False

        # Load trainer state
        state_path = os.path.join(directory, "trainer_state.json")
        if os.path.exists(state_path):
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            self._total_pretrain_steps = state.get("total_pretrain_steps", 0)
            self._total_online_steps = state.get("total_online_steps", 0)
            self.history.update({
                name: records
                for name, records in state.get("history", {}).items()
            })

        # Load each registered module
        loaded = 0
        for name, module in self._modules.items():
            path = os.path.join(directory, f"{name}.pt")
            if os.path.exists(path):
                if module.load(path):
                    loaded += 1
            else:
                logger.warning(
                    f"[Trainer] No checkpoint found for module '{name}'"
                )

        self._checkpoint_dir = directory
        logger.info(
            f"[Trainer] Checkpoint loaded from {directory} "
            f"({loaded}/{len(self._modules)} modules)"
        )
        return loaded > 0

    # ================================================================
    # Summary & Query
    # ================================================================

    def get_summary(self) -> Dict[str, Any]:
        """Return training summary across all registered modules.

        Returns:
            {
                "version": str,
                "total_pretrain_steps": int,
                "total_online_steps": int,
                "modules": {
                    module_name: {
                        "total_epochs": int,
                        "best_loss": float,
                        "last_loss": float,
                        "trainable": bool,
                        "total_trained": int,
                    }
                }
            }
        """
        module_summaries = {}
        for name, module in self._modules.items():
            hist = self.history.get(name, [])
            losses = [h["loss"] for h in hist if "loss" in h]
            module_summaries[name] = {
                "total_epochs": len(hist),
                "best_loss": float(min(losses)) if losses else float("inf"),
                "last_loss": float(losses[-1]) if losses else float("inf"),
                "trainable": module.trainable,
                "total_trained": module.total_trained,
            }

        return {
            "version": self._version,
            "total_pretrain_steps": self._total_pretrain_steps,
            "total_online_steps": self._total_online_steps,
            "modules": module_summaries,
        }

    def is_registered(self, module_name: str) -> bool:
        """Check if a module is registered."""
        return module_name in self._modules

    def get_training_history(self) -> Dict[str, Any]:
        """Return training history for Web UI (v7.5 Phase F).

        Returns per-module loss history and summary suitable for
        frontend sparkline charts and status panels.

        Returns:
            {module_name: {
                'latest_loss': float,
                'best_loss': float,
                'total_epochs': int,
                'total_steps': int,
                'loss_history': [float, ...] (last 100),
            }}
        """
        result = {}
        for name, records in self.history.items():
            if not records:
                continue
            losses = [r.get('loss', 0.0) for r in records if 'loss' in r]
            result[name] = {
                'latest_loss': losses[-1] if losses else 0.0,
                'best_loss': min(losses) if losses else 0.0,
                'total_epochs': len(records),
                'total_steps': sum(r.get('batches', 0) for r in records),
                'loss_history': losses[-100:],
            }
        return result

    # ================================================================
    # Internal Helpers
    # ================================================================

    def _get_torch(self):
        """Get torch module (lazy import)."""
        from cns.nn.bridge import _get_torch
        return _get_torch()

    def _tokenize_corpus(
        self, module: NeuralModule, corpus: List[str]
    ) -> np.ndarray:
        """Tokenize a corpus using the module's tokenizer.

        Tries module.tokenize() first, falls back to simple char-to-id.
        """
        all_ids = []

        if hasattr(module, 'tokenize'):
            for text in corpus:
                if text and isinstance(text, str) and len(text.strip()) > 0:
                    ids = module.tokenize(text.strip())
                    all_ids.append(ids)
        else:
            # Fallback: simple char-level tokenization
            # Build quick char-to-id map if module has vocab info
            for text in corpus:
                if text and isinstance(text, str) and len(text.strip()) > 0:
                    ids = [ord(c) % 10000 for c in text.strip()]
                    all_ids.append(ids)

        if len(all_ids) == 0:
            return np.array([], dtype=np.int64)

        # Pad to max length in batch
        max_len = max(len(ids) for ids in all_ids)
        padded = np.zeros((len(all_ids), max_len), dtype=np.int64)
        for i, ids in enumerate(all_ids):
            padded[i, :len(ids)] = ids

        return padded

    def _create_lr_scheduler(
        self, optimizer, epochs: int, batches_per_epoch: int
    ):
        """Create LR scheduler based on config.lr_scheduler setting.

        Args:
            optimizer: PyTorch optimizer
            epochs: Total number of epochs
            batches_per_epoch: Batches per epoch

        Returns:
            PyTorch LR scheduler, or None if 'none'
        """
        torch = self._get_torch()
        total_steps = epochs * batches_per_epoch

        scheduler_type = self.config.lr_scheduler.lower()
        if scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_steps
            )
        elif scheduler_type == "step":
            step_size = max(1, total_steps // 3)  # 3 steps over training
            return torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=0.5
            )
        elif scheduler_type == "none":
            return None
        else:
            logger.warning(
                f"Unknown LR scheduler '{scheduler_type}', "
                f"using none (valid: cosine, step, none)"
            )
            return None

    # ================================================================
    # Dunder
    # ================================================================

    def __repr__(self) -> str:
        return (
            f"Trainer(modules={list(self._modules.keys())}, "
            f"pretrain_steps={self._total_pretrain_steps}, "
            f"online_steps={self._total_online_steps})"
        )

    def __contains__(self, module_name: str) -> bool:
        return module_name in self._modules
