"""
methods/base_trainer.py
-----------------------
Abstract base class for all continual learning trainers.

Every trainer in methods/ inherits from CLTrainer and must implement:
  train_task(task_id, train_loader, epochs)  — train on one task
  evaluate(task_id, test_loader)             — evaluate on one task
  consolidate(task_id, train_loader)         — post-task consolidation hook
                                               (no-op for methods that don't need it)

The base class provides:
  - A shared SGD training loop (train_one_epoch)
  - Per-task accuracy evaluation
  - Device management
  - Gradient clipping
"""

from __future__ import annotations

import abc
import time
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class CLTrainer(abc.ABC):
    """
    Abstract base trainer for continual learning methods.

    Parameters
    ----------
    model     : any nn.Module with forward(x, task_id=...) signature
    lr        : learning rate for SGD with momentum
    momentum  : SGD momentum coefficient
    device    : 'cuda' or 'cpu'; auto-detected if None
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 0.01,
        momentum: float = 0.9,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.lr = lr
        self.momentum = momentum
        self.criterion = nn.CrossEntropyLoss()

        # Subclasses build their own optimiser (may differ per method)
        self._optimiser: Optional[torch.optim.Optimizer] = None

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def train_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        epochs: int = 5,
    ) -> None:
        """Train the model on a single task for `epochs` epochs."""

    def consolidate(
        self,
        task_id: int,
        train_loader: DataLoader,
    ) -> None:
        """
        Post-task consolidation hook (optional for most methods).

        Called after train_task() completes for `task_id`. Override in methods
        that need to compute Fisher diagonals, prune weights, store anchors, etc.
        Default: no-op.
        """

    # ------------------------------------------------------------------
    # Shared training utilities
    # ------------------------------------------------------------------

    def _build_optimiser(self) -> torch.optim.SGD:
        """Build a fresh SGD optimiser over all trainable parameters."""
        return torch.optim.SGD(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr,
            momentum=self.momentum,
        )

    def _train_one_epoch(
        self,
        task_id: int,
        train_loader: DataLoader,
        extra_loss_fn=None,
        grad_clip: float = 1.0,
    ) -> float:
        """
        Run one full epoch over train_loader.

        Parameters
        ----------
        task_id       : passed to model.forward()
        train_loader  : training DataLoader
        extra_loss_fn : callable(model, outputs, labels) -> additional_loss_tensor
                        used for EWC penalty, replay loss, etc.
        grad_clip     : max gradient norm (0 = disabled)

        Returns
        -------
        mean_loss : average total loss over the epoch
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for x, y in train_loader:
            x, y = x.to(self.device), y.to(self.device)
            self._optimiser.zero_grad()

            outputs = self.model(x, task_id=task_id)
            loss = self.criterion(outputs, y)

            if extra_loss_fn is not None:
                loss = loss + extra_loss_fn(outputs, y)

            loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=grad_clip
                )

            self._optimiser.step()
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, task_id: int, test_loader: DataLoader) -> float:
        """
        Compute accuracy on test_loader for the given task.

        Returns
        -------
        accuracy : float in [0, 1]
        """
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x, task_id=task_id)
                preds = outputs.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += len(y)

        return correct / total if total > 0 else 0.0

    def evaluate_all(
        self,
        test_loaders: list,
        up_to_task: Optional[int] = None,
    ) -> list:
        """
        Evaluate on all tasks from 0 to up_to_task (inclusive).

        Parameters
        ----------
        test_loaders : list of DataLoaders (one per task)
        up_to_task   : evaluate tasks [0 .. up_to_task]; defaults to all

        Returns
        -------
        accuracies : list[float]
        """
        n = (up_to_task + 1) if up_to_task is not None else len(test_loaders)
        return [self.evaluate(t, test_loaders[t]) for t in range(n)]

    # ------------------------------------------------------------------
    # Timed training wrapper
    # ------------------------------------------------------------------

    def run_sequence(
        self,
        train_loaders: list,
        test_loaders: list,
        epochs: int = 5,
        verbose: bool = True,
    ) -> dict:
        """
        Train and evaluate across the full task sequence.

        Returns a dict with:
          'per_task_acc'   : accuracy on each task right after training it
          'final_accs'     : accuracy on every task after the final task
          'acc_matrix'     : list of lists  acc_matrix[i][j] = acc(task i, after task j)
          'runtime_s'      : total wall-clock seconds
        """
        n_tasks = len(train_loaders)
        acc_matrix = [[None] * n_tasks for _ in range(n_tasks)]
        per_task_acc = []
        t0 = time.time()

        for task_id in range(n_tasks):
            self.train_task(task_id, train_loaders[task_id], epochs=epochs)
            self.consolidate(task_id, train_loaders[task_id])

            # Evaluate all tasks seen so far
            for prev_id in range(task_id + 1):
                acc = self.evaluate(prev_id, test_loaders[prev_id])
                acc_matrix[prev_id][task_id] = acc

            per_task_acc.append(acc_matrix[task_id][task_id])

            if verbose:
                accs = [acc_matrix[t][task_id] for t in range(task_id + 1)]
                accs_str = "  ".join(f"T{t}:{a:.3f}" for t, a in enumerate(accs))
                print(f"  After Task {task_id}: {accs_str}")

        final_accs = [acc_matrix[t][n_tasks - 1] for t in range(n_tasks)]
        runtime = time.time() - t0

        return {
            "per_task_acc": per_task_acc,
            "final_accs": final_accs,
            "acc_matrix": acc_matrix,
            "runtime_s": runtime,
        }
