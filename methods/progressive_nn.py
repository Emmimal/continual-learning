"""
methods/progressive_nn.py
--------------------------
Progressive Neural Network (PNN) trainer — Rusu et al. (2016).

PNN eliminates catastrophic forgetting by structural design:
  - Each new task gets its own neural network column
  - Prior columns are FROZEN immediately after their task trains
  - Lateral connections allow the new column to reuse prior knowledge
    without modifying prior weights

Zero forgetting is GUARANTEED for prior tasks because their column weights
are physically frozen — no gradient update can change them.

The trade-off: model size grows linearly with task count.
For T tasks with H hidden units per layer, total parameters ≈ O(T² × H²)
due to the quadratic growth of lateral connections.

When to use PNN
---------------
  ✓ Small number of tasks (2–10)
  ✓ Tasks where knowledge transfer (FWT) matters
  ✓ Task identity is KNOWN at inference
  ✗ Many tasks (model becomes prohibitively large)
  ✗ Class-incremental or domain-incremental settings

References
----------
[1] Rusu, A. A., Rabinowitz, N. C., Desjardins, G., Soyer, H., Kirkpatrick, J.,
    Kavukcuoglu, K., Pascanu, R., & Hadsell, R. (2016).
    Progressive Neural Networks. arXiv:1606.04671
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from methods.base_trainer import CLTrainer
from models.architectures import ProgressiveNeuralNet


class PNNTrainer(CLTrainer):
    """
    Progressive Neural Network continual learning trainer.

    The model MUST be a ProgressiveNeuralNet instance — the column-based
    architecture is fundamental to the method, not a configuration choice.

    Parameters
    ----------
    model    : ProgressiveNeuralNet instance (from models/architectures.py)
    lr       : SGD learning rate
    momentum : SGD momentum
    device   : 'cuda' or 'cpu'
    """

    def __init__(
        self,
        model: ProgressiveNeuralNet,
        lr: float = 0.01,
        momentum: float = 0.9,
        device: Optional[str] = None,
    ):
        if not isinstance(model, ProgressiveNeuralNet):
            raise TypeError(
                "PNNTrainer requires a ProgressiveNeuralNet model. "
                "Use NaiveTrainer, EWC, or ExperienceReplay for other architectures."
            )
        super().__init__(model, lr=lr, momentum=momentum, device=device)

    def train_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        epochs: int = 5,
    ) -> None:
        """
        Add a new column for task_id and train it with lateral connections
        from all prior frozen columns.

        The new column is the only one with requires_grad=True.
        Prior columns are already frozen from their own consolidation step.
        """
        # Add column for this task only if not already added
        if task_id < len(self.model.columns):
            col_id = task_id  # already added by FWT benchmark
        else:
            col_id = self.model.add_column()
            self.model.columns[col_id].to(self.device)

        if col_id != task_id:
            raise RuntimeError(
                f"Column index mismatch: expected {task_id}, got {col_id}. "
                "Train tasks in sequential order 0, 1, 2, ..."
            )

        # Only train the NEW column's parameters
        self._optimiser = torch.optim.SGD(
            self.model.columns[task_id].parameters(),
            lr=self.lr,
            momentum=self.momentum,
        )

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            n_batches = 0

            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self._optimiser.zero_grad()

                logits = self.model(x, task_id=task_id)
                loss = self.criterion(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.columns[task_id].parameters(), 1.0
                )
                self._optimiser.step()
                total_loss += loss.item()
                n_batches += 1

    def consolidate(self, task_id: int, train_loader: DataLoader) -> None:
        """
        Freeze the just-trained column.

        After freezing, prior task accuracy is structurally guaranteed —
        the frozen weights cannot change during any future training.
        """
        self.model.freeze_column(task_id)
        frozen_count = sum(
            p.numel()
            for p in self.model.columns[task_id].parameters()
        )
        print(f"  [PNN] Column {task_id} frozen. "
              f"Frozen params: {frozen_count:,}")

    def evaluate(self, task_id: int, test_loader: DataLoader) -> float:
        """Evaluate using the column corresponding to task_id."""
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x, task_id=task_id)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += len(y)
        return correct / total if total > 0 else 0.0

    def capacity_report(self) -> dict:
        """Return model capacity breakdown across columns."""
        total = sum(p.numel() for p in self.model.parameters())
        frozen = sum(p.numel() for p in self.model.parameters()
                     if not p.requires_grad)
        per_col = {}
        for i, col in enumerate(self.model.columns):
            per_col[i] = {
                "params": sum(p.numel() for p in col.parameters()),
                "frozen": not any(p.requires_grad for p in col.parameters()),
            }
        return {
            "n_columns": len(self.model.columns),
            "total_params": total,
            "frozen_params": frozen,
            "trainable_params": total - frozen,
            "per_column": per_col,
        }
