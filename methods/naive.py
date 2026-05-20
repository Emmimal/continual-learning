"""
methods/naive.py
----------------
Naive (Baseline) Continual Learner.

No forgetting protection whatsoever. Fine-tunes on each new task without
any constraint on weight movement. Used as the lower bound in benchmarks —
everything should outperform this on BWT.

This is what your production model does today if it is retrained on new
task data without any continual learning strategy.
"""

from torch.utils.data import DataLoader
from methods.base_trainer import CLTrainer
import torch.nn as nn
from typing import Optional


class NaiveTrainer(CLTrainer):
    """
    Baseline trainer: unconstrained sequential fine-tuning.

    Expected benchmark behaviour
    ----------------------------
    - High accuracy on the most-recently-trained task (high plasticity)
    - Severe BWT (high forgetting) — prior task accuracy collapses
    - FM (max forgetting) is typically 0.3–0.7 on Split-MNIST after Task 4
    """

    def __init__(self, model: nn.Module, lr: float = 0.01, momentum: float = 0.9,
                 device: Optional[str] = None):
        super().__init__(model, lr=lr, momentum=momentum, device=device)

    def train_task(self, task_id: int, train_loader: DataLoader,
                   epochs: int = 5) -> None:
        """
        Fine-tune unconstrainedly on task `task_id`.

        A fresh optimiser is built for each task so momentum does not carry
        stale gradient information across tasks.
        """
        # Add head if this is a new task (MultiHeadMLP)
        if hasattr(self.model, 'heads') and task_id >= len(self.model.heads):
            self.model.add_task_head()
            self.model.heads[task_id].to(self.device)

        self._optimiser = self._build_optimiser()

        for epoch in range(epochs):
            loss = self._train_one_epoch(task_id, train_loader)
