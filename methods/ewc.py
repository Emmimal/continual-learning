"""
methods/ewc.py
--------------
Elastic Weight Consolidation (EWC) — Kirkpatrick et al. (2017).

Implements both the original EWC (separate Fisher per task) and the
Online EWC variant (accumulated Fisher across tasks) — Schwarz et al. (2018).

Key mechanism
-------------
After training Task A, the Fisher Information diagonal F_i is estimated for
each parameter θ_i. When training Task B, a quadratic penalty is added:

    L_total = L_task_B + (λ/2) * Σ_i F_i * (θ_i - θ*_i)²

where θ*_i is the parameter value after training Task A (the anchor).

High F_i → parameter is important for Task A → expensive to change.
Low F_i  → parameter is unimportant for Task A → free to adapt.

References
----------
[1] Kirkpatrick, J. et al. (2017). Overcoming catastrophic forgetting in
    neural networks. PNAS 114(13), 3521–3526.
[2] Schwarz, J. et al. (2018). Progress & Compress. ICML.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from methods.base_trainer import CLTrainer


class EWC(CLTrainer):
    """
    EWC continual learning trainer.

    Parameters
    ----------
    model            : MultiHeadMLP or any model with forward(x, task_id=) signature
    lambda_ewc       : regularisation strength.  Controls the forgetting/plasticity trade-off:
                         - Too high  → resists learning new tasks (plasticity collapse)
                         - Too low   → insufficient forgetting protection
                       Tune on a validation set containing BOTH old and new task examples.
                       Typical range: 0.1 – 10.0.  Start at 0.4.
    n_fisher_samples : number of training samples used to estimate the Fisher diagonal.
                       More samples → more accurate Fisher but slower consolidation.
                       200–500 is sufficient for Split-MNIST; scale up for complex tasks.
    online           : if True, accumulates Fisher across tasks (Online EWC).
                       More memory-efficient for many tasks; equivalent to original EWC
                       for moderate task counts (≤ 10).
    lr               : SGD learning rate
    momentum         : SGD momentum
    device           : 'cuda' or 'cpu'
    """

    def __init__(
        self,
        model: nn.Module,
        lambda_ewc: float = 0.4,
        n_fisher_samples: int = 200,
        online: bool = True,
        lr: float = 0.01,
        momentum: float = 0.9,
        device: Optional[str] = None,
    ):
        super().__init__(model, lr=lr, momentum=momentum, device=device)
        self.lambda_ewc = lambda_ewc
        self.n_fisher_samples = n_fisher_samples
        self.online = online

        # Accumulated Fisher diagonal (Online EWC) — grows additively
        self._fisher_accum: Dict[str, torch.Tensor] = {}
        # Anchor: parameter values after each consolidation
        self._anchor: Dict[str, torch.Tensor] = {}

        # For non-online EWC: list of (fisher_diag, anchor) per task
        self._task_fishers: list = []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_task(self, task_id: int, train_loader: DataLoader,
                   epochs: int = 5) -> None:
        """Train on task_id with EWC penalty protecting all prior tasks."""

        if hasattr(self.model, 'heads') and task_id >= len(self.model.heads):
            self.model.add_task_head()
            self.model.heads[task_id].to(self.device)

        self._optimiser = self._build_optimiser()

        for epoch in range(epochs):
            self._train_one_epoch(
                task_id=task_id,
                train_loader=train_loader,
                extra_loss_fn=self._ewc_penalty if self._anchor else None,
            )

    def _ewc_penalty(self, outputs, labels) -> torch.Tensor:
        """
        Compute the EWC regularisation term.

        For Online EWC: uses the single accumulated Fisher and anchor.
        For standard EWC: sums penalties across all stored (Fisher, anchor) pairs.
        """
        if self.online:
            return self._penalty_from(self._fisher_accum, self._anchor)
        else:
            total = torch.tensor(0.0, device=self.device)
            for fisher, anchor in self._task_fishers:
                total = total + self._penalty_from(fisher, anchor)
            return total

    def _penalty_from(
        self,
        fisher: Dict[str, torch.Tensor],
        anchor: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        penalty = torch.tensor(0.0, device=self.device)
        for name, param in self.model.named_parameters():
            if name in fisher and name in anchor:
                # Skip parameters whose shape changed — e.g. when a
                # SingleHeadMLP's output head expands in class-incremental
                # learning. New rows have no Fisher estimate yet.
                if param.shape != anchor[name].shape:
                    continue
                diff = param - anchor[name]
                penalty = penalty + (fisher[name] * diff.pow(2)).sum()
        return (self.lambda_ewc / 2.0) * penalty

    # ------------------------------------------------------------------
    # Consolidation  — called after each task completes
    # ------------------------------------------------------------------

    def consolidate(self, task_id: int, train_loader: DataLoader) -> None:
        """
        Estimate Fisher diagonal and snapshot anchor weights.

        Must be called after train_task() and before train_task() for
        the next task. This is what makes EWC work — without consolidation
        there is no penalty.
        """
        print(f"  [EWC] Estimating Fisher diagonal (task {task_id}, "
              f"n_samples={self.n_fisher_samples})...")

        new_fisher = self._estimate_fisher(task_id, train_loader)
        new_anchor = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        if self.online:
            # Accumulate Fisher across tasks — handle head expansion gracefully
            for name in new_fisher:
                if name in self._fisher_accum:
                    old_f = self._fisher_accum[name]
                    new_f = new_fisher[name]
                    if old_f.shape == new_f.shape:
                        self._fisher_accum[name] = old_f + new_f
                    else:
                        # Head was expanded — replace with new (larger) Fisher
                        # The old Fisher values for existing rows are preserved
                        # by padding; simpler to just replace for the head layer
                        self._fisher_accum[name] = new_f.clone()
                else:
                    self._fisher_accum[name] = new_fisher[name].clone()
            # Anchor always points to the most recent consolidation
            self._anchor = new_anchor
        else:
            # Store separate (Fisher, anchor) for each task
            self._task_fishers.append((new_fisher, new_anchor))

    def _estimate_fisher(
        self, task_id: int, train_loader: DataLoader
    ) -> Dict[str, torch.Tensor]:
        """
        Estimate the empirical Fisher Information diagonal.

        For each sample: compute the gradient of log P(ŷ | x) with respect
        to each parameter, then square and average over n_fisher_samples.

        F_i ≈ (1/N) Σ_n [ ∂/∂θ_i log P(ŷ_n | x_n) ]²

        High F_i means θ_i strongly affects the model's output distribution
        → changing it will hurt Task A performance → penalise heavily.
        """
        fisher: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        self.model.eval()
        n_samples = 0

        for x, y in train_loader:
            if n_samples >= self.n_fisher_samples:
                break
            x = x.to(self.device)

            for i in range(x.size(0)):
                if n_samples >= self.n_fisher_samples:
                    break

                xi = x[i: i + 1]
                self.model.zero_grad()

                logits = self.model(xi, task_id=task_id)
                log_probs = torch.log_softmax(logits, dim=1)
                predicted = logits.argmax(dim=1)

                # Gradient of log P(ŷ | x) — empirical Fisher
                loss = -log_probs[0, predicted[0]]
                loss.backward()

                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        fisher[name] += param.grad.data.pow(2)

                n_samples += 1

        # Normalise by sample count
        for name in fisher:
            fisher[name] /= max(n_samples, 1)

        return fisher

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def fisher_stats(self) -> dict:
        """Return summary statistics of the accumulated Fisher diagonal."""
        if not self._fisher_accum:
            return {"status": "No Fisher computed yet"}
        all_vals = torch.cat([f.flatten() for f in self._fisher_accum.values()])
        return {
            "mean": float(all_vals.mean()),
            "max": float(all_vals.max()),
            "min": float(all_vals.min()),
            "nonzero_pct": float((all_vals > 1e-8).float().mean()),
            "n_params": int(all_vals.numel()),
        }
