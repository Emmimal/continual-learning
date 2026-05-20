"""
methods/gem.py
--------------
Gradient Episodic Memory (GEM) — Lopez-Paz & Ranzato (2017).

GEM stores a small episodic memory of past task examples and uses them
to CONSTRAIN gradient updates: the loss on stored examples must not
increase as a result of any gradient step.

Formally, after computing the gradient g for the new task, GEM solves
a constrained optimisation to find the closest gradient ĝ to g such
that ĝ · g_k ≥ 0 for all prior task gradients g_k stored in memory.

This is implemented as a quadratic programme (QP). To avoid the full QP
at every step, this implementation uses the efficient dual-variable approach
from the original paper.

How GEM differs from EWC and Replay
-------------------------------------
  EWC      : Soft constraint via regularisation penalty — prior loss CAN rise,
             just at a cost.
  Replay   : Keeps prior tasks in the gradient signal directly — effective
             but requires raw data storage.
  GEM      : Hard constraint — prior loss CANNOT rise on episodic examples.
             More principled than EWC; avoids raw data storage requirements
             of standard replay (stores a small fixed set, not a live stream).

Limitations
-----------
  - QP projection is O(T²) per step — slow for many tasks.
  - Episodic memory per task is small (200–500 examples).
  - Does not work in data-retention-prohibited settings (stores raw examples).

References
----------
[1] Lopez-Paz, D., & Ranzato, M. A. (2017). Gradient Episodic Memory for
    Continual Learning. NeurIPS 30.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from methods.base_trainer import CLTrainer


def _project_gradients(
    current_grad: torch.Tensor,
    memory_grads: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    """
    Project current_grad onto the GEM feasible region via dual QP.

    The feasible region is: { g : g · memory_grads[k] ≥ margin  ∀k }

    If current_grad already satisfies all constraints, return it unchanged.
    Otherwise, find the closest feasible gradient using the dual solution.

    Parameters
    ----------
    current_grad  : (D,) gradient vector for the new task
    memory_grads  : (n_tasks, D) gradient vectors for prior tasks
    margin        : violation tolerance (0 = strict non-increase in prior losses)

    Returns
    -------
    projected : (D,) projected gradient
    """
    dots = torch.mv(memory_grads, current_grad)

    if (dots >= margin - 1e-7).all():
        # No violation — return unchanged
        return current_grad

    # Dual QP: minimise ||g - g_new||² subject to G @ g ≥ margin
    # Solution via projected gradient on the dual (Algorithm 1 in GEM paper)
    G = memory_grads.cpu().double().numpy()
    g = current_grad.cpu().double().numpy()

    t = memory_grads.size(0)
    # Gram matrix
    GGt = G @ G.T
    # Dual variable initialisation — closed-form warm start via pseudo-inverse
    # for the unconstrained least-squares solution, then project to non-negative
    rhs = margin - G @ g          # violation vector; positive = violated
    eigmax = float(np.linalg.eigvalsh(GGt).max()) if t > 1 else float(GGt[0, 0])
    step = 1.0 / (eigmax + 1e-8)
    # Dual variables via projected gradient ASCENT on dual objective
    v = np.maximum(0.0, np.linalg.lstsq(GGt, rhs, rcond=None)[0])
    for _ in range(1000):                          # refine with PG steps
        grad_v = -(rhs - GGt @ v)                 # gradient of -dual
        v = np.maximum(0.0, v - step * grad_v)

    # Recover primal solution
    g_proj = g + G.T @ v
    return torch.tensor(g_proj, dtype=current_grad.dtype, device=current_grad.device)


class GEM(CLTrainer):
    """
    GEM continual learning trainer.

    Parameters
    ----------
    model          : nn.Module with forward(x, task_id=) interface
    memory_size    : number of examples stored per task in episodic memory.
                     These are selected as the FIRST memory_size examples
                     of each task's training data.
    margin         : GEM constraint margin (0 = strict; small positive = tolerance)
    lr             : SGD learning rate
    momentum       : SGD momentum
    device         : 'cuda' or 'cpu'
    """

    def __init__(
        self,
        model: nn.Module,
        memory_size: int = 200,
        margin: float = 0.0,
        lr: float = 0.01,
        momentum: float = 0.9,
        device: Optional[str] = None,
    ):
        super().__init__(model, lr=lr, momentum=momentum, device=device)
        self.memory_size = memory_size
        self.margin = margin

        # Episodic memory: task_id → (X, Y)
        self._memory: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        epochs: int = 5,
    ) -> None:
        """Train on task_id with GEM gradient projection."""

        if hasattr(self.model, 'heads') and task_id >= len(self.model.heads):
            self.model.add_task_head()
            self.model.heads[task_id].to(self.device)

        self._optimiser = self._build_optimiser()

        # Fill episodic memory for this task BEFORE training starts
        self._fill_memory(task_id, train_loader)

        for epoch in range(epochs):
            self.model.train()

            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)

                # --- Compute gradient for current task ---
                self._optimiser.zero_grad()
                loss = self.criterion(self.model(x, task_id=task_id), y)
                loss.backward()

                current_grad = self._get_flat_grad()

                # --- Project if there are prior tasks in memory ---
                if task_id > 0:
                    memory_grads = self._compute_memory_grads(task_id)
                    projected_grad = _project_gradients(
                        current_grad, memory_grads, self.margin
                    )
                    self._set_flat_grad(projected_grad)

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self._optimiser.step()

    def consolidate(self, task_id: int, train_loader: DataLoader) -> None:
        """No additional consolidation needed — memory is filled before training."""
        pass

    # ------------------------------------------------------------------
    # GEM helpers
    # ------------------------------------------------------------------

    def _fill_memory(self, task_id: int, train_loader: DataLoader) -> None:
        """Store the first memory_size examples of task_id in episodic memory."""
        xs, ys = [], []
        n_collected = 0

        for x, y in train_loader:
            remaining = self.memory_size - n_collected
            xs.append(x[:remaining])
            ys.append(y[:remaining])
            n_collected += x[:remaining].size(0)
            if n_collected >= self.memory_size:
                break

        self._memory[task_id] = (
            torch.cat(xs, dim=0).to(self.device),
            torch.cat(ys, dim=0).to(self.device),
        )

    def _compute_memory_grads(self, current_task: int) -> torch.Tensor:
        """
        Compute gradients of prior-task losses on episodic memory.

        Returns (n_prior_tasks, n_params) tensor.
        """
        grads = []
        self.model.eval()

        for tid in range(current_task):
            if tid not in self._memory:
                continue
            mx, my = self._memory[tid]
            self._optimiser.zero_grad()
            loss = self.criterion(self.model(mx, task_id=tid), my)
            loss.backward()
            grads.append(self._get_flat_grad().unsqueeze(0))

        self.model.train()
        return torch.cat(grads, dim=0) if grads else torch.zeros(1, self._n_params())

    def _get_flat_grad(self) -> torch.Tensor:
        """Flatten all parameter gradients into a single vector."""
        grads = []
        for p in self.model.parameters():
            if p.requires_grad:
                g = p.grad if p.grad is not None else torch.zeros_like(p)
                grads.append(g.view(-1))
        return torch.cat(grads)

    def _set_flat_grad(self, flat_grad: torch.Tensor) -> None:
        """Write a flat gradient vector back into model parameter .grad fields."""
        offset = 0
        for p in self.model.parameters():
            if p.requires_grad:
                n = p.numel()
                p.grad = flat_grad[offset: offset + n].view(p.shape).clone()
                offset += n

    def _n_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
