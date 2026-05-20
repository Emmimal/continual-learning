"""
methods/experience_replay.py
-----------------------------
Experience Replay for continual learning.

Maintains a fixed-size memory buffer of past training examples using
Vitter's reservoir sampling algorithm (1985) — guarantees a uniform
random sample of all examples seen so far, without knowing stream length.

During new task training, buffer examples are interleaved into every
mini-batch at a configurable ratio, keeping prior tasks present in the
gradient signal.

References
----------
[1] Robins, A. (1995). Catastrophic forgetting, rehearsal and pseudorehearsal.
    Connection Science 7(2), 123–146.
[2] Vitter, J. S. (1985). Random sampling with a reservoir.
    ACM TOMS 11(1), 37–57.
[3] Rolnick, D. et al. (2019). Experience Replay for Continual Learning. NeurIPS.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from methods.base_trainer import CLTrainer


# ---------------------------------------------------------------------------
# Replay Buffer — reservoir sampling
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """
    Fixed-capacity memory buffer using Vitter's reservoir sampling.

    Maintains a uniform random sample of all (x, y, task_id) triples seen
    since initialisation. The key property: each past example has an equal
    probability of being in the buffer regardless of when it was seen.

    Without reservoir sampling, naive approaches (keep first N, keep last N)
    produce biased buffers that over-represent whichever task was seen most
    recently — exactly what continual learning must avoid.

    Parameters
    ----------
    capacity : maximum number of (x, y) pairs stored at any time
    seed     : random seed for reproducible reservoir sampling
    """

    def __init__(self, capacity: int = 500, seed: int = 42):
        self.capacity = capacity
        self._rng = random.Random(seed)

        self._buffer_x: List[torch.Tensor] = []
        self._buffer_y: List[torch.Tensor] = []
        self._buffer_task_ids: List[int] = []
        self._n_seen: int = 0  # total examples ever offered (for sampling probability)

    def add_batch(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        task_id: int,
    ) -> None:
        """
        Offer a mini-batch to the reservoir.

        Each example in the batch replaces a random buffer slot with probability
        capacity / n_seen, maintaining uniform sampling guarantee.
        """
        for i in range(x.size(0)):
            self._n_seen += 1
            xi = x[i].cpu()
            yi = y[i].cpu()

            if len(self._buffer_x) < self.capacity:
                # Buffer not yet full — always add
                self._buffer_x.append(xi)
                self._buffer_y.append(yi)
                self._buffer_task_ids.append(task_id)
            else:
                # Replace slot j with probability capacity / n_seen
                j = self._rng.randint(0, self._n_seen - 1)
                if j < self.capacity:
                    self._buffer_x[j] = xi
                    self._buffer_y[j] = yi
                    self._buffer_task_ids[j] = task_id

    def sample(self, n: int) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Draw n samples uniformly from the buffer.

        Returns
        -------
        x        : (n, feature_dim) tensor
        y        : (n,) label tensor
        task_ids : list of task IDs for each sample
        """
        buf_size = len(self._buffer_x)
        if buf_size == 0:
            raise RuntimeError("ReplayBuffer is empty — cannot sample yet.")
        n = min(n, buf_size)
        indices = self._rng.sample(range(buf_size), n)
        xs = torch.stack([self._buffer_x[i] for i in indices])
        ys = torch.stack([self._buffer_y[i] for i in indices])
        task_ids = [self._buffer_task_ids[i] for i in indices]
        return xs, ys, task_ids

    def stats(self) -> Dict:
        """Return buffer occupancy and per-task distribution."""
        task_counts: Dict[int, int] = {}
        for tid in self._buffer_task_ids:
            task_counts[tid] = task_counts.get(tid, 0) + 1
        return {
            "size": len(self._buffer_x),
            "capacity": self.capacity,
            "n_seen_total": self._n_seen,
            "task_distribution": task_counts,
        }

    def __len__(self) -> int:
        return len(self._buffer_x)


# ---------------------------------------------------------------------------
# Experience Replay Trainer
# ---------------------------------------------------------------------------

class ExperienceReplay(CLTrainer):
    """
    Experience Replay continual learning trainer.

    At each gradient step during Task B training:
      1. Take a mini-batch from the current task data
      2. Sample `replay_n` examples from the buffer (prior tasks)
      3. Compute loss on the combined batch
      4. Update buffer with current task examples (reservoir sampling)

    This keeps all prior tasks present in every gradient update.

    Parameters
    ----------
    model        : nn.Module with forward(x, task_id=) interface
    buffer_size  : total replay buffer capacity across all tasks.
                   Rule of thumb: 100–200 examples per expected task.
                   With reservoir sampling, size is shared proportionally.
    replay_ratio : fraction of each mini-batch that comes from the buffer.
                   0.5 = 50% replay, 50% new task data.
                   Too low → gradient dominated by new task (forgetting).
                   Too high → convergence on new task slows unnecessarily.
    lr           : SGD learning rate
    momentum     : SGD momentum
    device       : 'cuda' or 'cpu'
    """

    def __init__(
        self,
        model: nn.Module,
        buffer_size: int = 500,
        replay_ratio: float = 0.5,
        lr: float = 0.01,
        momentum: float = 0.9,
        device: Optional[str] = None,
        seed: int = 42,
    ):
        super().__init__(model, lr=lr, momentum=momentum, device=device)
        self.buffer_size = buffer_size
        self.replay_ratio = replay_ratio
        self.buffer = ReplayBuffer(capacity=buffer_size, seed=seed)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        epochs: int = 5,
    ) -> None:
        """
        Train on task_id, interleaving replay examples from the buffer.

        The buffer is populated DURING training, not before, so the first
        task trains with an empty buffer (equivalent to naive training).
        From Task 1 onward, every batch contains a mix of new and old examples.
        """
        if hasattr(self.model, 'heads') and task_id >= len(self.model.heads):
            self.model.add_task_head()
            self.model.heads[task_id].to(self.device)

        self._optimiser = self._build_optimiser()

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            n_batches = 0

            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)

                # --- Replay mix ---
                if len(self.buffer) > 0:
                    n_replay = max(1, int(len(y) * self.replay_ratio
                                         / (1.0 - self.replay_ratio)))
                    rx, ry, rtask_ids = self.buffer.sample(n_replay)
                    rx, ry = rx.to(self.device), ry.to(self.device)

                    # Combine new and replay examples
                    # We compute separate losses per task to route through correct heads
                    new_loss = self.criterion(self.model(x, task_id=task_id), y)
                    replay_loss = self._compute_replay_loss(rx, ry, rtask_ids)
                    loss = new_loss + replay_loss
                else:
                    loss = self.criterion(self.model(x, task_id=task_id), y)

                self._optimiser.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self._optimiser.step()

                # --- Add current batch to replay buffer ---
                self.buffer.add_batch(x.cpu(), y.cpu(), task_id)

                total_loss += loss.item()
                n_batches += 1

    def _compute_replay_loss(
        self,
        rx: torch.Tensor,
        ry: torch.Tensor,
        rtask_ids: List[int],
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss on replayed examples, routing each
        sample through its correct task head.

        Groups samples by task ID for efficiency (one forward pass per task).
        """
        total_replay_loss = torch.tensor(0.0, device=self.device)
        unique_tasks = set(rtask_ids)

        for tid in unique_tasks:
            mask = torch.tensor(
                [t == tid for t in rtask_ids], dtype=torch.bool
            )
            rx_t = rx[mask]
            ry_t = ry[mask]
            out_t = self.model(rx_t, task_id=tid)
            total_replay_loss = total_replay_loss + self.criterion(out_t, ry_t)

        return total_replay_loss / max(len(unique_tasks), 1)

    def buffer_stats(self) -> Dict:
        """Return current replay buffer statistics."""
        return self.buffer.stats()
