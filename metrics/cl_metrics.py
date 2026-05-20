"""
metrics/cl_metrics.py
---------------------
Evaluation metrics for continual learning systems.

The four canonical metrics (Lopez-Paz & Ranzato, 2017; Diaz-Rodriguez et al., 2018):

  ACC  — Average accuracy across all tasks after the final task
  BWT  — Backward Transfer: average change in prior-task accuracy after training
  FWT  — Forward Transfer: average zero-shot accuracy boost on future tasks
  FM   — Forgetting Measure: maximum accuracy drop across any prior task

Additional production metrics:
  Stability   : how well the model preserves prior task accuracy
  Plasticity  : how well the model learns new tasks

Usage
-----
    tracker = CLMetricsTracker(n_tasks=5)
    tracker.record(task_id=0, after_task=0, accuracy=0.97)
    tracker.record(task_id=0, after_task=1, accuracy=0.94)
    tracker.record(task_id=0, after_task=2, accuracy=0.91)
    ...
    metrics = tracker.compute()
    print(metrics.summary())

References
----------
[1] Lopez-Paz, D., & Ranzato, M. A. (2017). Gradient Episodic Memory.
    NeurIPS 30.
[2] Diaz-Rodriguez, N., Lomonaco, V., Filliat, D., & Maltoni, D. (2018).
    Don't forget, there are many tasks! Towards Next-Gen NLP Systems.
    arXiv:1806.08568
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Accuracy matrix tracker
# ---------------------------------------------------------------------------

class CLMetricsTracker:
    """
    Tracks the accuracy matrix R[i, j] — accuracy on task i evaluated
    immediately after training on task j.

    Only entries where j >= i are meaningful:
      - R[i, i]  : accuracy on task i right after training it
      - R[i, j>i]: accuracy on task i after subsequent tasks were trained
                   (measures forgetting)

    Parameters
    ----------
    n_tasks : total number of tasks in the sequence
    """

    def __init__(self, n_tasks: int):
        self.n_tasks = n_tasks
        # R[i, j] = accuracy on task i right after task j was trained
        self._R: np.ndarray = np.full((n_tasks, n_tasks), np.nan)

    def record(self, task_id: int, after_task: int, accuracy: float) -> None:
        """
        Record accuracy on `task_id` evaluated right after `after_task` trained.

        Parameters
        ----------
        task_id    : which task was EVALUATED (0-indexed)
        after_task : which task was JUST TRAINED (0-indexed)
        accuracy   : accuracy in [0, 1]
        """
        if not (0 <= task_id < self.n_tasks):
            raise ValueError(f"task_id {task_id} out of range [0, {self.n_tasks})")
        if not (0 <= after_task < self.n_tasks):
            raise ValueError(f"after_task {after_task} out of range [0, {self.n_tasks})")
        self._R[task_id, after_task] = accuracy

    def compute(self) -> CLMetrics:
        """Compute all metrics from the recorded accuracy matrix."""
        R = self._R
        T = self.n_tasks

        # --- ACC: average accuracy on all tasks after the final task ---
        acc = float(np.nanmean(R[:, T - 1]))

        # --- BWT: average change in prior-task accuracy after training ---
        # BWT = (1 / (T-1)) * sum_{i=1}^{T-1} [R[i, T-1] - R[i, i]]
        # Negative BWT = forgetting; positive BWT = positive transfer (rare)
        if T > 1:
            bwt_terms = [R[i, T - 1] - R[i, i] for i in range(T - 1)
                         if not np.isnan(R[i, T - 1]) and not np.isnan(R[i, i])]
            bwt = float(np.mean(bwt_terms)) if bwt_terms else float("nan")
        else:
            bwt = 0.0

        # --- FWT: average zero-shot accuracy on future tasks ---
        # FWT = (1 / (T-1)) * sum_{i=2}^{T} [R[i, i-1] - b_i]
        # b_i = random baseline accuracy (0.5 for binary, 0.1 for 10-class)
        # Here we approximate FWT as the pre-training accuracy before task i
        # using R[i, i-1] as a proxy (model state just before task i trains)
        if T > 1:
            fwt_terms = [R[i, i - 1] for i in range(1, T)
                         if not np.isnan(R[i, i - 1])]
            fwt = float(np.mean(fwt_terms)) if fwt_terms else float("nan")
        else:
            fwt = float("nan")

        # --- FM: maximum forgetting across any single prior task ---
        # FM = max over i of [R[i, i] - R[i, T-1]]
        if T > 1:
            fm_terms = [R[i, i] - R[i, T - 1] for i in range(T - 1)
                        if not np.isnan(R[i, i]) and not np.isnan(R[i, T - 1])]
            fm = float(np.max(fm_terms)) if fm_terms else float("nan")
        else:
            fm = 0.0

        # --- Per-task forgetting: R[i,i] - R[i,T-1] ---
        per_task_forgetting = {}
        for i in range(T - 1):
            if not np.isnan(R[i, i]) and not np.isnan(R[i, T - 1]):
                per_task_forgetting[i] = float(R[i, i] - R[i, T - 1])

        # --- Intransigence: how much the model fails to learn new tasks ---
        # Measured as the gap between the best possible accuracy (R[T-1, T-1])
        # and a joint-training oracle. We approximate as 1 - R[T-1, T-1].
        intransigence = 1.0 - float(R[T - 1, T - 1]) if not np.isnan(R[T - 1, T - 1]) else float("nan")

        return CLMetrics(
            acc=acc,
            bwt=bwt,
            fwt=fwt,
            fm=fm,
            intransigence=intransigence,
            per_task_forgetting=per_task_forgetting,
            R=R.copy(),
            n_tasks=T,
        )

    @property
    def matrix(self) -> np.ndarray:
        return self._R.copy()


# ---------------------------------------------------------------------------
# Metrics dataclass
# ---------------------------------------------------------------------------

@dataclass
class CLMetrics:
    """All computed continual learning metrics for one method run."""

    acc: float              # Average accuracy across all tasks (↑)
    bwt: float              # Backward transfer — 0 = no forgetting (↑)
    fwt: float              # Forward transfer — proxy for knowledge reuse (↑)
    fm: float               # Forgetting measure — max drop on any task (↓)
    intransigence: float    # Failure to learn new tasks (↓)
    per_task_forgetting: Dict[int, float] = field(default_factory=dict)
    R: Optional[np.ndarray] = field(default=None, repr=False)
    n_tasks: int = 0

    def summary(self, method_name: str = "") -> str:
        sep = "=" * 62
        header = f"  CL METRICS SUMMARY{(' — ' + method_name) if method_name else ''}"
        lines = [
            sep,
            header,
            sep,
            f"  ACC (avg accuracy, final)   : {self.acc:.4f}",
            f"  BWT (backward transfer)     : {self.bwt:+.4f}  "
            f"({'no forgetting' if abs(self.bwt) < 0.005 else 'forgetting' if self.bwt < 0 else 'positive transfer'})",
            f"  FWT (forward transfer)      : {self.fwt:.4f}",
            f"  FM  (max forgetting)        : {self.fm:.4f}",
            f"  Intransigence               : {self.intransigence:.4f}",
            sep,
        ]
        if self.per_task_forgetting:
            lines.append("  Per-task forgetting:")
            for task_id, forget in self.per_task_forgetting.items():
                lines.append(f"    Task {task_id}: {forget:+.4f}")
            lines.append(sep)
        return "\n".join(lines)

    def accuracy_matrix_str(self) -> str:
        """Pretty-print the accuracy matrix R[i, j]."""
        if self.R is None:
            return "No accuracy matrix recorded."
        T = self.n_tasks
        col_header = "         " + "".join(f"  T{j:<4}" for j in range(T))
        rows = [col_header, "-" * (9 + 7 * T)]
        for i in range(T):
            row = f"  Task {i} |"
            for j in range(T):
                val = self.R[i, j]
                if np.isnan(val):
                    row += "   —  "
                else:
                    row += f" {val:.3f}"
            rows.append(row)
        return "\n".join(rows)


# ---------------------------------------------------------------------------
# Convenience: compute metrics from a flat list of (task_id, after_task, acc)
# ---------------------------------------------------------------------------

def compute_metrics_from_records(
    records: List[Tuple[int, int, float]],
    n_tasks: int,
    method_name: str = "",
) -> CLMetrics:
    """
    Build a CLMetricsTracker, fill it from records, and return CLMetrics.

    Parameters
    ----------
    records     : list of (task_id, after_task, accuracy) tuples
    n_tasks     : total task count
    method_name : label for the summary string
    """
    tracker = CLMetricsTracker(n_tasks=n_tasks)
    for task_id, after_task, accuracy in records:
        tracker.record(task_id, after_task, accuracy)
    metrics = tracker.compute()
    return metrics
