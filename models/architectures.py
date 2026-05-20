"""
models/architectures.py
-----------------------
Neural network architectures for continual learning.

Covers:
  - MultiHeadMLP          : shared trunk + per-task output heads (task-incremental)
  - SingleHeadMLP         : shared trunk + single growing output head (class-incremental)
  - ProgressiveNeuralNet  : lateral connections across task-specific columns (PNN)
  - DomainMLP             : shared trunk + single fixed head (domain-incremental)

All models expose a unified interface:
    forward(x, task_id=None) -> logits
    add_task_head()          -> None        (MultiHead / SingleHead)
    freeze_column(task_id)   -> None        (ProgressiveNeuralNet)

References
----------
[1] Rusu et al. (2016). Progressive Neural Networks. arXiv:1606.04671
[2] van de Ven & Tolias (2019). Three scenarios for continual learning. arXiv:1904.07734
"""

import torch
import torch.nn as nn
from typing import List, Optional


# ---------------------------------------------------------------------------
# Shared building block
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Simple feed-forward MLP with ReLU activations."""

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int,
                 dropout: float = 0.0):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# 1. MultiHeadMLP  —  Task-Incremental Learning
# ---------------------------------------------------------------------------

class MultiHeadMLP(nn.Module):
    """
    Shared trunk with one output head per task.

    Used in: task-incremental learning — task identity is KNOWN at inference.

    Architecture
    ------------
    Input → [shared hidden layers] → task-specific linear head → logits

    Parameters
    ----------
    input_dim       : number of input features
    hidden_dims     : list of hidden layer widths for the shared trunk
    head_output_dim : number of output units per task head (e.g. 2 for binary)
    dropout         : dropout probability (0 = disabled)
    """

    def __init__(self, input_dim: int, hidden_dims: List[int],
                 head_output_dim: int = 2, dropout: float = 0.0):
        super().__init__()
        self.head_output_dim = head_output_dim

        # Shared trunk
        trunk_layers = []
        in_dim = input_dim
        for h in hidden_dims:
            trunk_layers += [nn.Linear(in_dim, h), nn.ReLU()]
            if dropout > 0.0:
                trunk_layers.append(nn.Dropout(dropout))
            in_dim = h
        self.trunk = nn.Sequential(*trunk_layers)
        self._trunk_out_dim = in_dim

        # Task heads — grown dynamically via add_task_head()
        self.heads = nn.ModuleList()

    def add_task_head(self) -> int:
        """Append a new linear output head. Returns task index."""
        self.heads.append(nn.Linear(self._trunk_out_dim, self.head_output_dim))
        return len(self.heads) - 1

    def forward(self, x: torch.Tensor, task_id: int = 0) -> torch.Tensor:
        if task_id >= len(self.heads):
            raise ValueError(
                f"task_id={task_id} but only {len(self.heads)} head(s) exist. "
                "Call add_task_head() first."
            )
        features = self.trunk(x)
        return self.heads[task_id](features)

    @property
    def n_tasks(self) -> int:
        return len(self.heads)


# ---------------------------------------------------------------------------
# 2. SingleHeadMLP  —  Class-Incremental Learning
# ---------------------------------------------------------------------------

class SingleHeadMLP(nn.Module):
    """
    Shared trunk with a single output head that grows as new classes arrive.

    Used in: class-incremental learning — task identity is UNKNOWN at inference.
    The model must distinguish ALL classes seen so far in a single forward pass.

    Parameters
    ----------
    input_dim       : number of input features
    hidden_dims     : list of hidden layer widths
    initial_classes : number of output classes to start with
    dropout         : dropout probability
    """

    def __init__(self, input_dim: int, hidden_dims: List[int],
                 initial_classes: int = 2, dropout: float = 0.0):
        super().__init__()

        trunk_layers = []
        in_dim = input_dim
        for h in hidden_dims:
            trunk_layers += [nn.Linear(in_dim, h), nn.ReLU()]
            if dropout > 0.0:
                trunk_layers.append(nn.Dropout(dropout))
            in_dim = h
        self.trunk = nn.Sequential(*trunk_layers)
        self._trunk_out_dim = in_dim
        self._n_classes = initial_classes
        self.head = nn.Linear(in_dim, initial_classes)

    def expand_head(self, n_new_classes: int) -> None:
        """
        Add n_new_classes units to the output head.

        Copies existing weights into the expanded layer so prior class
        representations are not destroyed by random re-initialisation.
        """
        old_head = self.head
        new_n = self._n_classes + n_new_classes
        new_head = nn.Linear(self._trunk_out_dim, new_n)

        # Preserve old weights
        with torch.no_grad():
            new_head.weight[:self._n_classes] = old_head.weight
            new_head.bias[:self._n_classes] = old_head.bias

        self.head = new_head
        self._n_classes = new_n

    def forward(self, x: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        # task_id ignored — single head covers all classes
        features = self.trunk(x)
        return self.head(features)

    @property
    def n_classes(self) -> int:
        return self._n_classes


# ---------------------------------------------------------------------------
# 3. DomainMLP  —  Domain-Incremental Learning
# ---------------------------------------------------------------------------

class DomainMLP(nn.Module):
    """
    Shared trunk with a single FIXED output head.

    Used in: domain-incremental learning — the output space is the same across
    all tasks (same classes), but the input distribution shifts between tasks.
    Task identity is neither needed nor available at inference.

    Parameters
    ----------
    input_dim   : number of input features
    hidden_dims : list of hidden layer widths
    output_dim  : number of output classes (fixed, same across all domains)
    dropout     : dropout probability
    """

    def __init__(self, input_dim: int, hidden_dims: List[int],
                 output_dim: int = 2, dropout: float = 0.0):
        super().__init__()
        self.net = MLP(input_dim, hidden_dims, output_dim, dropout)

    def forward(self, x: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# 4. ProgressiveNeuralNet  —  Architecture-based Continual Learning
# ---------------------------------------------------------------------------

class PNNColumn(nn.Module):
    """
    One column of a Progressive Neural Network.

    A column is a per-task MLP. Each hidden layer receives:
      (a) its own previous layer's activations
      (b) lateral connections from ALL prior columns at the same depth

    Parameters
    ----------
    input_dim   : raw feature dimension
    hidden_dims : list of hidden widths (same for every column)
    output_dim  : number of output units for this task
    n_laterals  : number of prior columns (0 for the first column)
    """

    def __init__(self, input_dim: int, hidden_dims: List[int],
                 output_dim: int, n_laterals: int = 0):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.n_laterals = n_laterals

        # Vertical connections (standard MLP layers within this column)
        self.vertical = nn.ModuleList()
        in_dim = input_dim
        for h in hidden_dims:
            self.vertical.append(nn.Linear(in_dim, h))
            in_dim = h
        self.output_layer = nn.Linear(in_dim, output_dim)

        # Lateral connections — one linear per (prior column, layer depth)
        # lateral[depth][col_idx] maps prior column's hidden_dims[depth] → hidden_dims[depth]
        self.lateral = nn.ModuleList()
        for depth, h in enumerate(hidden_dims):
            col_laterals = nn.ModuleList()
            for _ in range(n_laterals):
                col_laterals.append(nn.Linear(h, h, bias=False))
            self.lateral.append(col_laterals)

    def forward(self, x: torch.Tensor,
                prior_hiddens: Optional[List[List[torch.Tensor]]] = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x              : input tensor (batch, input_dim)
        prior_hiddens  : list of lists  [column_idx][depth] = hidden activation tensor
                         None or empty list for the first column
        """
        h = x
        current_hiddens = []

        for depth, layer in enumerate(self.vertical):
            h = layer(h)

            # Add lateral signals from all prior columns at this depth
            if prior_hiddens:
                for col_idx, col_hidden_list in enumerate(prior_hiddens):
                    if depth < len(col_hidden_list):
                        h = h + self.lateral[depth][col_idx](col_hidden_list[depth])

            h = torch.relu(h)
            current_hiddens.append(h)

        logits = self.output_layer(h)
        return logits, current_hiddens


class ProgressiveNeuralNet(nn.Module):
    """
    Progressive Neural Network (PNN) — Rusu et al. (2016).

    Each new task instantiates a new column. Prior columns are FROZEN.
    Lateral connections transfer knowledge from old columns to the new one
    without any risk of catastrophic forgetting (frozen weights cannot change).

    Limitations
    -----------
    - Model size grows linearly with task count.
    - Task identity MUST be known at inference.
    - Not suitable for class-incremental or domain-incremental settings.

    Parameters
    ----------
    input_dim   : raw feature dimension
    hidden_dims : hidden widths used for every column
    output_dim  : output units per task column
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.columns: nn.ModuleList = nn.ModuleList()

    def add_column(self) -> int:
        """Add a new task column. Returns the task index."""
        col = PNNColumn(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim,
            n_laterals=len(self.columns),
        )
        self.columns.append(col)
        return len(self.columns) - 1

    def freeze_column(self, task_id: int) -> None:
        """Freeze all parameters in the specified column."""
        col = self.columns[task_id]
        for param in col.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor, task_id: int = 0) -> torch.Tensor:
        if task_id >= len(self.columns):
            raise ValueError(
                f"task_id={task_id} but only {len(self.columns)} column(s) exist."
            )
        prior_hiddens = []

        # Collect activations from all prior frozen columns
        with torch.no_grad():
            for col_idx in range(task_id):
                _, hiddens = self.columns[col_idx](x, prior_hiddens[:col_idx])
                prior_hiddens.append(hiddens)

        # Forward through the active column with lateral connections
        logits, _ = self.columns[task_id](x, prior_hiddens)
        return logits

    @property
    def n_tasks(self) -> int:
        return len(self.columns)

    def parameter_count(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {
            "total_params": total,
            "frozen_params": frozen,
            "trainable_params": total - frozen,
        }
