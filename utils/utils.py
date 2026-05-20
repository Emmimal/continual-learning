"""
utils/utils.py
--------------
Shared utilities for the continual learning codebase.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set all random seeds for full reproducibility.

    Covers Python random, NumPy, PyTorch CPU and CUDA.
    Also sets PYTHONHASHSEED and disables CuDNN non-determinism.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(prefer_cuda: bool = True) -> str:
    """Return 'cuda' if available and preferred, else 'cpu'."""
    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def count_parameters(model: torch.nn.Module) -> dict:
    """Return total, trainable, and frozen parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "trainable_pct": 100.0 * trainable / total if total > 0 else 0.0,
    }


def format_params(n: int) -> str:
    """Format parameter count with K/M suffix."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)
