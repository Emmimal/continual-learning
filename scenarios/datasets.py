"""
scenarios/datasets.py
---------------------
Dataset generators for the three canonical continual learning scenarios.

Scenario taxonomy (van de Ven & Tolias, 2019):
  - Task-Incremental  : task ID known at train AND test time  → multi-head
  - Domain-Incremental: task ID known at train, NOT at test   → single fixed head
  - Class-Incremental : task ID NEVER available               → single growing head

Datasets implemented
--------------------
  SplitMNIST        : 5 binary tasks from MNIST digit pairs (task-incremental reference)
  PermutedMNIST     : 10 domain tasks; same classes, shuffled pixels (domain-incremental)
  SplitFashionMNIST : 5 binary tasks from FashionMNIST (harder than SplitMNIST)
  RotatedMNIST      : domain shift via rotation angle per task

All datasets return torch DataLoaders matching the interface expected by
every trainer in methods/.
"""

import os
import random
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mnist_base(root: str = "./data", train: bool = True) -> datasets.MNIST:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    return datasets.MNIST(root=root, train=train, download=True, transform=transform)


def _fashion_base(root: str = "./data", train: bool = True) -> datasets.FashionMNIST:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    return datasets.FashionMNIST(root=root, train=train, download=True, transform=transform)


def _filter_by_labels(dataset, labels: List[int]) -> Subset:
    """Return a Subset containing only samples whose targets are in `labels`."""
    targets = torch.tensor(dataset.targets)
    mask = torch.zeros(len(targets), dtype=torch.bool)
    for lbl in labels:
        mask |= (targets == lbl)
    indices = mask.nonzero(as_tuple=True)[0].tolist()
    return Subset(dataset, indices)


def _remap_labels(subset: Subset, label_map: dict) -> TensorDataset:
    """Convert a Subset into a TensorDataset with remapped binary labels."""
    xs, ys = [], []
    for x, y in subset:
        xs.append(x.view(-1))           # flatten to 784
        ys.append(label_map[int(y)])
    return TensorDataset(
        torch.stack(xs),
        torch.tensor(ys, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# 1. Split-MNIST  —  Task-Incremental reference benchmark
# ---------------------------------------------------------------------------

# Standard digit pairs used throughout the continual learning literature
SPLIT_MNIST_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]


def get_split_mnist(
    root: str = "./data",
    batch_size: int = 64,
    pairs: Optional[List[Tuple[int, int]]] = None,
    seed: int = 42,
) -> Tuple[List[DataLoader], List[DataLoader]]:
    """
    Return (train_loaders, test_loaders) for Split-MNIST.

    Each loader covers one binary classification task (digit pair).
    Labels are remapped to {0, 1} within each task.

    Parameters
    ----------
    root       : directory for data download
    batch_size : mini-batch size
    pairs      : list of (digit_a, digit_b) tuples; defaults to SPLIT_MNIST_PAIRS
    seed       : random seed for DataLoader shuffling

    Returns
    -------
    train_loaders : List[DataLoader] — one per task, training split
    test_loaders  : List[DataLoader] — one per task, test split
    """
    if pairs is None:
        pairs = SPLIT_MNIST_PAIRS

    g = torch.Generator()
    g.manual_seed(seed)

    train_base = _mnist_base(root=root, train=True)
    test_base = _mnist_base(root=root, train=False)

    train_loaders, test_loaders = [], []

    for d0, d1 in pairs:
        label_map = {d0: 0, d1: 1}

        tr_ds = _remap_labels(_filter_by_labels(train_base, [d0, d1]), label_map)
        te_ds = _remap_labels(_filter_by_labels(test_base, [d0, d1]), label_map)

        train_loaders.append(DataLoader(tr_ds, batch_size=batch_size,
                                        shuffle=True, generator=g))
        test_loaders.append(DataLoader(te_ds, batch_size=256, shuffle=False))

    return train_loaders, test_loaders


# ---------------------------------------------------------------------------
# 2. Permuted-MNIST  —  Domain-Incremental benchmark
# ---------------------------------------------------------------------------

def get_permuted_mnist(
    root: str = "./data",
    n_tasks: int = 10,
    batch_size: int = 64,
    seed: int = 42,
) -> Tuple[List[DataLoader], List[DataLoader]]:
    """
    Return (train_loaders, test_loaders) for Permuted-MNIST.

    Each task applies a fixed random pixel permutation to the original MNIST
    images. The 10-class output space is the same across all tasks — only the
    input distribution shifts. This is the canonical domain-incremental benchmark.

    The first task uses the IDENTITY permutation (no shuffle) so the initial
    task is equivalent to standard MNIST digit classification.

    Parameters
    ----------
    root    : data download directory
    n_tasks : number of domain tasks (each = one unique permutation)
    seed    : master seed; each task's permutation is seeded from seed + task_id
    """
    rng = np.random.RandomState(seed)
    permutations = [None]                               # Task 0 = identity
    for _ in range(n_tasks - 1):
        permutations.append(rng.permutation(784))

    g = torch.Generator().manual_seed(seed)

    train_loaders, test_loaders = [], []

    for perm in permutations:
        if perm is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                transforms.Lambda(lambda x: x.view(-1)),
            ])
        else:
            perm_tensor = torch.from_numpy(perm).long()
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                transforms.Lambda(lambda x: x.view(-1)[perm_tensor]),
            ])

        tr = datasets.MNIST(root=root, train=True, download=True, transform=transform)
        te = datasets.MNIST(root=root, train=False, download=True, transform=transform)

        train_loaders.append(DataLoader(tr, batch_size=batch_size,
                                        shuffle=True, generator=g))
        test_loaders.append(DataLoader(te, batch_size=256, shuffle=False))

    return train_loaders, test_loaders


# ---------------------------------------------------------------------------
# 3. Split-FashionMNIST  —  Harder task-incremental benchmark
# ---------------------------------------------------------------------------

FASHION_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
FASHION_NAMES = {
    0: "T-shirt", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
    5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot",
}


def get_split_fashion_mnist(
    root: str = "./data",
    batch_size: int = 64,
    pairs: Optional[List[Tuple[int, int]]] = None,
    seed: int = 42,
) -> Tuple[List[DataLoader], List[DataLoader]]:
    """
    Return (train_loaders, test_loaders) for Split-FashionMNIST.

    Identical structure to Split-MNIST but using FashionMNIST, which has
    visually overlapping classes and is significantly harder than digit pairs.
    Useful for stress-testing continual learning methods that look too easy
    on the digit benchmark.
    """
    if pairs is None:
        pairs = FASHION_PAIRS

    g = torch.Generator().manual_seed(seed)

    train_base = _fashion_base(root=root, train=True)
    test_base = _fashion_base(root=root, train=False)

    train_loaders, test_loaders = [], []

    for d0, d1 in pairs:
        label_map = {d0: 0, d1: 1}
        tr_ds = _remap_labels(_filter_by_labels(train_base, [d0, d1]), label_map)
        te_ds = _remap_labels(_filter_by_labels(test_base, [d0, d1]), label_map)

        train_loaders.append(DataLoader(tr_ds, batch_size=batch_size,
                                        shuffle=True, generator=g))
        test_loaders.append(DataLoader(te_ds, batch_size=256, shuffle=False))

    return train_loaders, test_loaders


# ---------------------------------------------------------------------------
# 4. Rotated-MNIST  —  Domain-Incremental with controllable shift magnitude
# ---------------------------------------------------------------------------

def get_rotated_mnist(
    root: str = "./data",
    n_tasks: int = 5,
    max_rotation: float = 180.0,
    batch_size: int = 64,
    seed: int = 42,
) -> Tuple[List[DataLoader], List[DataLoader]]:
    """
    Return (train_loaders, test_loaders) for Rotated-MNIST.

    Each task rotates the images by a fixed angle uniformly spaced in
    [0, max_rotation]. Domain shift magnitude is controlled by max_rotation.

    Parameters
    ----------
    n_tasks      : number of domain tasks
    max_rotation : maximum rotation in degrees (180 = full half-turn)
    """
    angles = np.linspace(0, max_rotation, n_tasks)
    g = torch.Generator().manual_seed(seed)

    train_loaders, test_loaders = [], []

    for angle in angles:
        transform = transforms.Compose([
            transforms.RandomRotation(degrees=(angle, angle)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.view(-1)),
        ])

        tr = datasets.MNIST(root=root, train=True, download=True, transform=transform)
        te = datasets.MNIST(root=root, train=False, download=True, transform=transform)

        train_loaders.append(DataLoader(tr, batch_size=batch_size,
                                        shuffle=True, generator=g))
        test_loaders.append(DataLoader(te, batch_size=256, shuffle=False))

    return train_loaders, test_loaders
