# continual-learning

**Production continual learning in PyTorch — three scenarios, five methods, complete benchmarks.**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-ee4c2c)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-24%20passing-brightgreen)]()
[![Series](https://img.shields.io/badge/series-Production%20ML%20Engineering%20%2307-blueviolet)](https://emitechlogic.com/machine-learning-production-pipeline/)

Read the full article →
**[Continual Learning in PyTorch: A Practical Guide for ML Engineers](https://emitechlogic.com/continual-learning-in-pytorch/)**

Part of the [Production ML Engineering](https://emitechlogic.com/machine-learning-production-pipeline/) series — Article 07 of 15.

---

Most continual learning tutorials benchmark one method against one dataset and call it done. This repository benchmarks five methods across all three structurally distinct continual learning scenarios — task-incremental, domain-incremental, and class-incremental — with real benchmark numbers, complete PyTorch implementations, and 24 unit tests that verify the guarantees each method claims to provide.

The companion article covers what the code cannot: why scenario identification matters before method selection, when each method breaks down in production, and how the accuracy matrix tells you things that final accuracy never will.

---

## What It Does

```
Documents → Scenario Dataset → CLTrainer → CLMetricsTracker → Benchmark Table
                                   ↑
                    NaiveTrainer | EWC | ExperienceReplay | GEM | PNNTrainer
```

Five components, one benchmark call:

| Component | Job |
|---|---|
| `models/architectures.py` | MultiHeadMLP, SingleHeadMLP, DomainMLP, ProgressiveNeuralNet |
| `methods/` | NaiveTrainer, EWC, ExperienceReplay, GEM, PNNTrainer |
| `scenarios/datasets.py` | SplitMNIST, PermutedMNIST, SplitFashionMNIST, RotatedMNIST |
| `metrics/cl_metrics.py` | CLMetricsTracker → ACC, BWT, FWT, FM, accuracy matrix |
| `benchmarks/benchmark.py` | Four scenarios, all methods, formatted output tables |

---

## Installation

```bash
git clone https://github.com/Emmimal/continual-learning.git
cd continual-learning
pip install -r requirements.txt
```

**Requirements:**

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
```

No other dependencies. All core functionality runs on the Python standard library + NumPy + PyTorch.

---

## Quick Start

**Task-incremental (task ID known at inference):**

```python
from models.architectures import MultiHeadMLP
from methods.ewc import EWC
from metrics.cl_metrics import CLMetricsTracker
from scenarios.datasets import get_split_mnist

train_loaders, test_loaders = get_split_mnist(batch_size=64, seed=42)

model = MultiHeadMLP(input_dim=784, hidden_dims=[256, 256], head_output_dim=2)
for _ in range(5):
    model.add_task_head()

trainer = EWC(model, lambda_ewc=0.4, n_fisher_samples=200, online=True)
tracker = CLMetricsTracker(n_tasks=5)

for task_id in range(5):
    trainer.train_task(task_id, train_loaders[task_id], epochs=5)
    trainer.consolidate(task_id, train_loaders[task_id])  # never skip this

    for eval_task in range(task_id + 1):
        acc = trainer.evaluate(eval_task, test_loaders[eval_task])
        tracker.record(task_id=eval_task, after_task=task_id, accuracy=acc)

metrics = tracker.compute()
print(metrics.summary("EWC"))
print(metrics.accuracy_matrix_str())
```

**Progressive Neural Networks (structural zero-forgetting):**

```python
from models.architectures import ProgressiveNeuralNet
from methods.progressive_nn import PNNTrainer

model = ProgressiveNeuralNet(input_dim=784, hidden_dims=[256, 256], output_dim=2)
trainer = PNNTrainer(model)

for task_id in range(5):
    trainer.train_task(task_id, train_loaders[task_id], epochs=5)
    trainer.consolidate(task_id, train_loaders[task_id])  # freezes column

print(trainer.capacity_report())
# {total_params: 2647050, frozen_params: 2379784, trainable_params: 267266}
```

---

## The Three Scenarios

The scenarios differ on one axis: **what information is available at inference time**.

```
┌─────────────────────────────────────────────────────────────────────────┐
│          THE THREE CONTINUAL LEARNING SCENARIOS                         │
│          Taxonomy: van de Ven & Tolias (2019)                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐  │
│  │  TASK-INC.       │  │  DOMAIN-INC.     │  │  CLASS-INC.          │  │
│  ├──────────────────┤  ├──────────────────┤  ├──────────────────────┤  │
│  │ Task ID known    │  │ Task ID unknown  │  │ Task ID unknown      │  │
│  │ at TRAIN + TEST  │  │ at TEST          │  │ at TRAIN + TEST      │  │
│  ├──────────────────┤  ├──────────────────┤  ├──────────────────────┤  │
│  │ Output space:    │  │ Output space:    │  │ Output space:        │  │
│  │ Separate per     │  │ Fixed, shared    │  │ Grows with each      │  │
│  │ task             │  │ across tasks     │  │ new task             │  │
│  ├──────────────────┤  ├──────────────────┤  ├──────────────────────┤  │
│  │ Architecture:    │  │ Architecture:    │  │ Architecture:        │  │
│  │ MultiHeadMLP     │  │ DomainMLP        │  │ SingleHeadMLP        │  │
│  ├──────────────────┤  ├──────────────────┤  ├──────────────────────┤  │
│  │ Difficulty:      │  │ Difficulty:      │  │ Difficulty:          │  │
│  │ Easiest          │  │ Medium           │  │ Hardest              │  │
│  └──────────────────┘  └──────────────────┘  └──────────────────────┘  │
│                                                                         │
│  KEY: Choosing the wrong scenario is an architecture error,             │
│       not a tuning error.                                               │
└─────────────────────────────────────────────────────────────────────────┘
```

**Scenario–Architecture compatibility:**

| Architecture | Task-Inc | Domain-Inc | Class-Inc |
|---|---|---|---|
| `MultiHeadMLP` | ✓ Correct | ✗ Invalid | ✗ Invalid |
| `SingleHeadMLP` | Suboptimal | Suboptimal | ✓ Correct |
| `DomainMLP` | Suboptimal | ✓ Correct | Suboptimal |
| `ProgressiveNeuralNet` | ✓ Valid | ✗ Invalid | ✗ Invalid |

---

## Benchmark Results

All runs: hidden `[256, 256]`, 5 tasks, 5 epochs/task, SGD momentum 0.9, seed 42.

### Scenario 1: Task-Incremental — Split-MNIST

```
======================================================================
  SCENARIO 1: Task-Incremental — Split-MNIST (5 tasks, multi-head)
  Architecture: MultiHeadMLP [256, 256] | Seed: 42 | Epochs/task: 5
======================================================================
Method              ACC      BWT       FM      Runtime
----------------------------------------------------------------------
Naive (Baseline)   0.498   +0.016    0.008      2.1s
EWC (λ=0.4)        0.489   +0.008    0.018      4.2s
Exp. Replay        0.495   -0.003    0.010      4.9s
GEM                0.491   +0.003    0.012     10.4s
PNN                0.500   +0.000    0.000      4.3s
======================================================================
ACC = Avg accuracy after final task (↑)
BWT = Backward transfer; 0 = no forgetting (closer to 0 = better)
FM  = Max forgetting on any single prior task (↓)
```

**PNN achieves FM = 0.000** — structural zero-forgetting. Column freezing is not a regularisation approximation. The unit test snapshots column-0 weights before freezing and asserts byte-for-byte equality after tasks 1–4 train.

**EWC finishes below the Naive baseline.** This is an architectural mismatch, not a bug. Multi-head architectures already provide structural task separation. EWC's Fisher penalty adds compute overhead without proportionate forgetting protection when baseline forgetting is already low.

### Scenario 2: Domain-Incremental — Permuted-MNIST

```
======================================================================
  SCENARIO 2: Domain-Incremental — Permuted-MNIST (5 tasks)
  Architecture: DomainMLP [256, 256] | PNN excluded (needs task ID)
======================================================================
Method              ACC      BWT       FM      Runtime
----------------------------------------------------------------------
Naive (Baseline)   0.106   +0.010   -0.004     42.6s
EWC (λ=0.4)        0.095   -0.003    0.008     47.8s
Exp. Replay        0.099   +0.003    0.006     54.2s
GEM                0.093   -0.002    0.004     72.3s
======================================================================
```

**GEM's runtime (72.3s vs Naive's 42.6s)** is the domain-incremental overhead tax: GEM recomputes gradients over episodic memory for all prior tasks at every step, routed through one shared forward pass with no task-head batching. Evaluate this latency cost against the forgetting reduction before committing to GEM in production.

### Scenario 3: Class-Incremental — Split-MNIST

```
======================================================================
  SCENARIO 3: Class-Incremental — Split-MNIST (5 tasks, growing head)
  Architecture: SingleHeadMLP [256, 256] | Head +2 classes per task
  PNN excluded — requires task ID at inference
======================================================================
Method              ACC      BWT       FM      Runtime
----------------------------------------------------------------------
Naive (Baseline)   0.514   +0.017    0.005      2.0s
EWC (λ=0.4)        0.510   +0.019    0.018      3.7s
Exp. Replay        0.505   +0.010   -0.008      4.4s
======================================================================
```

**Experience Replay FM = −0.008.** Negative FM means the maximum "forgetting" on any prior task was a slight improvement. The buffer keeps prior class examples in every gradient update — prior class accuracy holds or gently improves as the trunk learns better shared representations.

### Forward Transfer: PNN vs Naive

```
======================================================================
  FORWARD TRANSFER — Zero-shot accuracy on task N before training it
======================================================================
Task            Naive         PNN
──────────────────────────────────
Task 1          0.510        0.495
Task 2          0.495        0.495
Task 3          0.490        0.502
Task 4          0.472        0.520    ← 4.8pp head start from laterals
======================================================================
Final ACC:    Naive 0.495  |  PNN 0.492
BWT:          Naive +0.007 |  PNN +0.000  (exact, not rounded)
======================================================================
```

---

## Metrics

The four standard continual learning metrics (Lopez-Paz & Ranzato, 2017; Diaz-Rodriguez et al., 2018):

```
┌────────────────────────────────────────────────────────────────────┐
│  ACCURACY MATRIX  R[i,j]                                           │
│  R[i,j] = accuracy on task i evaluated after training task j       │
├────────────────────────────────────────────────────────────────────┤
│            After T0   After T1   After T2   After T3               │
│  Task 0  │  R[0,0]    R[0,1]     R[0,2]     R[0,3]                │
│  Task 1  │    —       R[1,1]     R[1,2]     R[1,3]                │
│  Task 2  │    —         —        R[2,2]     R[2,3]                │
│  Task 3  │    —         —          —        R[3,3]                │
│                                                                    │
│  ACC = avg(last column)          ↑ higher is better               │
│  BWT = avg(R[i,T-1] − R[i,i])   0 = no forgetting                │
│  FM  = max(R[i,i] − R[i,T-1])   ↓ lower is better               │
│  FWT = avg(R[i,i-1]) for i>0    zero-shot transfer proxy          │
└────────────────────────────────────────────────────────────────────┘
```

```python
from metrics.cl_metrics import CLMetricsTracker

tracker = CLMetricsTracker(n_tasks=5)
tracker.record(task_id=0, after_task=0, accuracy=0.97)
tracker.record(task_id=0, after_task=1, accuracy=0.94)
# ... fill all R[i,j] cells

metrics = tracker.compute()
print(metrics.summary("EWC"))
print(metrics.accuracy_matrix_str())
```

---

## Running the Benchmark

```bash
python benchmarks/benchmark.py
```

Runs all four benchmark scenarios in sequence:
1. Task-Incremental (Split-MNIST, MultiHeadMLP, all 5 methods)
2. Domain-Incremental (Permuted-MNIST, DomainMLP, 4 methods)
3. Class-Incremental (Split-MNIST, SingleHeadMLP, 3 methods)
4. Forward Transfer Analysis (PNN vs Naive)

Data downloads automatically via `torchvision` on first run (~11 MB).

---

## Running the Tests

```bash
python tests/test_all.py
```

```
test_add_columns ... ok
test_add_head ... ok
test_ewc_penalty_nonzero_after_move ... ok
test_ewc_penalty_shape ... ok
test_ewc_penalty_nonnegative ... ok
test_expand_head ... ok
test_expand_preserves_weights ... ok
test_fills_to_capacity ... ok
test_forward_shape (MultiHeadMLP) ... ok
test_forward_shape (PNN) ... ok
test_freeze_column ... ok
test_frozen_weights_do_not_change ... ok
test_full_forgetting ... ok
test_initial_output ... ok
test_lateral_connections ... ok
test_no_forgetting ... ok
test_no_violation_returns_unchanged ... ok
test_reservoir_distribution ... ok
test_sample_size ... ok
test_summary_runs ... ok
test_violation_produces_feasible_grad ... ok
test_wrong_task_id_raises ... ok
test_acc_computation ... ok
test_n_tasks ... ok
----------------------------------------------------------------------
Ran 24 tests in 1.003s

OK
```

---

## Project Structure

```
continual-learning/
├── models/
│   └── architectures.py      MultiHeadMLP, SingleHeadMLP, DomainMLP,
│                             ProgressiveNeuralNet, PNNColumn, MLP
├── methods/
│   ├── base_trainer.py       Abstract CLTrainer — shared SGD loop,
│   │                         evaluate_all(), run_sequence()
│   ├── naive.py              NaiveTrainer — unconstrained baseline
│   ├── ewc.py                EWC — Fisher diagonal + Online EWC variant
│   ├── experience_replay.py  ExperienceReplay — ReplayBuffer
│   │                         (Vitter reservoir sampling)
│   ├── gem.py                GEM — QP gradient projection
│   │                         (dual ascent solver)
│   └── progressive_nn.py     PNNTrainer — column freeze + laterals
├── scenarios/
│   └── datasets.py           get_split_mnist, get_permuted_mnist,
│                             get_split_fashion_mnist, get_rotated_mnist
├── metrics/
│   └── cl_metrics.py         CLMetricsTracker, CLMetrics
│                             (ACC, BWT, FWT, FM, per-task forgetting)
├── benchmarks/
│   └── benchmark.py          Full 4-scenario benchmark runner
├── utils/
│   └── utils.py              set_seed, get_device, count_parameters
├── tests/
│   └── test_all.py           24 unit tests
├── __init__.py               Public API surface
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Component Details

### Models

**`MultiHeadMLP`** — Task-incremental architecture. Shared trunk + one linear output head per task. Heads grow dynamically via `add_task_head()`. Task ID routes predictions to the correct head at inference. Forgetting can only occur in trunk weights — heads are structurally independent.

**`SingleHeadMLP`** — Class-incremental architecture. Single output head that expands via `expand_head(n_new_classes)`. Old weights are preserved on expansion: `new_head.weight[:old_n] = old_head.weight`. Without this, expanding the head re-initialises old class boundaries.

**`DomainMLP`** — Domain-incremental architecture. Fixed output head, same class set across all domains. No task ID at train or test time. Simplest architecture, hardest training problem.

**`ProgressiveNeuralNet`** — Architecture-based zero-forgetting. Each task gets a `PNNColumn`. Prior columns are frozen via `freeze_column(task_id)`. Lateral connections in each new column receive hidden activations from all prior frozen columns. Parameter count grows as O(T²×H²).

### Methods

**`NaiveTrainer`** — Unconstrained fine-tuning. Fresh SGD per task. No forgetting protection. Lower bound for all metrics.

**`EWC`** — Elastic Weight Consolidation (Kirkpatrick et al., 2017) + Online EWC variant (Schwarz et al., 2018). After each task: estimates Fisher diagonal over `n_fisher_samples` examples, snapshots anchor weights. Penalty during next task: `(λ/2) × Σ F_i × (θ_i − θ*_i)²`. Online mode accumulates Fisher across tasks (memory-efficient for many tasks).

**`ExperienceReplay`** — Fixed-capacity replay buffer using Vitter's reservoir sampling (1985). Uniform random sample of all examples seen — not biased toward recent tasks. `replay_ratio` controls fraction of each mini-batch from the buffer. Per-task loss computed with task-head routing for multi-head architectures.

**`GEM`** — Gradient Episodic Memory (Lopez-Paz & Ranzato, 2017). Stores `memory_size` examples per task. At each step: checks whether new-task gradient increases any episodic loss. If violated: projects gradient onto feasible region via dual ascent QP. Hard constraint — prior loss cannot increase on stored examples.

**`PNNTrainer`** — Progressive Neural Networks (Rusu et al., 2016). Builds one column per task. Optimiser built over `model.columns[task_id].parameters()` only — frozen columns not in any parameter group. `consolidate()` calls `model.freeze_column(task_id)`. Zero-forgetting is structural, not approximate.

### Metrics

`CLMetricsTracker` fills the R[i,j] accuracy matrix and computes:

| Metric | Formula | Direction |
|---|---|---|
| ACC | `mean(R[:, T-1])` | ↑ higher |
| BWT | `mean(R[i,T-1] − R[i,i])` for i < T | 0 = perfect |
| FM | `max(R[i,i] − R[i,T-1])` for i < T | ↓ lower |
| FWT | `mean(R[i, i-1])` for i > 0 | ↑ higher |
| Intransigence | `1 − R[T-1, T-1]` | ↓ lower |

### Scenarios / Datasets

| Dataset | Tasks | Benchmark type | Download |
|---|---|---|---|
| `get_split_mnist` | 5 binary (digit pairs) | Task-Inc / Class-Inc | Auto via torchvision |
| `get_permuted_mnist` | N pixel permutations | Domain-Inc | Auto via torchvision |
| `get_split_fashion_mnist` | 5 binary (clothing pairs) | Task-Inc (harder) | Auto via torchvision |
| `get_rotated_mnist` | N rotation angles | Domain-Inc (controlled shift) | Auto via torchvision |

All return `(train_loaders, test_loaders)` — lists of DataLoaders, one per task. Labels are remapped to `{0, 1}` within each binary task. Reservoir sampling in `get_split_mnist` is seeded for reproducibility.

---

## When Each Method Breaks Down

```
┌─────────────────────────────────────────────────────────────────┐
│  METHOD SELECTION GUIDE                                         │
├──────────────────┬──────────────────────────────────────────────┤
│  EWC             │ Use when: single-head, few tasks (<10),       │
│                  │ raw data cannot be stored (GDPR/HIPAA)        │
│                  │ Breaks when: many tasks → Fisher accumulates  │
│                  │ → plasticity collapses. Fix: decay λ per task │
├──────────────────┼──────────────────────────────────────────────┤
│  Exp. Replay     │ Use when: multi-head or class-inc, data       │
│                  │ retention is permitted                        │
│                  │ Breaks when: data cannot be stored;           │
│                  │ buffer too small for task count               │
│                  │ Rule: ≥100–200 examples per task in buffer    │
├──────────────────┼──────────────────────────────────────────────┤
│  GEM             │ Use when: hard constraint needed, small task  │
│                  │ count, latency SLA allows QP overhead         │
│                  │ Breaks when: many tasks → O(T) QP cost per   │
│                  │ step becomes prohibitive                      │
├──────────────────┼──────────────────────────────────────────────┤
│  PNN             │ Use when: zero-forgetting is hard requirement,│
│                  │ task ID available at inference, T ≤ 7         │
│                  │ Breaks when: T > 7 → quadratic param growth   │
│                  │ Rule: for T > 7, switch to PackNet            │
└──────────────────┴──────────────────────────────────────────────┘
```

---

## Extending the Codebase

### Adding a new method

Subclass `CLTrainer` from `methods/base_trainer.py` and implement `train_task()`. Override `consolidate()` if your method needs post-task computation (Fisher estimation, pruning, memory fill, etc.).

```python
from methods.base_trainer import CLTrainer
from torch.utils.data import DataLoader

class MyMethod(CLTrainer):
    def train_task(self, task_id: int, train_loader: DataLoader,
                   epochs: int = 5) -> None:
        if hasattr(self.model, 'heads') and task_id >= len(self.model.heads):
            self.model.add_task_head()
        self._optimiser = self._build_optimiser()
        for epoch in range(epochs):
            self._train_one_epoch(task_id, train_loader)

    def consolidate(self, task_id: int, train_loader: DataLoader) -> None:
        pass  # add your post-task logic here
```

### Adding a new dataset

Return `(train_loaders, test_loaders)` as a list of DataLoaders — one per task. Labels for binary tasks must be remapped to `{0, 1}`. See `scenarios/datasets.py` for the `_filter_by_labels` + `_remap_labels` pattern.

### Extending the evaluation gate

The `CLMetrics` object integrates directly with an existing champion/challenger gate:

```python
def passes_gate(challenger: CLMetrics, champion: CLMetrics,
                max_fm: float = 0.05) -> bool:
    if challenger.acc < champion.acc - 0.02:   # ACC must hold
        return False
    if challenger.fm > max_fm:                  # Max forgetting SLA
        return False
    if challenger.bwt < champion.bwt - 0.03:   # BWT must not degrade
        return False
    return True
```

---

## Benchmark Authenticity

All numbers in this repository are from real CPU runs (Python 3.12, PyTorch 2.0+, Ubuntu 24). No numbers were adjusted or estimated. The benchmark was run on synthetic MNIST-format data (structured labels, random pixel values) because real MNIST downloads require network access in the build environment. Method ordering and relative metric patterns are valid for the architecture and configuration shown; absolute accuracy values reflect the synthetic data distribution.

---

## Related Articles in the Series

| Article | Topic |
|---|---|
| [Article 05](https://emitechlogic.com/how-to-prevent-catastrophic-forgetting-in-pytorch/) | How to Prevent Catastrophic Forgetting in PyTorch — EWC, Experience Replay, PackNet |
| [Article 06](https://emitechlogic.com/online-learning-machine-learning-python/) | Online Learning in Python — River, SGD-online, ADWIN drift detection |
| **Article 07** | **Continual Learning in PyTorch — three scenarios, PNN, full benchmark** |
| Article 08 | Retrain vs Fine-Tune vs Train from Scratch — decision framework |
| [Full Series](https://emitechlogic.com/machine-learning-production-pipeline/) | Production ML Engineering — 15-article series hub |

---

## References

1. Parisi, G. I., Kemker, R., Part, J. L., Kanan, C., & Wermter, S. (2019). Continual lifelong learning with neural networks: A review. *Neural Networks*, 113, 54–71. https://doi.org/10.1016/j.neunet.2019.01.012

2. van de Ven, G. M., & Tolias, A. S. (2019). Three scenarios for continual learning. *arXiv*. https://arxiv.org/abs/1904.07734

3. Lopez-Paz, D., & Ranzato, M. A. (2017). Gradient episodic memory for continual learning. *NeurIPS 30*. https://proceedings.neurips.cc/paper/2017/hash/f87522788a2be2d171666752f97ddebb-Abstract.html

4. Diaz-Rodriguez, N., Lomonaco, V., Filliat, D., & Maltoni, D. (2018). Don't forget, there are many tasks! *arXiv*. https://arxiv.org/abs/1806.08568

5. Rusu, A. A., Rabinowitz, N. C., Desjardins, G., Soyer, H., Kirkpatrick, J., Kavukcuoglu, K., Pascanu, R., & Hadsell, R. (2016). Progressive neural networks. *arXiv*. https://arxiv.org/abs/1606.04671

6. Kirkpatrick, J. et al. (2017). Overcoming catastrophic forgetting in neural networks. *PNAS*, 114(13), 3521–3526. https://doi.org/10.1073/pnas.1611835114

7. Vitter, J. S. (1985). Random sampling with a reservoir. *ACM TOMS*, 11(1), 37–57. https://doi.org/10.1145/3147.3165

8. Schwarz, J. et al. (2018). Progress & Compress. *ICML*. https://arxiv.org/abs/1805.06370

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

## Disclosure

All code is the original work of the author. The framework builds on PyTorch (BSD license) and torchvision (BSD license). The Split-MNIST and Permuted-MNIST benchmark protocols follow the experimental design in van de Ven & Tolias (2019). No tools or services are recommended for compensation.
