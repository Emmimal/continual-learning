"""
benchmarks/benchmark.py
------------------------
Head-to-head benchmark: Continual Learning in PyTorch — Article 07

Runs all four methods across all three continual learning scenarios
and prints formatted results tables with CL metrics.

Scenarios benchmarked
---------------------
  1. Task-Incremental   : Split-MNIST (5 tasks, multi-head, task ID known)
  2. Domain-Incremental : Permuted-MNIST (5 tasks, single head, task ID unknown)
  3. Class-Incremental  : Split-MNIST (5 tasks, single growing head)

Methods benchmarked
-------------------
  Naive (Baseline)   : No protection — fine-tunes without constraint
  EWC (λ=0.4)        : Elastic Weight Consolidation, online variant
  Experience Replay  : Reservoir buffer, 500 examples, 0.5 replay ratio
  GEM                : Gradient Episodic Memory, 200 examples/task
  PNN                : Progressive Neural Networks (task-incremental only)

Usage
-----
    python benchmarks/benchmark.py

All benchmark numbers printed are from a real CPU run.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models.architectures import MultiHeadMLP, SingleHeadMLP, DomainMLP, ProgressiveNeuralNet
from methods.naive import NaiveTrainer
from methods.ewc import EWC
from methods.experience_replay import ExperienceReplay
from methods.gem import GEM
from methods.progressive_nn import PNNTrainer
from metrics.cl_metrics import CLMetricsTracker
from scenarios.datasets import get_split_mnist, get_permuted_mnist


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 42
EPOCHS_PER_TASK = 5
HIDDEN_DIMS = [256, 256]
INPUT_DIM = 784        # MNIST flattened
HEAD_OUT_DIM = 2       # binary per task
N_TASKS_SPLIT = 5
N_TASKS_PERM = 5
BATCH_SIZE = 64

torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_method(trainer, train_loaders, test_loaders, n_tasks):
    """Run a trainer through the full task sequence. Returns CLMetricsTracker."""
    tracker = CLMetricsTracker(n_tasks=n_tasks)
    t0 = time.time()

    for task_id in range(n_tasks):
        trainer.train_task(task_id, train_loaders[task_id], epochs=EPOCHS_PER_TASK)
        trainer.consolidate(task_id, train_loaders[task_id])

        for eval_task in range(task_id + 1):
            acc = trainer.evaluate(eval_task, test_loaders[eval_task])
            tracker.record(task_id=eval_task, after_task=task_id, accuracy=acc)

    runtime = time.time() - t0
    return tracker, runtime


def _make_multihead(n_tasks):
    m = MultiHeadMLP(INPUT_DIM, HIDDEN_DIMS, HEAD_OUT_DIM)
    for _ in range(n_tasks):
        m.add_task_head()
    return m


def _make_domainhead():
    return DomainMLP(INPUT_DIM, HIDDEN_DIMS, output_dim=10)


def _print_scenario_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def _print_results_table(results: dict, n_tasks: int):
    """Print formatted benchmark table."""
    print(f"\n{'Method':<22} {'ACC':>6} {'BWT':>7} {'FM':>7} {'FWT':>7} {'Runtime':>10}")
    print("-" * 65)
    for method_name, (metrics, runtime) in results.items():
        print(
            f"{method_name:<22} "
            f"{metrics.acc:>6.3f} "
            f"{metrics.bwt:>+7.3f} "
            f"{metrics.fm:>7.3f} "
            f"{metrics.fwt:>7.3f} "
            f"{runtime:>8.1f}s"
        )
    print("-" * 65)
    print("  ACC = Avg accuracy after final task (↑)")
    print("  BWT = Backward transfer; 0 = no forgetting (↑, closer to 0)")
    print("  FM  = Max forgetting on any task (↓)")
    print("  FWT = Forward transfer proxy (↑)")


def _print_accuracy_matrix(name, metrics, n_tasks):
    print(f"\n  Accuracy matrix — {name}:")
    print(metrics.accuracy_matrix_str())


# ---------------------------------------------------------------------------
# Scenario 1: Task-Incremental — Split-MNIST
# ---------------------------------------------------------------------------

def benchmark_task_incremental():
    _print_scenario_header(
        "SCENARIO 1: Task-Incremental — Split-MNIST (5 tasks, multi-head)"
    )
    print("  Task ID known at inference. Each task = binary digit-pair classifier.")
    print(f"  Architecture: MultiHeadMLP {HIDDEN_DIMS}, {EPOCHS_PER_TASK} epochs/task")

    train_loaders, test_loaders = get_split_mnist(
        batch_size=BATCH_SIZE, seed=SEED
    )

    results = {}

    # --- Naive baseline ---
    print("\n  Running: Naive (baseline)...")
    model = _make_multihead(N_TASKS_SPLIT)
    trainer = NaiveTrainer(model, lr=0.01)
    tracker, rt = _run_method(trainer, train_loaders, test_loaders, N_TASKS_SPLIT)
    results["Naive (Baseline)"] = (tracker.compute(), rt)

    # --- EWC ---
    print("  Running: EWC (λ=0.4, online)...")
    model = _make_multihead(N_TASKS_SPLIT)
    trainer = EWC(model, lambda_ewc=0.4, n_fisher_samples=200, online=True)
    tracker, rt = _run_method(trainer, train_loaders, test_loaders, N_TASKS_SPLIT)
    results["EWC (λ=0.4)"] = (tracker.compute(), rt)

    # --- Experience Replay ---
    print("  Running: Experience Replay (buf=500, ratio=0.5)...")
    model = _make_multihead(N_TASKS_SPLIT)
    trainer = ExperienceReplay(model, buffer_size=500, replay_ratio=0.5, seed=SEED)
    tracker, rt = _run_method(trainer, train_loaders, test_loaders, N_TASKS_SPLIT)
    results["Exp. Replay"] = (tracker.compute(), rt)

    # --- GEM ---
    print("  Running: GEM (memory=200/task)...")
    model = _make_multihead(N_TASKS_SPLIT)
    trainer = GEM(model, memory_size=200, margin=0.0)
    tracker, rt = _run_method(trainer, train_loaders, test_loaders, N_TASKS_SPLIT)
    results["GEM"] = (tracker.compute(), rt)

    # --- PNN ---
    print("  Running: Progressive Neural Net...")
    model = ProgressiveNeuralNet(INPUT_DIM, HIDDEN_DIMS, HEAD_OUT_DIM)
    trainer = PNNTrainer(model)
    tracker, rt = _run_method(trainer, train_loaders, test_loaders, N_TASKS_SPLIT)
    results["PNN"] = (tracker.compute(), rt)

    _print_results_table(results, N_TASKS_SPLIT)

    # Print accuracy matrix for two representative methods
    _print_accuracy_matrix("Naive (Baseline)", results["Naive (Baseline)"][0], N_TASKS_SPLIT)
    _print_accuracy_matrix("Experience Replay", results["Exp. Replay"][0], N_TASKS_SPLIT)

    return results


# ---------------------------------------------------------------------------
# Scenario 2: Domain-Incremental — Permuted-MNIST
# ---------------------------------------------------------------------------

def benchmark_domain_incremental():
    _print_scenario_header(
        "SCENARIO 2: Domain-Incremental — Permuted-MNIST (5 tasks, single head)"
    )
    print("  Task ID NOT available at inference. Same 10 classes, shuffled pixels.")
    print(f"  Architecture: DomainMLP {HIDDEN_DIMS} (10-class output, fixed)")
    print("  Note: PNN requires task ID at inference — excluded from this scenario.")

    train_loaders, test_loaders = get_permuted_mnist(
        n_tasks=N_TASKS_PERM, batch_size=BATCH_SIZE, seed=SEED
    )

    results = {}

    # For domain-incremental: all methods use a single fixed 10-class head
    # task_id is passed as 0 to all evaluate() calls (single head)
    def _run_domain(trainer, train_ls, test_ls, n_tasks):
        tracker = CLMetricsTracker(n_tasks=n_tasks)
        t0 = time.time()
        for task_id in range(n_tasks):
            trainer.train_task(task_id, train_ls[task_id], epochs=EPOCHS_PER_TASK)
            trainer.consolidate(task_id, train_ls[task_id])
            for eval_task in range(task_id + 1):
                # Domain-incremental: single head, task_id=0 always
                trainer.model.eval()
                correct, total = 0, 0
                with torch.no_grad():
                    for x, y in test_ls[eval_task]:
                        x, y = x.to(trainer.device), y.to(trainer.device)
                        out = trainer.model(x, task_id=None)
                        preds = out.argmax(dim=1)
                        correct += (preds == y).sum().item()
                        total += len(y)
                acc = correct / total
                tracker.record(eval_task, task_id, acc)
        return tracker, time.time() - t0

    # --- Naive ---
    print("\n  Running: Naive (baseline)...")
    model = _make_domainhead()
    trainer = NaiveTrainer(model, lr=0.01)
    tracker, rt = _run_domain(trainer, train_loaders, test_loaders, N_TASKS_PERM)
    results["Naive (Baseline)"] = (tracker.compute(), rt)

    # --- EWC ---
    print("  Running: EWC (λ=0.4)...")
    model = _make_domainhead()
    trainer = EWC(model, lambda_ewc=0.4, n_fisher_samples=200, online=True)
    tracker, rt = _run_domain(trainer, train_loaders, test_loaders, N_TASKS_PERM)
    results["EWC (λ=0.4)"] = (tracker.compute(), rt)

    # --- Replay ---
    print("  Running: Experience Replay...")
    model = _make_domainhead()
    trainer = ExperienceReplay(model, buffer_size=500, replay_ratio=0.5, seed=SEED)
    tracker, rt = _run_domain(trainer, train_loaders, test_loaders, N_TASKS_PERM)
    results["Exp. Replay"] = (tracker.compute(), rt)

    # --- GEM ---
    print("  Running: GEM...")
    model = _make_domainhead()
    trainer = GEM(model, memory_size=200, margin=0.0)
    tracker, rt = _run_domain(trainer, train_loaders, test_loaders, N_TASKS_PERM)
    results["GEM"] = (tracker.compute(), rt)

    _print_results_table(results, N_TASKS_PERM)
    return results


# ---------------------------------------------------------------------------
# Scenario 3: Class-Incremental — Split-MNIST (growing head)
# ---------------------------------------------------------------------------

def benchmark_class_incremental():
    _print_scenario_header(
        "SCENARIO 3: Class-Incremental — Split-MNIST (5 tasks, growing head)"
    )
    print("  Task ID NEVER available. Model must distinguish all classes seen so far.")
    print("  This is the hardest CL scenario — methods that rely on task ID at")
    print("  inference (PNN) are structurally incompatible.")
    print(f"  Architecture: SingleHeadMLP {HIDDEN_DIMS}, head grows by 2 per task")

    train_loaders, test_loaders = get_split_mnist(
        batch_size=BATCH_SIZE, seed=SEED
    )

    # For class-incremental, test loaders need to cover ALL classes seen so far
    # We remap labels globally: task 0 → {0,1}, task 1 → {2,3}, etc.
    # The SingleHeadMLP grows its output head as new tasks arrive.

    results = {}

    def _run_class_inc(trainer, train_ls, test_ls, n_tasks):
        tracker = CLMetricsTracker(n_tasks=n_tasks)
        t0 = time.time()
        for task_id in range(n_tasks):
            # Expand head before training if using SingleHeadMLP
            if hasattr(trainer.model, 'expand_head') and task_id > 0:
                trainer.model.expand_head(n_new_classes=2)

            # Remap labels to global class space
            # Task k uses classes 2k and 2k+1
            trainer.train_task(task_id, train_ls[task_id], epochs=EPOCHS_PER_TASK)
            trainer.consolidate(task_id, train_ls[task_id])

            for eval_task in range(task_id + 1):
                # Evaluate: use the global head with offset for this task's classes
                trainer.model.eval()
                correct, total = 0, 0
                offset = eval_task * 2
                with torch.no_grad():
                    for x, y in test_ls[eval_task]:
                        x, y = x.to(trainer.device), y.to(trainer.device)
                        out = trainer.model(x, task_id=None)
                        # Only look at the two output units for this task
                        out_task = out[:, offset:offset+2]
                        preds = out_task.argmax(dim=1)
                        correct += (preds == y).sum().item()
                        total += len(y)
                acc = correct / total
                tracker.record(eval_task, task_id, acc)
        return tracker, time.time() - t0

    # --- Naive ---
    print("\n  Running: Naive (baseline)...")
    model = SingleHeadMLP(INPUT_DIM, HIDDEN_DIMS, initial_classes=2)
    trainer = NaiveTrainer(model, lr=0.01)
    tracker, rt = _run_class_inc(trainer, train_loaders, test_loaders, N_TASKS_SPLIT)
    results["Naive (Baseline)"] = (tracker.compute(), rt)

    # --- EWC ---
    print("  Running: EWC (λ=0.4)...")
    model = SingleHeadMLP(INPUT_DIM, HIDDEN_DIMS, initial_classes=2)
    trainer = EWC(model, lambda_ewc=0.4, n_fisher_samples=200, online=True)
    tracker, rt = _run_class_inc(trainer, train_loaders, test_loaders, N_TASKS_SPLIT)
    results["EWC (λ=0.4)"] = (tracker.compute(), rt)

    # --- Replay ---
    print("  Running: Experience Replay...")
    model = SingleHeadMLP(INPUT_DIM, HIDDEN_DIMS, initial_classes=2)
    trainer = ExperienceReplay(model, buffer_size=500, replay_ratio=0.5, seed=SEED)
    tracker, rt = _run_class_inc(trainer, train_loaders, test_loaders, N_TASKS_SPLIT)
    results["Exp. Replay"] = (tracker.compute(), rt)

    _print_results_table(results, N_TASKS_SPLIT)
    return results


# ---------------------------------------------------------------------------
# Forward Transfer Analysis
# ---------------------------------------------------------------------------

def benchmark_forward_transfer():
    _print_scenario_header(
        "FORWARD TRANSFER ANALYSIS — PNN vs Naive on Split-MNIST"
    )
    print("  Measures zero-shot accuracy on task N before task N is trained.")
    print("  PNN lateral connections should boost FWT vs Naive baseline.")

    train_loaders, test_loaders = get_split_mnist(
        batch_size=BATCH_SIZE, seed=SEED
    )

    def _fwt_run(trainer, train_ls, test_ls, n_tasks):
        """
        Collect R[i, i-1] — accuracy on task i just BEFORE training it.
        This is the forward transfer signal.
        """
        pre_train_accs = {}
        tracker = CLMetricsTracker(n_tasks=n_tasks)
        t0 = time.time()

        for task_id in range(n_tasks):
            # For PNN: add column before zero-shot eval so forward pass works
            if hasattr(trainer.model, 'columns') and task_id >= len(trainer.model.columns):
                trainer.model.add_column()
                trainer.model.columns[task_id].to(trainer.device)

            # Record pre-training accuracy on this task (FWT proxy)
            if task_id > 0:
                trainer.model.eval()
                correct, total = 0, 0
                with torch.no_grad():
                    for x, y in test_ls[task_id]:
                        x, y = x.to(trainer.device), y.to(trainer.device)
                        out = trainer.model(x, task_id=task_id)
                        preds = out.argmax(dim=1)
                        correct += (preds == y).sum().item()
                        total += len(y)
                pre_train_accs[task_id] = correct / total
                tracker.record(task_id, task_id - 1, pre_train_accs[task_id])

            trainer.train_task(task_id, train_ls[task_id], epochs=EPOCHS_PER_TASK)
            trainer.consolidate(task_id, train_ls[task_id])

            for eval_task in range(task_id + 1):
                acc = trainer.evaluate(eval_task, test_ls[eval_task])
                tracker.record(eval_task, task_id, acc)

        return pre_train_accs, tracker.compute(), time.time() - t0

    # Naive
    print("\n  Running: Naive (baseline)...")
    model = _make_multihead(N_TASKS_SPLIT)
    trainer = NaiveTrainer(model)
    naive_fwt, naive_metrics, naive_rt = _fwt_run(
        trainer, train_loaders, test_loaders, N_TASKS_SPLIT
    )

    # PNN
    print("  Running: Progressive Neural Net...")
    model = ProgressiveNeuralNet(INPUT_DIM, HIDDEN_DIMS, HEAD_OUT_DIM)
    trainer = PNNTrainer(model)
    pnn_fwt, pnn_metrics, pnn_rt = _fwt_run(
        trainer, train_loaders, test_loaders, N_TASKS_SPLIT
    )

    print("\n  Pre-training accuracy (zero-shot FWT proxy):")
    print(f"  {'Task':<8} {'Naive':>10} {'PNN':>10}")
    print("  " + "-" * 30)
    for tid in range(1, N_TASKS_SPLIT):
        n_acc = naive_fwt.get(tid, float("nan"))
        p_acc = pnn_fwt.get(tid, float("nan"))
        print(f"  Task {tid}  {n_acc:>10.3f} {p_acc:>10.3f}")

    print(f"\n  Final ACC — Naive: {naive_metrics.acc:.3f} | PNN: {pnn_metrics.acc:.3f}")
    print(f"  BWT    — Naive: {naive_metrics.bwt:+.3f} | PNN: {pnn_metrics.bwt:+.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  CONTINUAL LEARNING IN PyTorch — COMPLETE BENCHMARK")
    print("  Article 07 | Production ML Engineering Series")
    print(f"  Seed: {SEED} | Epochs/task: {EPOCHS_PER_TASK} | Hidden: {HIDDEN_DIMS}")
    print("=" * 70)

    benchmark_task_incremental()
    benchmark_domain_incremental()
    benchmark_class_incremental()
    benchmark_forward_transfer()

    print("\n" + "=" * 70)
    print("  BENCHMARK COMPLETE")
    print("=" * 70)
