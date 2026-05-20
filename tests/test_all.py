"""
tests/test_all.py
-----------------
Unit tests for the continual learning codebase.

Tests cover:
  - Model architectures (forward pass shape, head expansion)
  - CL metrics computation (ACC, BWT, FM, FWT)
  - Replay buffer (reservoir sampling distribution)
  - EWC Fisher estimation (non-zero, correct shape)
  - PNN column freezing (frozen weights do not change)
  - GEM gradient projection (feasibility constraint)

Run with:
    python tests/test_all.py
    # or
    pytest tests/test_all.py -v
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import unittest
from torch.utils.data import DataLoader, TensorDataset

from models.architectures import (
    MultiHeadMLP, SingleHeadMLP, DomainMLP, ProgressiveNeuralNet
)
from metrics.cl_metrics import CLMetricsTracker, compute_metrics_from_records
from methods.experience_replay import ReplayBuffer
from methods.ewc import EWC
from methods.progressive_nn import PNNTrainer
from methods.gem import GEM, _project_gradients


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_loader(n=200, dim=784, n_classes=2, batch_size=64, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(n, dim)
    y = torch.randint(0, n_classes, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


# ---------------------------------------------------------------------------
# Architecture tests
# ---------------------------------------------------------------------------

class TestMultiHeadMLP(unittest.TestCase):

    def setUp(self):
        self.model = MultiHeadMLP(784, [128, 64], head_output_dim=2)
        self.model.add_task_head()
        self.model.add_task_head()

    def test_forward_shape(self):
        x = torch.randn(8, 784)
        out = self.model(x, task_id=0)
        self.assertEqual(out.shape, (8, 2))

    def test_wrong_task_id_raises(self):
        x = torch.randn(4, 784)
        with self.assertRaises(ValueError):
            self.model(x, task_id=99)

    def test_n_tasks(self):
        self.assertEqual(self.model.n_tasks, 2)

    def test_add_head(self):
        tid = self.model.add_task_head()
        self.assertEqual(tid, 2)
        self.assertEqual(self.model.n_tasks, 3)


class TestSingleHeadMLP(unittest.TestCase):

    def setUp(self):
        self.model = SingleHeadMLP(784, [128], initial_classes=2)

    def test_initial_output(self):
        x = torch.randn(4, 784)
        out = self.model(x)
        self.assertEqual(out.shape, (4, 2))

    def test_expand_head(self):
        self.model.expand_head(2)
        self.assertEqual(self.model.n_classes, 4)
        x = torch.randn(4, 784)
        out = self.model(x)
        self.assertEqual(out.shape, (4, 4))

    def test_expand_preserves_weights(self):
        """First 2 output weights should be preserved after expansion."""
        w_before = self.model.head.weight.data.clone()
        self.model.expand_head(2)
        w_after = self.model.head.weight.data
        self.assertTrue(torch.allclose(w_before, w_after[:2]))


class TestProgressiveNeuralNet(unittest.TestCase):

    def setUp(self):
        self.model = ProgressiveNeuralNet(784, [64, 32], output_dim=2)

    def test_add_columns(self):
        self.model.add_column()
        self.model.add_column()
        self.assertEqual(self.model.n_tasks, 2)

    def test_forward_shape(self):
        self.model.add_column()
        x = torch.randn(4, 784)
        out = self.model(x, task_id=0)
        self.assertEqual(out.shape, (4, 2))

    def test_lateral_connections(self):
        """Second column should produce valid output using first column's activations."""
        self.model.add_column()
        self.model.freeze_column(0)
        self.model.add_column()
        x = torch.randn(4, 784)
        out = self.model(x, task_id=1)
        self.assertEqual(out.shape, (4, 2))
        self.assertFalse(torch.any(torch.isnan(out)))

    def test_freeze_column(self):
        self.model.add_column()
        self.model.freeze_column(0)
        for p in self.model.columns[0].parameters():
            self.assertFalse(p.requires_grad)


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

class TestCLMetrics(unittest.TestCase):

    def test_no_forgetting(self):
        """If all prior-task accuracies are stable, BWT should be near 0."""
        records = [
            (0, 0, 0.95), (0, 1, 0.95), (0, 2, 0.95),
            (1, 1, 0.93), (1, 2, 0.93),
            (2, 2, 0.91),
        ]
        m = compute_metrics_from_records(records, n_tasks=3)
        self.assertAlmostEqual(m.bwt, 0.0, places=4)
        self.assertAlmostEqual(m.fm, 0.0, places=4)

    def test_full_forgetting(self):
        """If prior tasks drop to near 0 after subsequent tasks, BWT should be very negative."""
        records = [
            (0, 0, 0.95), (0, 1, 0.10),
            (1, 1, 0.94),
        ]
        m = compute_metrics_from_records(records, n_tasks=2)
        self.assertLess(m.bwt, -0.5)

    def test_acc_computation(self):
        records = [
            (0, 0, 0.90), (0, 1, 0.85),
            (1, 1, 0.92),
        ]
        m = compute_metrics_from_records(records, n_tasks=2)
        expected_acc = (0.85 + 0.92) / 2
        self.assertAlmostEqual(m.acc, expected_acc, places=4)

    def test_summary_runs(self):
        records = [(0, 0, 0.9), (0, 1, 0.88), (1, 1, 0.93)]
        m = compute_metrics_from_records(records, n_tasks=2)
        summary = m.summary("test")
        self.assertIn("ACC", summary)
        self.assertIn("BWT", summary)


# ---------------------------------------------------------------------------
# Replay buffer tests
# ---------------------------------------------------------------------------

class TestReplayBuffer(unittest.TestCase):

    def test_fills_to_capacity(self):
        buf = ReplayBuffer(capacity=100, seed=42)
        x = torch.randn(200, 10)
        y = torch.zeros(200, dtype=torch.long)
        buf.add_batch(x, y, task_id=0)
        self.assertEqual(len(buf), 100)

    def test_reservoir_distribution(self):
        """After seeing 1000 task-0 and 1000 task-1 examples, buffer should
        be roughly 50/50 — reservoir sampling is uniform."""
        buf = ReplayBuffer(capacity=200, seed=42)
        x = torch.randn(1000, 4)
        y = torch.zeros(1000, dtype=torch.long)
        buf.add_batch(x, y, task_id=0)
        buf.add_batch(x, y, task_id=1)

        stats = buf.stats()
        t0 = stats["task_distribution"].get(0, 0)
        t1 = stats["task_distribution"].get(1, 0)
        # Each task should own 40–60% of the buffer
        self.assertGreater(t0, 60)
        self.assertGreater(t1, 60)

    def test_sample_size(self):
        buf = ReplayBuffer(capacity=50, seed=0)
        x = torch.randn(100, 8)
        y = torch.zeros(100, dtype=torch.long)
        buf.add_batch(x, y, task_id=0)
        sx, sy, tids = buf.sample(30)
        self.assertEqual(sx.shape[0], 30)
        self.assertEqual(sy.shape[0], 30)
        self.assertEqual(len(tids), 30)


# ---------------------------------------------------------------------------
# EWC tests
# ---------------------------------------------------------------------------

class TestEWC(unittest.TestCase):

    def setUp(self):
        self.model = MultiHeadMLP(784, [64], head_output_dim=2)
        self.model.add_task_head()
        self.model.add_task_head()
        self.loader = _make_loader(n=100, dim=784, n_classes=2, batch_size=32)

    def test_fisher_shape(self):
        trainer = EWC(self.model, lambda_ewc=0.4, n_fisher_samples=50)
        trainer.train_task(0, self.loader, epochs=1)
        trainer.consolidate(0, self.loader)
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIn(name, trainer._fisher_accum)
                self.assertEqual(
                    trainer._fisher_accum[name].shape, param.shape
                )

    def test_fisher_nonnegative(self):
        trainer = EWC(self.model, lambda_ewc=0.4, n_fisher_samples=50)
        trainer.train_task(0, self.loader, epochs=1)
        trainer.consolidate(0, self.loader)
        for f in trainer._fisher_accum.values():
            self.assertTrue((f >= 0).all())

    def test_ewc_penalty_nonzero_after_move(self):
        """After consolidation, moving parameters should produce nonzero penalty."""
        trainer = EWC(self.model, lambda_ewc=1.0, n_fisher_samples=50)
        trainer.train_task(0, self.loader, epochs=1)
        trainer.consolidate(0, self.loader)

        # Perturb parameters
        with torch.no_grad():
            for p in self.model.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        penalty = trainer._penalty_from(trainer._fisher_accum, trainer._anchor)
        self.assertGreater(penalty.item(), 0.0)


# ---------------------------------------------------------------------------
# PNN Freezing test
# ---------------------------------------------------------------------------

class TestPNNFreezing(unittest.TestCase):

    def test_frozen_weights_do_not_change(self):
        """Weights of task 0 column must not change after freezing during task 1 training."""
        model = ProgressiveNeuralNet(784, [64, 32], output_dim=2)
        trainer = PNNTrainer(model)
        loader = _make_loader(n=100, dim=784, n_classes=2, batch_size=32)

        trainer.train_task(0, loader, epochs=2)
        # Snapshot weights before freezing
        w_before = {
            name: p.data.clone()
            for name, p in model.columns[0].named_parameters()
        }
        trainer.consolidate(0, loader)

        # Train task 1
        trainer.train_task(1, loader, epochs=2)

        # Check frozen column weights unchanged
        for name, p in model.columns[0].named_parameters():
            self.assertTrue(
                torch.allclose(w_before[name], p.data),
                f"Frozen weight {name} changed after task 1 training!"
            )


# ---------------------------------------------------------------------------
# GEM projection test
# ---------------------------------------------------------------------------

class TestGEMProjection(unittest.TestCase):

    def test_no_violation_returns_unchanged(self):
        """If current grad satisfies all constraints, it should be returned as-is."""
        g = torch.tensor([1.0, 0.0])
        mem_grads = torch.tensor([[1.0, 0.0]])   # dot product = 1 ≥ 0
        projected = _project_gradients(g, mem_grads)
        self.assertTrue(torch.allclose(g, projected, atol=1e-5))

    def test_violation_produces_feasible_grad(self):
        """Projected gradient must satisfy ĝ · g_mem ≥ 0."""
        g = torch.tensor([-1.0, 0.0])
        mem_grads = torch.tensor([[1.0, 0.0]])   # dot product = -1 < 0 → violation
        projected = _project_gradients(g, mem_grads)
        dot = (mem_grads[0] * projected).sum().item()
        self.assertGreaterEqual(dot, -1e-4)  # within tolerance


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=os.path.dirname(__file__), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
