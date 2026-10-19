"""Testes pequenos da equivalência do gradiente sem depender do dataset TLC."""
import unittest
import numpy as np
from train import loss_and_gradient, split_partitions, train_distributed, train_sequential


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[-1.0, 0.2], [0.4, 1.1], [1.2, -0.5], [0.7, 0.9], [-0.3, -1.0]])
        self.y = np.array([0., 1., 1., 1., 0.])

    def test_weighted_partition_gradients_equal_full_gradient(self):
        weights = np.array([0.1, -0.2, 0.3])
        _, expected = loss_and_gradient(self.X, self.y, weights)
        partitions = split_partitions(self.X, self.y, 3)
        local = []
        from train import _partition_metrics
        for xp, yp in partitions:
            local.append(_partition_metrics(xp, yp, weights))
        actual = sum((g for _, g, _ in local), start=np.zeros_like(weights)) / sum(n for n, _, _ in local)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_sequential_matches_real_processes(self):
        seq, seq_loss, _ = train_sequential(self.X, self.y, 4, 0.1)
        dist, dist_loss, _ = train_distributed(self.X, self.y, 2, 4, 0.1)
        np.testing.assert_allclose(dist, seq, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(dist_loss, seq_loss, rtol=1e-12, atol=1e-12)

    def test_partition_sizes_cover_each_row_once(self):
        parts = split_partitions(self.X, self.y, 8)
        self.assertEqual(sum(len(yp) for _, yp in parts), len(self.y))
        self.assertEqual(len(parts), len(self.y))


if __name__ == "__main__":
    unittest.main()
