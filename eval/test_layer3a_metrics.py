"""Tests for checkpoint-level Layer 3a metrics."""

import unittest

from eval.layer3a_metrics import binary_metrics, bootstrap_group_ci, group_mean_metrics


class Layer3aMetricsTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"base_checkpoint_id": "a", "label": 1, "score": 0.9},
            {"base_checkpoint_id": "a", "label": 0, "score": 0.1},
            {"base_checkpoint_id": "b", "label": 1, "score": 0.4},
            {"base_checkpoint_id": "b", "label": 0, "score": 0.8},
        ]

    def test_patch_metrics(self):
        result = binary_metrics(self.rows)
        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["fp"], 1)
        self.assertEqual(result["fn"], 1)
        self.assertEqual(result["tn"], 1)

    def test_group_average_gives_equal_checkpoint_weight(self):
        result = group_mean_metrics(self.rows)
        self.assertEqual(result["groups"], 2)
        self.assertAlmostEqual(result["balanced_accuracy"], 0.5)

    def test_group_bootstrap_is_reproducible(self):
        self.assertEqual(bootstrap_group_ci(self.rows, iterations=100), bootstrap_group_ci(self.rows, iterations=100))


if __name__ == "__main__":
    unittest.main()
