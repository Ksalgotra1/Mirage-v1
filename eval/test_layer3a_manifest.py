"""Regression tests for checkpoint-grouped Layer 3a dataset splits."""

from __future__ import annotations

import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fixtures.layer3a_manifest import audit_summary, validate_manifest


def record(artifact_id: str, base_id: str, split: str, label: str) -> dict:
    attack = {"type": "none"}
    if label == "stego":
        attack = {"type": "x_lsb", "x_bits": 8, "payload_class": "random", "payload_seed": 7}
    return {
        "artifact_id": artifact_id,
        "base_checkpoint_id": base_id,
        "filepath": f"fixtures/{artifact_id}.pt",
        "split": split,
        "architecture_family": "resnet",
        "format": "pt",
        "label": label,
        "attack": attack,
        "generator_version": "dataset-v1",
    }


class Layer3aManifestTests(unittest.TestCase):
    def test_clean_and_stego_derivatives_can_share_a_group_and_split(self):
        rows = [record("a-clean", "base-a", "train", "clean"), record("a-x8", "base-a", "train", "stego")]
        validate_manifest(rows)
        self.assertEqual(audit_summary(rows)["split_intersections"], {})

    def test_group_crossing_splits_is_rejected(self):
        rows = [record("a-clean", "base-a", "train", "clean"), record("a-x8", "base-a", "test", "stego")]
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_manifest(rows)

    def test_split_intersections_are_empty_for_independent_groups(self):
        rows = [
            record("a-clean", "base-a", "train", "clean"),
            record("a-x8", "base-a", "train", "stego"),
            record("b-clean", "base-b", "test", "clean"),
            record("b-x8", "base-b", "test", "stego"),
        ]
        summary = audit_summary(rows)
        self.assertEqual(summary["split_intersections"], {"test__train": []})


if __name__ == "__main__":
    unittest.main()
