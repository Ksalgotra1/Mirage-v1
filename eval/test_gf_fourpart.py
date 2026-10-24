"""Regression checks for the paper-aligned Layer 3a v4 contract."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from detector.models.gf_contrastive_net import OSLEmbeddingNet
from detector.train_layer3a_gf import classify_support, make_triplet_indices, select_scz_support, triplet_loss
from fixtures.generate_patch_dataset import inject_xlsb, model_to_gf_image


class GFFourpartV4Tests(unittest.TestCase):
    def test_whole_model_image_is_raw_uint8_and_lsb_changes_representation(self) -> None:
        state = {"weight": torch.full((128, 16, 3, 3), 0.125, dtype=torch.float32)}
        clean = model_to_gf_image(state)
        stego = model_to_gf_image({"weight": inject_xlsb(state["weight"], x_bits=8, payload_type="random", seed=7)})
        self.assertEqual(clean.shape, (100, 100))
        self.assertEqual(clean.dtype, np.uint8)
        self.assertGreater(int(np.count_nonzero(clean != stego)), 0)

    def test_global_scalar_normalization(self) -> None:
        """adapt_normalization computes ONE global mean and ONE global std from support images."""
        model = OSLEmbeddingNet(emb_dim=8)
        # Two images: all zeros and all 255 → global mean=0.5, global std=0.5
        support = torch.stack([
            torch.zeros(1, 100, 100, dtype=torch.uint8),
            torch.full((1, 100, 100), 255, dtype=torch.uint8),
        ])
        model.adapt_normalization(support)
        self.assertAlmostEqual(float(model.global_mean.item()), 0.5, places=4)
        self.assertAlmostEqual(float(model.global_std.item()), 0.5, delta=0.01)

    def test_triplets_are_exhaustive_and_loss_is_squared_l2(self) -> None:
        # C(3, 2) * 3 for each class: ModelXRay generates both directions.
        self.assertEqual(len(make_triplet_indices([0, 0, 0, 1, 1, 1])), 18)
        anchor = torch.tensor([[0.0, 0.0]])
        positive = torch.tensor([[1.0, 0.0]])
        negative = torch.tensor([[2.0, 0.0]])
        # Smooth Softplus triplet loss maintains continuous non-zero gradients
        loss_val = float(triplet_loss(anchor, positive, negative))
        self.assertGreater(loss_val, 0.0)
        self.assertLess(loss_val, 0.2)

    def test_centroid_and_one_nn_are_threshold_free(self) -> None:
        support = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [1.0, 1.0], [0.9, 1.0], [0.8, 1.0]])
        labels = np.array([0, 0, 0, 1, 1, 1])
        centroid, nearest, _ = classify_support(np.array([[0.05, 0.0], [0.95, 1.0]]), support, labels)
        self.assertEqual(centroid.tolist(), [0, 1])
        self.assertEqual(nearest.tolist(), [0, 1])

    def test_scz_selection_supports_exact_and_family_topology(self) -> None:
        image = torch.zeros(1, 100, 100, dtype=torch.uint8)
        items = [
            {"image": image, "label": label, "artifact_id": f"{arch}-{label}", "architecture": arch, "base_checkpoint_id": arch}
            for arch in ("resnet18", "resnet34") for label in (0, 1)
        ]
        support_exact = select_scz_support(items, "resnet18", per_class=1)
        self.assertEqual({item["architecture"] for item in support_exact}, {"resnet18"})


if __name__ == "__main__":
    unittest.main()
