"""Paper-aligned OSL embedding network for Layer 3a (V32b/V33 — Q3-only + L2-norm).

Architecture:
  - Input: 256×256 GF image -> crop to Q3 (bottom-right quadrant = LSB byte plane) -> 128×128
  - 3-block Conv(32→64→128, 3×3, same, relu) + MaxPool2d(2)
  - AdaptiveAvgPool2d((4, 4)) -> Flatten(2048)
  - Dropout(0.5) + Linear(2048, emb_dim)
  - L2-normalized output (unit sphere for purely angular metric classification)
  - Global scalar normalization on Q3 support pixels
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OSLEmbeddingNet(nn.Module):
    """V32b/V33 Q3-Only OSL Contrastive Network with L2 Unit-Sphere Normalization."""

    GF_IMAGE_SIZE = 256
    RECOMMENDED_MARGIN = 1.0

    def __init__(self, emb_dim: int = 128, dropout: float = 0.5) -> None:
        super().__init__()

        # 3-block conv backbone on 128×128 Q3 input
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)  # 128→64

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)  # 64→32

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2)  # 32→16

        self.gap = nn.AdaptiveAvgPool2d((4, 4))
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(128 * 16, emb_dim)

        # Global scalar normalization buffers for Q3
        self.register_buffer("global_mean", torch.tensor(0.5))
        self.register_buffer("global_std", torch.tensor(0.5))
        self.register_buffer("normalization_fitted", torch.tensor(False))

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def _extract_q3(x: torch.Tensor) -> torch.Tensor:
        """Crop to Q3 (bottom-right quadrant = LSB byte plane)."""
        _, _, h, w = x.shape
        return x[:, :, h // 2 :, w // 2 :]

    @torch.no_grad()
    def adapt_normalization(self, images: torch.Tensor) -> None:
        """Compute global mean/std from Q3 of support images only."""
        q3 = self._extract_q3(images).float() / 255.0
        self.global_mean.fill_(q3.mean().item())
        self.global_std.fill_(q3.std().clamp_min(1e-6).item())
        self.normalization_fitted.fill_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Crop to Q3 only (removes architecture fingerprint from Q0/Q1/Q2)
        q3 = self._extract_q3(x).float() / 255.0

        # 2. Global normalize using support Q3 statistics
        norm_q3 = (q3 - self.global_mean) / self.global_std

        # 3. 3-block CNN feature extraction
        h = self.pool1(F.relu(self.conv1(norm_q3)))
        h = self.pool2(F.relu(self.conv2(h)))
        h = self.pool3(F.relu(self.conv3(h)))
        h = self.gap(h).flatten(1)
        h = self.drop(h)

        # 4. L2-normalized embedding (unit sphere for angular distance)
        return F.normalize(self.fc(h), p=2, dim=1)


SRNetEmbeddingNet = OSLEmbeddingNet
EmbeddingNet = OSLEmbeddingNet
