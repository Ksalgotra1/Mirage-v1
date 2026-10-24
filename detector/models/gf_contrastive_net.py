"""V5 OSL embedding network for Layer 3a — 4-Channel Differential + SRNet Stem + Moment Pooling.

Architecture (V5 — Web-Agent-Approved Patches Applied):
  - Input: Full GF-Fourpart image (H×W, e.g. 512×512)
  - 4-Channel Q2-standardized differential extraction:
      ch0 = (Q3 - μ_Q2) / σ_Q2       (LSB plane standardized by clean mantissa reference)
      ch1 = (Q3 - Q2)   / σ_Q2       (pointwise mantissa differential)
      ch2 = (HP(Q3) - HP(Q2)) / σ_Q2 (3×3 high-pass stego residual)
      ch3 = (|Q3 - μ_Q3| - |Q2 - μ_Q2|) / σ_Q2 - 0.1*(ch0 - Q1_norm)
  - SRNet-style stem: Conv(4→32)→BN→ReLU (unpooled) → Conv→BN→ReLU→AvgPool ×3
  - Global Moment Pooling: [mean, std, max, RMS] per channel → 512-D
  - Dropout(0.5) + Linear(512, emb_dim)
  - L2-normalized output (unit sphere for angular metric classification)

Fixes applied from web-agent review:
  1. Per-sample Q2-instance standardization (cancels 1/√N CLT shrinkage)
  2. RMS energy instead of raw L2 norm (resolution-invariant)
  3. Unpooled first conv block (preserves 3×3 correlations at full resolution)
  4. Per-channel (1,4,1,1) normalization buffers instead of scalar
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OSLEmbeddingNet(nn.Module):
    """V5 — 4-Channel Differential + SRNet Stem + Global Moment Pooling."""

    GF_IMAGE_SIZE = 256
    RECOMMENDED_MARGIN = 1.0
    N_MOMENT_STATS = 4  # [mean, std, max, RMS] per channel
    ARCHITECTURE_VERSION = "v5_4ch_srnet_moment"

    def __init__(self, emb_dim: int = 128, dropout: float = 0.5) -> None:
        super().__init__()

        # SRNet-style stem — Block 1 is UNPOOLED (full quadrant resolution)
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # No pool after conv1 — preserves 3×3 pixel correlations at full resolution

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.AvgPool2d(2)  # First downsampling after conv2

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool2 = nn.AvgPool2d(2)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool3 = nn.AvgPool2d(2)

        # Global Moment Pooling: 128 channels × 4 moments = 512 dim
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(128 * self.N_MOMENT_STATS, emb_dim)

        # Per-channel normalization buffers (1, 4, 1, 1) — not scalar
        self.register_buffer("global_mean", torch.zeros(1, 4, 1, 1))
        self.register_buffer("global_std", torch.ones(1, 4, 1, 1))
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
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def _extract_4ch_differential(x: torch.Tensor) -> torch.Tensor:
        """Build 4-channel scale-invariant & architecture-invariant tensor from GF-Fourpart image.

        Input x: (B, 1, H, W) where H = W = 2 * quadrant_size
        Output:  (B, 4, H/2, W/2)

        Why Q2-instance standardization solves Wall A (CLT Variance Collapse):
          In any clean model of size N_params, both Q2 (bits 9-16) and Q3 (bits 1-8)
          are uncorrupted mantissa planes whose pixel std shrinks identically as ~1/sqrt(N_params).
          Dividing by each sample's own q2_std makes clean AlexNet (61M) and ShuffleNet (1.4M)
          both have unit variance (1.0) and zero mean (0.0), while LSB payloads in Q3
          create a sharp divergence from Q2.
        """
        _, _, h, w = x.shape
        half_h, half_w = h // 2, w // 2

        q1 = x[:, :, :half_h, half_w:].float()   # Upper mantissa + exp LSB
        q2 = x[:, :, half_h:, :half_w].float()   # Middle mantissa (bits 9-16: clean internal reference)
        q3 = x[:, :, half_h:, half_w:].float()   # LSB byte plane (bits 1-8: target stego plane)

        # Per-sample reference statistics from uncorrupted mantissa plane Q2
        q2_mean = q2.mean(dim=(2, 3), keepdim=True)
        q2_std = q2.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(1e-4)
        q3_mean = q3.mean(dim=(2, 3), keepdim=True)

        q1_mean = q1.mean(dim=(2, 3), keepdim=True)
        q1_std = q1.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(1e-4)
        q1_norm = (q1 - q1_mean) / q1_std

        # Ch 0: Q3 standardized by its own model's Q2 mean & std (captures global variance/mean shift)
        ch0 = (q3 - q2_mean) / q2_std

        # Ch 1: Pointwise mantissa differential (Q3 - Q2) scaled by Q2 std
        ch1 = (q3 - q2) / q2_std

        # Ch 2: Local 3×3 high-pass stego residual of Q3 vs Q2 (captures fine bit-boundary texture)
        q3_hp = q3 - F.avg_pool2d(q3, kernel_size=3, stride=1, padding=1)
        q2_hp = q2 - F.avg_pool2d(q2, kernel_size=3, stride=1, padding=1)
        ch2 = (q3_hp - q2_hp) / q2_std

        # Ch 3: Local dispersion / entropy contrast map (|Q3 - μ_Q3| - |Q2 - μ_Q2|) / σ_Q2
        # Zero-mean for all clean models regardless of parameter count; non-zero when Q3 has stego
        ch3 = (torch.abs(q3 - q3_mean) - torch.abs(q2 - q2_mean)) / q2_std - 0.1 * (ch0 - q1_norm)

        return torch.cat([ch0, ch1, ch2, ch3], dim=1).clamp(-10.0, 10.0)

    @staticmethod
    def _global_moment_pool(x: torch.Tensor) -> torch.Tensor:
        """Compute 4 scale-normalized statistical moments per channel → (B, C * 4).

        Moments:
          1. Mean  — average activation (baseline signal level)
          2. Std   — activation spread (entropy proxy)
          3. Max   — peak activation (strongest local evidence)
          4. RMS   — root-mean-square energy (resolution-invariant, O(1))
        """
        B, C, H, W = x.shape
        flat = x.view(B, C, -1)  # (B, C, H*W)

        m_mean = flat.mean(dim=2)
        m_std = flat.std(dim=2, unbiased=False).clamp_min(1e-6)
        m_max = flat.amax(dim=2)
        # RMS energy: sqrt(mean(x^2)) — O(1) and resolution-invariant (no sqrt(H*W) factor)
        m_rms = torch.sqrt((flat ** 2).mean(dim=2) + 1e-6)

        return torch.cat([m_mean, m_std, m_max, m_rms], dim=1)  # (B, C*4)

    @torch.no_grad()
    def adapt_normalization(self, images: torch.Tensor) -> None:
        """Compute per-channel mean/std from 4-channel differential representation of support images."""
        diff_4ch = self._extract_4ch_differential(images)
        self.global_mean.copy_(diff_4ch.mean(dim=(0, 2, 3), keepdim=True))
        self.global_std.copy_(diff_4ch.std(dim=(0, 2, 3), keepdim=True, unbiased=False).clamp_min(1e-4))
        self.normalization_fitted.fill_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Extract 4-channel Q2-standardized differential tensor
        diff_4ch = self._extract_4ch_differential(x)  # (B, 4, H/2, W/2)

        # 2. Per-channel normalize using support statistics
        norm_input = (diff_4ch - self.global_mean) / self.global_std

        # 3. SRNet-style stem — Block 1 unpooled at full quadrant resolution
        h = F.relu(self.bn1(self.conv1(norm_input)))          # Full resolution
        h = self.pool1(F.relu(self.bn2(self.conv2(h))))       # /2
        h = self.pool2(F.relu(self.bn3(self.conv3(h))))       # /4
        h = self.pool3(F.relu(self.bn4(self.conv4(h))))       # /8

        # 4. Global Moment Pooling — kills spatial barcoding completely
        h = self._global_moment_pool(h)  # (B, 128*4=512)
        h = self.drop(h)

        # 5. L2-normalized embedding (unit sphere for angular distance)
        return F.normalize(self.fc(h), p=2, dim=1)


SRNetEmbeddingNet = OSLEmbeddingNet
EmbeddingNet = OSLEmbeddingNet
