"""Polarity-Invariant, Exchange-Antisymmetric 4-Channel SRNet + 2D Spectral & Moment Network
for Layer 3a GF-Fourpart Steganalysis.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OSLEmbeddingNet(nn.Module):
    """Polarity-Invariant Exchange-Antisymmetric 4-Channel SRNet + Spectral/Moment Branch."""

    GF_IMAGE_SIZE = 256
    RECOMMENDED_MARGIN = 1.0
    N_MOMENT_STATS = 4  # [mean, std, max, RMS] per channel
    COND_FEAT_DIM = 32
    ARCHITECTURE_VERSION = "v5_4ch_srnet_moment"

    def __init__(self, emb_dim: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.emb_dim = emb_dim

        # SRNet-style stem — Block 1 is UNPOOLED (full quadrant resolution)
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.AvgPool2d(2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool2 = nn.AvgPool2d(2)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool3 = nn.AvgPool2d(2)

        # Global Moment Pooling: 128 channels * 4 moments = 512 dim
        self.moment_norm = nn.LayerNorm(128 * self.N_MOMENT_STATS)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(128 * self.N_MOMENT_STATS, emb_dim)

        # Antisymmetric Multi-Scale Moment & Spectral Highway (32 scale-invariant features)
        self.cond_skip = nn.Linear(self.COND_FEAT_DIM, emb_dim, bias=False)
        self.cond_mlp = nn.Sequential(
            nn.Linear(self.COND_FEAT_DIM, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, emb_dim, bias=False),
        )

        # Scalar buffers for legacy unit-test contract (model.global_mean.item())
        self.register_buffer("global_mean", torch.zeros(1))
        self.register_buffer("global_std", torch.ones(1))
        # Per-channel normalization buffers (1, 4, 1, 1)
        self.register_buffer("channel_mean", torch.zeros(1, 4, 1, 1))
        self.register_buffer("channel_std", torch.ones(1, 4, 1, 1))
        self.register_buffer("cond_mean", torch.zeros(1, self.COND_FEAT_DIM))
        self.register_buffer("cond_std", torch.ones(1, self.COND_FEAT_DIM))
        self.register_buffer("normalization_fitted", torch.tensor(False))

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.8)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def _interior_crop(
        q0: torch.Tensor,
        q1: torch.Tensor,
        q2: torch.Tensor,
        q3: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Crop out the 4-pixel boundary strip so tail zero-padding and seams are ignored."""
        _, _, h, w = q3.shape
        pad = min(4, max(1, min(h, w) // 16))
        if h > 2 * pad and w > 2 * pad:
            return (
                q0[:, :, pad : h - pad, pad : w - pad],
                q1[:, :, pad : h - pad, pad : w - pad],
                q2[:, :, pad : h - pad, pad : w - pad],
                q3[:, :, pad : h - pad, pad : w - pad],
                pad,
            )
        return q0, q1, q2, q3, 0

    @classmethod
    def _split_and_harmonize(
        cls, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split 2x2 GF-Fourpart image into (q0, q1, q2, q3) in [0, 255] float32 scale."""
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x_f = x.float()
        if x_f.max() <= 1.5:
            x_f = x_f * 255.0

        _, _, h, w = x_f.shape
        half_h, half_w = h // 2, w // 2

        q0 = x_f[:, :, :half_h, :half_w]
        q1 = x_f[:, :, :half_h, half_w:]
        q2 = x_f[:, :, half_h:, :half_w]
        q3 = x_f[:, :, half_h:, half_w:]

        return q0, q1, q2, q3, q2, q3

    @classmethod
    def _extract_4ch_differential(cls, x: torch.Tensor) -> torch.Tensor:
        """Build 4 polarity-invariant, exchange-antisymmetric interior differential channels:
          ch0: Polarity-invariant midpoint deviation (|Q3 - 127.5| - |Q2 - 127.5|) / sigma_pool
          ch1: Polarity-invariant 8x8 patch deviation (|P3 - 127.5| - |P2 - 127.5|) / sigma_P
          ch2: SRM 3x3 high-pass residual contrast (|R(Q3)| - |R(Q2)|) / sigma_R
          ch3: Local 8x8 dispersion contrast (|Q3 - P3| - |Q2 - P2|) / sigma_pool
        """
        q0, q1, q2, q3, _, _ = cls._split_and_harmonize(x)
        _, _, q2_int, q3_int, pad = cls._interior_crop(q0, q1, q2, q3)

        var2 = q2_int.var(dim=(2, 3), keepdim=True, unbiased=False)
        var3 = q3_int.var(dim=(2, 3), keepdim=True, unbiased=False)
        std3 = torch.sqrt(var3 + 1e-6)
        # Disarmed zero-plane gate: if Q3 has been zeroed out by LSB sanitization (std3 < 0.5),
        # suppress differential activation so disarmed models evaluate as CLEAN.
        valid_q3 = torch.clamp((std3 - 0.25) / 1.0, 0.0, 1.0)

        sigma_pool = torch.sqrt(0.5 * (var2 + var3) + 0.25)

        # Ch 0: Polarity-invariant pixel deviation from 127.5
        ch0 = (((q3 - 127.5).abs() - (q2 - 127.5).abs()) / sigma_pool) * valid_q3

        # Ch 1: Polarity-invariant 8x8 patch deviation from 127.5
        k_patch = min(8, max(2, q3.shape[2] // 4))
        p3 = F.avg_pool2d(q3, kernel_size=k_patch, stride=1, padding=k_patch // 2)[:, :, : q3.shape[2], : q3.shape[3]]
        p2 = F.avg_pool2d(q2, kernel_size=k_patch, stride=1, padding=k_patch // 2)[:, :, : q2.shape[2], : q2.shape[3]]
        if pad > 0 and p2.shape[2] > 2 * pad:
            p2_int = p2[:, :, pad:-pad, pad:-pad]
            p3_int = p3[:, :, pad:-pad, pad:-pad]
        else:
            p2_int, p3_int = p2, p3
        p_sigma = torch.sqrt(
            0.5 * (p2_int.var(dim=(2, 3), keepdim=True, unbiased=False) + p3_int.var(dim=(2, 3), keepdim=True, unbiased=False))
            + 0.05
        )
        ch1 = (((p3 - 127.5).abs() - (p2 - 127.5).abs()) / p_sigma) * valid_q3

        # Ch 2: Antisymmetric SRM 3x3 high-pass residual magnitude differential
        r3 = q3 - F.avg_pool2d(F.pad(q3, (1, 1, 1, 1), mode="reflect"), kernel_size=3, stride=1)
        r2 = q2 - F.avg_pool2d(F.pad(q2, (1, 1, 1, 1), mode="reflect"), kernel_size=3, stride=1)
        if pad > 0 and r2.shape[2] > 2 * pad:
            r2_int = r2[:, :, pad:-pad, pad:-pad]
            r3_int = r3[:, :, pad:-pad, pad:-pad]
        else:
            r2_int, r3_int = r2, r3
        r_sigma = torch.sqrt(
            0.5 * (r2_int.var(dim=(2, 3), keepdim=True, unbiased=False) + r3_int.var(dim=(2, 3), keepdim=True, unbiased=False))
            + 0.10
        )
        ch2 = ((r3.abs() - r2.abs()) / r_sigma) * valid_q3

        # Ch 3: Antisymmetric local dispersion differential
        ch3 = (((q3 - p3).abs() - (q2 - p2).abs()) / sigma_pool) * valid_q3

        out = torch.cat([ch0, ch1, ch2, ch3], dim=1)
        if pad > 0 and out.shape[2] > 2 * pad:
            mask = torch.zeros_like(out)
            mask[:, :, pad:-pad, pad:-pad] = 1.0
            out = out * mask

        return out.clamp(-8.0, 8.0)

    @classmethod
    def _extract_conditional_stego_features(cls, x: torch.Tensor) -> torch.Tensor:
        """Extract 32 polarity-invariant, exchange-antisymmetric interior features:
          g(Q3) - g(Q2)
        Covers both:
          - Non-uniform byte payloads (Families A, B, C regardless of whether mu3 < 127.5 or mu3 > 127.5)
          - Uniform encrypted repeating payloads (Family D via multi-row variance collapse & 2D FFT harmonics)
        """
        eps = 1e-4
        q0, q1, q2, q3, _, _ = cls._split_and_harmonize(x)
        q0_i, q1_i, q2_i, q3_i, _ = cls._interior_crop(q0, q1, q2, q3)

        B, _, H, W = q3_i.shape
        q1_s = q1_i.squeeze(1)
        q2_s = q2_i.squeeze(1)
        q3_s = q3_i.squeeze(1)

        mu2 = q2_s.mean(dim=(1, 2))
        mu3 = q3_s.mean(dim=(1, 2))
        std2 = q2_s.std(dim=(1, 2), unbiased=False).clamp_min(0.1)
        std3_raw = q3_s.std(dim=(1, 2), unbiased=False)
        # Disarmed zero-plane gate: 0 when Q3 is zeroed out (std3_raw < 0.25), 1 on normal/stego models
        valid_q3 = torch.clamp((std3_raw - 0.25) / 1.0, 0.0, 1.0)
        std3 = std3_raw.clamp_min(0.1)
        sigma_pool = torch.sqrt(0.5 * (std2 ** 2 + std3 ** 2) + eps)

        feats: list[torch.Tensor] = []

        # ── Group 1: Polarity-Invariant Multi-Scale Midpoint Deviation & Log-Std Ratios (8 features) ──
        # Works identically for Low-Byte (mu3 < 127.5) AND High-Byte (mu3 > 127.5) payloads!
        feats.append(((mu3 - 127.5).abs() - (mu2 - 127.5).abs()) / sigma_pool)
        feats.append(torch.log((std3 + eps) / (std2 + eps)).abs() - 0.0)

        for ks in (4, 8, 16):
            k = min(ks, max(2, min(H, W) // 2))
            p2 = F.avg_pool2d(q2_i, kernel_size=k, stride=k).squeeze(1)
            p3 = F.avg_pool2d(q3_i, kernel_size=k, stride=k).squeeze(1)
            p_std2 = p2.std(dim=(1, 2), unbiased=False).clamp_min(0.05)
            p_std3 = p3.std(dim=(1, 2), unbiased=False).clamp_min(0.05)
            p_pool = torch.sqrt(0.5 * (p_std2 ** 2 + p_std3 ** 2) + eps)
            feats.append(((p3 - 127.5).abs().mean(dim=(1, 2)) - (p2 - 127.5).abs().mean(dim=(1, 2))) / p_pool)
            feats.append(torch.log((p_std3 + eps) / (p_std2 + eps)).abs())

        # ── Group 2: Polarity-Invariant Marginal Quantile Deviations (8 features) ──
        flat2 = q2_s.reshape(B, -1)
        flat3 = q3_s.reshape(B, -1)
        q_levels = torch.tensor([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95], device=x.device, dtype=flat2.dtype)
        quant2 = torch.quantile(flat2, q_levels, dim=1).T  # (B, 7)
        quant3 = torch.quantile(flat3, q_levels, dim=1).T  # (B, 7)
        # Compare symmetric folded quantiles |Quant_p(Q) - 127.5| between Q3 and Q2
        for qi in range(7):
            feats.append(((quant3[:, qi] - 127.5).abs() - (quant2[:, qi] - 127.5).abs()) / sigma_pool)
        iqr2 = (quant2[:, 4] - quant2[:, 2]).clamp_min(0.1)
        iqr3 = (quant3[:, 4] - quant3[:, 2]).clamp_min(0.1)
        feats.append(torch.log((iqr3 + eps) / (iqr2 + eps)).abs())

        # ── Group 3: Multi-Row Periodic Variance Collapse for Uniform Encrypted Payloads (Family D) (8 features) ──
        # When a 16 KB payload is tiled across N_w weights, row means in Q3 have collapsed variance
        # and strong multi-row periodicity regardless of whether byte mean is 127.5!
        row_q3 = q3_s.mean(dim=2)  # (B, H)
        row_q2 = q2_s.mean(dim=2)  # (B, H)
        col_q3 = q3_s.mean(dim=1)  # (B, W)
        col_q2 = q2_s.mean(dim=1)  # (B, W)

        r_std3 = row_q3.std(dim=1, unbiased=False).clamp_min(1e-3)
        r_std2 = row_q2.std(dim=1, unbiased=False).clamp_min(1e-3)
        c_std3 = col_q3.std(dim=1, unbiased=False).clamp_min(1e-3)
        c_std2 = col_q2.std(dim=1, unbiased=False).clamp_min(1e-3)

        # Signed and magnitude row-variance collapse log(r_std2 / r_std3)
        feats.append(torch.log((r_std2 + eps) / (r_std3 + eps)))
        feats.append(torch.log((r_std2 + eps) / (r_std3 + eps)).abs())
        feats.append(torch.log((c_std2 + eps) / (c_std3 + eps)).abs())

        # Multi-row moving-average variance collapse across window sizes k = 2, 4, 8, 16 rows
        for kr in (2, 4, 8, 16):
            if H >= 2 * kr:
                rm3 = F.avg_pool1d(row_q3.unsqueeze(1), kernel_size=kr, stride=1).squeeze(1)
                rm2 = F.avg_pool1d(row_q2.unsqueeze(1), kernel_size=kr, stride=1).squeeze(1)
                rms3 = rm3.std(dim=1, unbiased=False).clamp_min(1e-3)
                rms2 = rm2.std(dim=1, unbiased=False).clamp_min(1e-3)
                feats.append(torch.log((rms2 + eps) / (rms3 + eps)).abs())
            else:
                feats.append(torch.zeros(B, device=x.device, dtype=flat2.dtype))

        # Row-to-Pixel variance ratio contrast (captures 1D row periodicity vs 2D pixel variance)
        rv_ratio2 = r_std2 / std2
        rv_ratio3 = r_std3 / std3
        feats.append(torch.log((rv_ratio2 + eps) / (rv_ratio3 + eps)).abs())

        # ── Group 4: 2D FFT Bragg Harmonic Peaks & Spatial Autocorrelation Contrasts (8 features) ──
        z2_c = (q2_s - mu2[:, None, None]) / std2[:, None, None]
        z3_c = (q3_s - mu3[:, None, None]) / std3[:, None, None]

        # 2D FFT Power Spectrum peak-to-mean & top-k harmonic energy ratios
        fft2 = torch.fft.rfft2(z2_c)
        fft3 = torch.fft.rfft2(z3_c)
        psd2 = (fft2.real ** 2 + fft2.imag ** 2).reshape(B, -1)[:, 1:]  # exclude DC (0,0)
        psd3 = (fft3.real ** 2 + fft3.imag ** 2).reshape(B, -1)[:, 1:]

        psd2_norm = psd2 / (psd2.mean(dim=1, keepdim=True) + eps)
        psd3_norm = psd3 / (psd3.mean(dim=1, keepdim=True) + eps)

        # Max harmonic peak & top-5 / top-20 Bragg peak energy concentration
        feats.append(torch.log((psd3_norm.amax(dim=1) + eps) / (psd2_norm.amax(dim=1) + eps)).abs())
        k_top5 = min(5, psd3_norm.shape[1])
        k_top20 = min(20, psd3_norm.shape[1])
        top5_3 = torch.topk(psd3_norm, k_top5, dim=1).values.mean(dim=1)
        top5_2 = torch.topk(psd2_norm, k_top5, dim=1).values.mean(dim=1)
        top20_3 = torch.topk(psd3_norm, k_top20, dim=1).values.mean(dim=1)
        top20_2 = torch.topk(psd2_norm, k_top20, dim=1).values.mean(dim=1)
        feats.append(torch.log((top5_3 + eps) / (top5_2 + eps)).abs())
        feats.append(torch.log((top20_3 + eps) / (top20_2 + eps)).abs())

        # 1D Row-Profile FFT peak energy contrast
        rfft2 = torch.fft.rfft(row_q2 - row_q2.mean(dim=1, keepdim=True), dim=1)[:, 1:]
        rfft3 = torch.fft.rfft(row_q3 - row_q3.mean(dim=1, keepdim=True), dim=1)[:, 1:]
        rpsd2 = rfft2.real ** 2 + rfft2.imag ** 2
        rpsd3 = rfft3.real ** 2 + rfft3.imag ** 2
        rpsd2_n = rpsd2 / (rpsd2.mean(dim=1, keepdim=True) + eps)
        rpsd3_n = rpsd3 / (rpsd3.mean(dim=1, keepdim=True) + eps)
        feats.append(torch.log((rpsd3_n.amax(dim=1) + eps) / (rpsd2_n.amax(dim=1) + eps)).abs())

        # SRM High-Pass Residual log-ratios (dx, dy, laplacian) + Q1 correlation symmetry
        dx3 = q3_s[:, :, 1:] - q3_s[:, :, :-1]
        dx2 = q2_s[:, :, 1:] - q2_s[:, :, :-1]
        dy3 = q3_s[:, 1:, :] - q3_s[:, :-1, :]
        dy2 = q2_s[:, 1:, :] - q2_s[:, :-1, :]
        rms_dx3 = torch.sqrt((dx3 ** 2).mean(dim=(1, 2)) + eps)
        rms_dx2 = torch.sqrt((dx2 ** 2).mean(dim=(1, 2)) + eps)
        rms_dy3 = torch.sqrt((dy3 ** 2).mean(dim=(1, 2)) + eps)
        rms_dy2 = torch.sqrt((dy2 ** 2).mean(dim=(1, 2)) + eps)
        feats.append(torch.log(rms_dx3 / rms_dx2).abs())
        feats.append(torch.log(rms_dy3 / rms_dy2).abs())

        if W > 1 and H > 1:
            ac_h3 = (z3_c[:, :, 1:] * z3_c[:, :, :-1]).mean(dim=(1, 2))
            ac_h2 = (z2_c[:, :, 1:] * z2_c[:, :, :-1]).mean(dim=(1, 2))
            ac_v3 = (z3_c[:, 1:, :] * z3_c[:, :-1, :]).mean(dim=(1, 2))
            ac_v2 = (z2_c[:, 1:, :] * z2_c[:, :-1, :]).mean(dim=(1, 2))
            feats.append((ac_h3 - ac_h2).abs())
            feats.append((ac_v3 - ac_v2).abs())
        else:
            feats.append(torch.zeros(B, device=x.device, dtype=flat2.dtype))
            feats.append(torch.zeros(B, device=x.device, dtype=flat2.dtype))

        out = torch.stack(feats, dim=1) * valid_q3.unsqueeze(1)
        return torch.nan_to_num(out, nan=0.0, posinf=5.0, neginf=-5.0)

    @staticmethod
    def _global_moment_pool(x: torch.Tensor) -> torch.Tensor:
        """Compute 4 scale-normalized statistical moments per channel -> (B, C * 4)."""
        B, C, H, W = x.shape
        flat = x.view(B, C, -1)

        m_mean = flat.mean(dim=2)
        m_std = flat.std(dim=2, unbiased=False).clamp_min(1e-6)
        m_max = flat.amax(dim=2)
        m_rms = torch.sqrt((flat ** 2).mean(dim=2) + 1e-6)

        return torch.cat([m_mean, m_std, m_max, m_rms], dim=1)

    @torch.no_grad()
    def adapt_normalization(self, images: torch.Tensor) -> None:
        """Fit scalar & 4ch normalization AND 32-dim feature standardization."""
        img_f = images.float()
        img_01 = img_f / 255.0 if img_f.max() > 1.5 else img_f
        self.global_mean.copy_(img_01.mean().view(1))
        self.global_std.copy_(img_01.std(unbiased=False).clamp_min(1e-3).view(1))

        diff_4ch = self._extract_4ch_differential(images)
        self.channel_mean.copy_(diff_4ch.mean(dim=(0, 2, 3), keepdim=True))
        self.channel_std.copy_(diff_4ch.std(dim=(0, 2, 3), keepdim=True, unbiased=False).clamp_min(1e-3))

        cond_feats = self._extract_conditional_stego_features(images)
        c_mean = cond_feats.mean(dim=0, keepdim=True)
        c_std = cond_feats.std(dim=0, keepdim=True, unbiased=False)
        c_mad = (cond_feats - cond_feats.median(dim=0, keepdim=True).values).abs().mean(dim=0, keepdim=True) * 1.4826
        self.cond_mean.copy_(c_mean)
        self.cond_std.copy_(torch.maximum(c_std, c_mad).clamp_min(0.01))

        self.normalization_fitted.fill_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        diff_4ch = self._extract_4ch_differential(x)
        norm_input = 4.0 * torch.tanh(((diff_4ch - self.channel_mean) / self.channel_std) / 4.0)

        h = F.relu(self.bn1(self.conv1(norm_input)))
        h = self.pool1(F.relu(self.bn2(self.conv2(h))))
        h = self.pool2(F.relu(self.bn3(self.conv3(h))))
        h = self.pool3(F.relu(self.bn4(self.conv4(h))))

        h_mom = self._global_moment_pool(h)
        h_mom = self.drop(self.moment_norm(h_mom))
        emb_spatial = self.fc(h_mom)

        cond_raw = self._extract_conditional_stego_features(x)
        cond_z = 3.5 * torch.tanh(((cond_raw - self.cond_mean) / self.cond_std.clamp_min(0.01)) / 3.5)
        emb_cond = 0.70 * self.cond_skip(cond_z) + 0.30 * self.cond_mlp(cond_z)

        out = 0.45 * emb_spatial + 0.55 * emb_cond
        return F.normalize(out, p=2, dim=1)


SRNetEmbeddingNet = OSLEmbeddingNet
EmbeddingNet = OSLEmbeddingNet
GFContrastiveNet = OSLEmbeddingNet
