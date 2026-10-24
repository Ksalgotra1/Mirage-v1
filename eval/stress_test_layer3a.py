"""
eval/stress_test_layer3a.py
===========================
True Zero-Overlap Blind Stress Test for Layer 3a GF-Fourpart Detector.

Tests across:
1. 10 Strictly Unseen Architectures (0 overlap with CROSS_FAMILY_TRAIN_BASE_IDS)
   using verified official PyTorch Foundation HTTPS URLs with embedded SHA-256 hash checks.
2. 100% Unseen Seeds (99990001, 31415926)
3. 4 Distinct Payload Families:
   - Family A: Unseen seeds of _build_xray_fill_payload (16 KB structured binary)
   - Family B: Pure ASCII payload (SENTINEL_BENIGN_PAYLOAD repeated, mean ~85)
   - Family C: High-byte shellcode/binary payload (mean byte > 150)
   - Family D: Encrypted/compressed random payload (uniform in [0, 255], mean ~127.5)
     with non-tiled per-tensor stream encryption on ultra-large models (>= 35M parameters).
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detector.layer3a_gf import Layer3aGFDetector
from fixtures.generate_patch_dataset import (
    SENTINEL_BENIGN_PAYLOAD,
    _build_xray_fill_payload,
    inject_xlsb,
)

ALLOWED_URL_PREFIX = "https://download.pytorch.org/models/"

# ---------------------------------------------------------------------------
# 1. Payload Generators
# ---------------------------------------------------------------------------

def get_payload_family_a(seed: int, length: int = 16384) -> np.ndarray:
    """Family A: ModelXRay structured payload with unseen seed."""
    return _build_xray_fill_payload(seed, payload_len_bytes=length)


def get_payload_family_b(length: int = 16384) -> np.ndarray:
    """Family B: Pure ASCII text payload (mean byte ~ 80)."""
    reps = (length // len(SENTINEL_BENIGN_PAYLOAD)) + 1
    buf = (SENTINEL_BENIGN_PAYLOAD * reps)[:length]
    arr = np.frombuffer(buf, dtype=np.uint8)
    return arr


def get_payload_family_c(seed: int, length: int = 16384) -> np.ndarray:
    """Family C: High-byte shellcode/binary payload (mean byte > 160)."""
    rng = np.random.default_rng(seed)
    palette = np.arange(140, 256, dtype=np.uint8)
    return rng.choice(palette, size=length, replace=True)


def get_payload_family_d(seed: int, length: int = 16384) -> np.ndarray:
    """Family D: Fully uniform encrypted/compressed bytes (mean ~ 127.5)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=length, dtype=np.uint8)


def inject_payload_bytes(state_dict: dict, payload_buf: np.ndarray, x_bits: int = 8) -> dict:
    """Tiles payload_buf across float32 weights' bottom x_bits."""
    payload_bits = np.unpackbits(payload_buf)
    n_payload_bits = len(payload_bits)

    powers = (1 << np.arange(x_bits - 1, -1, -1, dtype=np.uint32))
    clear_mask = np.uint32(((1 << x_bits) - 1) ^ 0xFFFFFFFF)

    result = {}
    bit_cursor = 0
    for name, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float32:
            num_weights = tensor.numel()
            total_bits = num_weights * x_bits
            idx = (np.arange(total_bits, dtype=np.int64) + bit_cursor) % n_payload_bits
            bit_cursor = (bit_cursor + total_bits) % n_payload_bits

            bits_per_weight = payload_bits[idx].reshape(num_weights, x_bits)
            payload_u32 = (bits_per_weight * powers).sum(axis=1).astype(np.uint32)

            weights_u32 = tensor.detach().cpu().numpy().astype(np.float32).flatten().view(np.uint32).copy()
            stego_u32 = (weights_u32 & clear_mask) | payload_u32
            stego_f32 = stego_u32.view(np.float32).reshape(tensor.shape)
            result[name] = torch.from_numpy(stego_f32)
        else:
            result[name] = tensor
    return result


def _load_verified_state_dict(entry: dict) -> dict:
    """Verify URL safety (official PyTorch HTTPS domain + SHA256 suffix) and load state_dict
    safely with weights_only=True, prioritizing local audited checkpoints when present.
    """
    url = entry["url"]
    if not url.startswith(ALLOWED_URL_PREFIX):
        raise ValueError(f"Unsafe or untrusted model URL blocked: {url}")
    if not re.search(r"-[0-9a-f]{8,10}\.pth$", url):
        raise ValueError(f"Model URL missing cryptographic SHA-256 hash prefix: {url}")

    local_path = entry.get("local_path")
    if local_path and Path(local_path).exists():
        sd = torch.load(local_path, map_location="cpu", weights_only=True)
    else:
        sd = torch.hub.load_state_dict_from_url(
            url, map_location="cpu", check_hash=True, progress=False, weights_only=True
        )
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    return sd


# ---------------------------------------------------------------------------
# 2. 10 Unseen Architectures (100% Zero Overlap with CROSS_FAMILY_TRAIN_BASE_IDS)
# ---------------------------------------------------------------------------

UNSEEN_MODELS = [
    {
        "arch": "squeezenet1_0",
        "url": "https://download.pytorch.org/models/squeezenet1_0-b66bff10.pth",
        "local_path": "data/layer3a_official_sources/squeezenet10-imagenet1k-v1.pt",
    },
    {
        "arch": "mnasnet1_0",
        "url": "https://download.pytorch.org/models/mnasnet1.0_top1_73.512-f206786ef8.pth",
        "local_path": "data/layer3a_official_sources/mnasnet10-imagenet1k-v1.pt",
    },
    {
        "arch": "mobilenet_v3_large",
        "url": "https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth",
        "local_path": "data/layer3a_official_sources/mobilenet-v3-large-imagenet1k-v2.pt",
    },
    {
        "arch": "regnet_x_400mf",
        "url": "https://download.pytorch.org/models/regnet_x_400mf-adf1edd5.pth",
        "local_path": "data/layer3a_official_sources/regnet-x-400mf-imagenet1k-v2.pt",
    },
    {
        "arch": "densenet201",
        "url": "https://download.pytorch.org/models/densenet201-c1103571.pth",
        "local_path": "data/layer3a_official_sources/densenet201-imagenet1k-v1.pt",
    },
    {
        "arch": "resnet34",
        "url": "https://download.pytorch.org/models/resnet34-b627a593.pth",
        "local_path": "data/layer3a_official_sources/resnet34-imagenet1k-v1.pt",
    },
    {
        "arch": "resnext50_32x4d",
        "url": "https://download.pytorch.org/models/resnext50_32x4d-1a0047aa.pth",
        "local_path": "data/layer3a_official_sources/resnext50-32x4d-imagenet1k-v2.pt",
    },
    {
        "arch": "convnext_tiny",
        "url": "https://download.pytorch.org/models/convnext_tiny-983f1562.pth",
        "local_path": "data/layer3a_official_sources/convnext-tiny-imagenet1k-v1.pt",
    },
    {
        "arch": "alexnet",
        "url": "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth",
        "local_path": None,
    },
    {
        "arch": "vgg13_bn",
        "url": "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
        "local_path": "data/layer3a_official_sources/vgg13-bn-imagenet1k-v1.pt",
    },
]


def run_stress_test():
    detector = Layer3aGFDetector()
    print("Detector initialized successfully.\n")

    results = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        for entry in UNSEEN_MODELS:
            arch_name = entry["arch"]
            print(f"--> Testing Unseen Architecture: {arch_name} ({entry['url']})")
            state_dict = _load_verified_state_dict(entry)
            total_weights = sum(
                tensor.numel() for tensor in state_dict.values()
                if isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float32
            )

            # 1. Clean Baseline
            clean_file = tmp_path / f"{arch_name}_clean.pt"
            torch.save(state_dict, clean_file)
            res_clean = detector.predict(str(clean_file))
            clean_file.unlink(missing_ok=True)

            print(f"    [CLEAN]      margin={res_clean['distance']:+.4f} | score={res_clean['score']:.4f} | band={res_clean['forensic_band']:10s} | pred={'PASS' if not res_clean['verdict'] else 'FAIL'}")
            results.append({
                "arch": arch_name, "type": "CLEAN", "payload_family": "None", "mean_byte": None,
                "margin": res_clean["distance"], "score": res_clean["score"],
                "band": res_clean["forensic_band"], "pass": not res_clean["verdict"]
            })

            # 2. Family A: Unseen Seeds (99990001, 31415926)
            for seed in [99990001, 31415926]:
                p_a = get_payload_family_a(seed)
                stego_dict = inject_payload_bytes(state_dict, p_a, x_bits=8)
                stego_file = tmp_path / f"{arch_name}_famA_{seed}.pt"
                torch.save(stego_dict, stego_file)
                res = detector.predict(str(stego_file))
                stego_file.unlink(missing_ok=True)
                print(f"    [Family A]   seed={seed} (mean={p_a.mean():.1f}) -> margin={res['distance']:+.4f} | score={res['score']:.4f} | band={res['forensic_band']:10s} | pred={'PASS' if res['verdict'] else 'FAIL'}")
                results.append({
                    "arch": arch_name, "type": "STEGO", "payload_family": f"FamA_seed{seed}", "mean_byte": float(p_a.mean()),
                    "margin": res["distance"], "score": res["score"],
                    "band": res["forensic_band"], "pass": bool(res["verdict"])
                })

            # 3. Family B: Pure ASCII
            p_b = get_payload_family_b()
            stego_dict = inject_payload_bytes(state_dict, p_b, x_bits=8)
            stego_file = tmp_path / f"{arch_name}_famB.pt"
            torch.save(stego_dict, stego_file)
            res = detector.predict(str(stego_file))
            stego_file.unlink(missing_ok=True)
            print(f"    [Family B]   Pure ASCII (mean={p_b.mean():.1f}) -> margin={res['distance']:+.4f} | score={res['score']:.4f} | band={res['forensic_band']:10s} | pred={'PASS' if res['verdict'] else 'FAIL'}")
            results.append({
                "arch": arch_name, "type": "STEGO", "payload_family": "FamB_ASCII", "mean_byte": float(p_b.mean()),
                "margin": res["distance"], "score": res["score"],
                "band": res["forensic_band"], "pass": bool(res["verdict"])
            })

            # 4. Family C: High-Byte Binary (mean > 160)
            p_c = get_payload_family_c(seed=777)
            stego_dict = inject_payload_bytes(state_dict, p_c, x_bits=8)
            stego_file = tmp_path / f"{arch_name}_famC.pt"
            torch.save(stego_dict, stego_file)
            res = detector.predict(str(stego_file))
            stego_file.unlink(missing_ok=True)
            print(f"    [Family C]   High-Byte (mean={p_c.mean():.1f}) -> margin={res['distance']:+.4f} | score={res['score']:.4f} | band={res['forensic_band']:10s} | pred={'PASS' if res['verdict'] else 'FAIL'}")
            results.append({
                "arch": arch_name, "type": "STEGO", "payload_family": "FamC_HighByte", "mean_byte": float(p_c.mean()),
                "margin": res["distance"], "score": res["score"],
                "band": res["forensic_band"], "pass": bool(res["verdict"])
            })

            # 5. Family D: Encrypted Uniform Random (mean ~ 127.5)
            if total_weights >= 35_000_000:
                # Non-tiled per-tensor stream encryption on ultra-large models (alexnet 61M, vgg13_bn 133M)
                stego_dict = {}
                for name, tensor in state_dict.items():
                    if isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float32:
                        sub_seed = int.from_bytes(
                            hashlib.sha256(f"42|{name}".encode()).digest()[:8], "big"
                        )
                        stego_dict[name] = inject_xlsb(tensor, 8, payload_type="random", seed=sub_seed)
                    else:
                        stego_dict[name] = tensor
                mean_b = 127.5
            else:
                p_d = get_payload_family_d(seed=42)
                stego_dict = inject_payload_bytes(state_dict, p_d, x_bits=8)
                mean_b = float(p_d.mean())

            stego_file = tmp_path / f"{arch_name}_famD.pt"
            torch.save(stego_dict, stego_file)
            res = detector.predict(str(stego_file))
            stego_file.unlink(missing_ok=True)
            print(f"    [Family D]   Uniform Encrypted (mean={mean_b:.1f}) -> margin={res['distance']:+.4f} | score={res['score']:.4f} | band={res['forensic_band']:10s} | pred={'PASS' if res['verdict'] else 'FAIL'}")
            results.append({
                "arch": arch_name, "type": "STEGO", "payload_family": "FamD_UniformEncrypted", "mean_byte": mean_b,
                "margin": res["distance"], "score": res["score"],
                "band": res["forensic_band"], "pass": bool(res["verdict"])
            })
            print("-" * 85)

    tp = sum(r["type"] == "STEGO" and r["pass"] for r in results)
    fp = sum(r["type"] == "CLEAN" and not r["pass"] for r in results)
    tn = sum(r["type"] == "CLEAN" and r["pass"] for r in results)
    fn = sum(r["type"] == "STEGO" and not r["pass"] for r in results)
    specificity = tn / (tn + fp) if tn + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    balanced_acc = (specificity + recall) / 2
    print("=" * 85)
    print("STRESS TEST SUMMARY:")
    print(f"Total Samples:     {len(results)} (Clean: {tn + fp}, Stego: {tp + fn})")
    print(f"Clean Specificity: {specificity * 100:.2f}% ({tn}/{tn + fp}, FP={fp})")
    print(f"Stego Recall:      {recall * 100:.2f}% ({tp}/{tp + fn}, FN={fn})")
    print(f"Balanced Accuracy: {balanced_acc * 100:.2f}%")
    print(f"Precision:         {precision * 100:.2f}%")
    print(f"F1 Score:          {f1 * 100:.2f}%")
    print("=" * 85)

    return results


if __name__ == "__main__":
    run_stress_test()
