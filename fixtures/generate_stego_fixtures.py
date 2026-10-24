"""Generate LSB steganography and weight-distribution backdoor test fixtures.

Produces synthetic test checkpoints with known embedding rates (6-100%)
for offline validation and unit tests.
"""

import hashlib
import json
import math
import os

import numpy as np

try:
    import torch
    import torchvision.models as models
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[WARNING] PyTorch not found — .pt fixture generation will be skipped.")

try:
    from safetensors.torch import save_file as st_save_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False
    print("[WARNING] safetensors package not found — .safetensors fixture generation will be skipped.")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIXTURES_DIR = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(FIXTURES_DIR, "manifest.json")
PROJECT_ROOT = os.path.dirname(FIXTURES_DIR)

# EICAR-style benign sentinel (won't trigger AV; clearly marks the payload)
SENTINEL_PAYLOAD = b"SENTINEL_EICAR_BENIGN_MARKER_PAYLOAD_LSB_STEGO_MIRAGE_V1"

# Embedding rates (fraction of total weight LSB capacity to fill)
EMBEDDING_RATES = [0.06, 0.15, 0.25, 0.50, 1.00]


# ---------------------------------------------------------------------------
# Core LSB embed / extract functions (Section 2 of analysis.md spec)
# ---------------------------------------------------------------------------

def x_lsb_fill_attack(
    weight_tensors: list,
    payload_bytes: bytes,
    x_bits: int = 1,
    embedding_rate: float = 1.0,
) -> list:
    """
    Embed payload_bytes into the x least-significant bits of weight tensor floats.

    The IEEE-754 float32 LSBs are manipulated at the bit level in the uint32
    representation of each weight value, so the weight magnitudes change by < 2^(-23+x),
    i.e., imperceptible to the model's accuracy.

    Args:
        weight_tensors : list of torch.Tensor (from model.state_dict().values())
        payload_bytes  : bytes to embed (tiled to fill requested capacity)
        x_bits         : number of LSBs per float to use (1 or 2 recommended)
        embedding_rate : fraction of the total LSB capacity to fill (0.0 to 1.0)

    Returns:
        list of torch.Tensor with the same shapes, LSBs modified.
    """
    # Flatten all weights into uint32 views of their raw float32 bits
    flat_arrays = [
        w.cpu().numpy().astype(np.float32).flatten().view(np.uint32).copy()
        for w in weight_tensors
    ]
    total_weights = sum(a.size for a in flat_arrays)

    # Total bits we can store and how many we actually fill
    total_capacity_bits = total_weights * x_bits
    bits_to_write = int(math.floor(total_capacity_bits * embedding_rate))
    bytes_to_write = bits_to_write // 8

    if bytes_to_write == 0:
        return [
            torch.from_numpy(a.view(np.float32).reshape(w.shape))
            for a, w in zip(flat_arrays, weight_tensors)
        ]

    # Tile the sentinel payload to fill the required byte count
    tiled_payload = (payload_bytes * (bytes_to_write // len(payload_bytes) + 1))[:bytes_to_write]

    # Convert payload to a flat bit array
    payload_bits = np.unpackbits(np.frombuffer(tiled_payload, dtype=np.uint8))[:bits_to_write]

    # Build a global flat uint32 array across all tensors
    global_flat = np.concatenate(flat_arrays)

    # LSB mask: clear the bottom x_bits of each weight
    # Use explicit 32-bit masking to avoid Python int overflow on np.uint32(~...)
    lsb_mask = np.uint32(((1 << x_bits) - 1) ^ 0xFFFFFFFF)

    bit_idx = 0
    weight_idx = 0
    while bit_idx < bits_to_write and weight_idx < total_weights:
        bits_for_this_weight = payload_bits[bit_idx : bit_idx + x_bits]
        if len(bits_for_this_weight) == 0:
            break
        if len(bits_for_this_weight) < x_bits:
            bits_for_this_weight = np.pad(bits_for_this_weight, (0, x_bits - len(bits_for_this_weight)))

        packed = np.uint32(0)
        for b in bits_for_this_weight:
            packed = (packed << np.uint32(1)) | np.uint32(int(b))

        global_flat[weight_idx] = (global_flat[weight_idx] & lsb_mask) | packed
        bit_idx += x_bits
        weight_idx += 1

    # Split back into per-tensor arrays and convert back to torch tensors
    result_tensors = []
    offset = 0
    for orig_tensor, orig_array in zip(weight_tensors, flat_arrays):
        size = orig_array.size
        chunk = global_flat[offset : offset + size]
        result_tensors.append(
            torch.from_numpy(chunk.view(np.float32).reshape(orig_tensor.shape))
        )
        offset += size

    return result_tensors


def x_lsb_extract(
    weight_tensors: list,
    x_bits: int = 1,
    embedding_rate: float = 1.0,
) -> bytes:
    """
    Inverse of x_lsb_fill_attack — extract payload bytes from LSBs of weights.

    Used during verified disarmament (Phase 5).
    """
    flat_arrays = [
        w.cpu().numpy().astype(np.float32).flatten().view(np.uint32)
        for w in weight_tensors
    ]
    total_weights = sum(a.size for a in flat_arrays)

    total_capacity_bits = total_weights * x_bits
    bits_to_read = int(math.floor(total_capacity_bits * embedding_rate))
    bytes_to_read = bits_to_read // 8

    lsb_mask = np.uint32((1 << x_bits) - 1)
    global_flat = np.concatenate(flat_arrays)

    extracted_bits = []
    for weight_idx in range(min(bits_to_read // x_bits + 1, total_weights)):
        lsb_val = int(global_flat[weight_idx] & lsb_mask)
        for bit_pos in range(x_bits - 1, -1, -1):
            extracted_bits.append((lsb_val >> bit_pos) & 1)
        if len(extracted_bits) >= bits_to_read:
            break

    extracted_bits = extracted_bits[:bits_to_read]
    bit_array = np.array(extracted_bits, dtype=np.uint8)

    pad_size = (8 - len(bit_array) % 8) % 8
    if pad_size:
        bit_array = np.pad(bit_array, (0, pad_size))

    return np.packbits(bit_array).tobytes()[:bytes_to_read]


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Fixture: LSB steganography (.pt format)
# ---------------------------------------------------------------------------

def generate_pt_stego_fixtures(base_model_path: str, new_entries: list):
    """Load base benign_resnet18_real.pt and generate one stego .pt per embedding rate."""
    if not HAS_TORCH:
        print("[SKIP] PyTorch unavailable — skipping .pt stego fixtures.")
        return

    print(f"\n[PT STEGO] Loading base model from {base_model_path}")
    if not os.path.exists(base_model_path):
        print("  Base model not found. Generating resnet18 state_dict...")
        m = models.resnet18(weights=None)
        torch.save(m.state_dict(), base_model_path)

    state_dict = torch.load(base_model_path, map_location="cpu", weights_only=True)
    weight_tensors = [
        v for v in state_dict.values()
        if isinstance(v, torch.Tensor) and v.dtype == torch.float32
    ]
    print(f"  Base model has {sum(t.numel() for t in weight_tensors):,} float32 weights")

    for rate in EMBEDDING_RATES:
        rate_pct = int(rate * 100)
        out_name = f"stego_resnet18_{rate_pct}pct.pt"
        out_path = os.path.join(FIXTURES_DIR, out_name)

        stego_tensors = x_lsb_fill_attack(weight_tensors, SENTINEL_PAYLOAD, x_bits=1, embedding_rate=rate)

        stego_state_dict = {}
        stego_iter = iter(stego_tensors)
        for key, val in state_dict.items():
            if isinstance(val, torch.Tensor) and val.dtype == torch.float32:
                stego_state_dict[key] = next(stego_iter)
            else:
                stego_state_dict[key] = val

        torch.save(stego_state_dict, out_path)
        print(f"  [OK] {out_name} ({rate_pct}% embedding rate)")

        new_entries.append({
            "filepath": os.path.relpath(out_path, PROJECT_ROOT).replace("\\", "/"),
            "format": "pt",
            "ground_truth_label": "MALICIOUS",
            "attack_type": "lsb_stego",
            "embedding_rate": rate,
            "perturbation_magnitude": 0.0,
            "payload_sha256": sha256_of_bytes(SENTINEL_PAYLOAD),
            "x_bits": 1,
            "notes": f"resnet18 base, {rate_pct}% LSB capacity filled with SENTINEL_PAYLOAD",
        })


# ---------------------------------------------------------------------------
# Fixture: LSB steganography (.safetensors format)
# ---------------------------------------------------------------------------

def generate_safetensors_stego_fixtures(new_entries: list):
    """Generate a clean synthetic CNN safetensors model and create LSB stego variants."""
    if not HAS_SAFETENSORS:
        print("[SKIP] safetensors unavailable — skipping .safetensors stego fixtures.")
        return
    if not HAS_TORCH:
        print("[SKIP] PyTorch unavailable — skipping .safetensors stego fixtures.")
        return

    print("\n[SAFETENSORS STEGO] Generating clean safetensors base model...")

    rng = np.random.default_rng(seed=42)
    base_state = {
        "conv1.weight":    torch.tensor(rng.normal(0, 0.02, (64, 3, 7, 7)),   dtype=torch.float32),
        "conv1.bias":      torch.tensor(rng.normal(0, 0.01, (64,)),           dtype=torch.float32),
        "conv2.weight":    torch.tensor(rng.normal(0, 0.02, (128, 64, 3, 3)), dtype=torch.float32),
        "conv2.bias":      torch.tensor(rng.normal(0, 0.01, (128,)),          dtype=torch.float32),
        "conv3.weight":    torch.tensor(rng.normal(0, 0.02, (256, 128, 3, 3)),dtype=torch.float32),
        "conv3.bias":      torch.tensor(rng.normal(0, 0.01, (256,)),          dtype=torch.float32),
        "fc1.weight":      torch.tensor(rng.normal(0, 0.02, (512, 1024)),     dtype=torch.float32),
        "fc1.bias":        torch.tensor(rng.normal(0, 0.01, (512,)),          dtype=torch.float32),
        "fc2.weight":      torch.tensor(rng.normal(0, 0.02, (10, 512)),       dtype=torch.float32),
        "fc2.bias":        torch.tensor(rng.normal(0, 0.01, (10,)),           dtype=torch.float32),
    }

    base_st_path = os.path.join(FIXTURES_DIR, "benign_cnn_base.safetensors")
    st_save_file(base_state, base_st_path)
    print("  [OK] Clean base: benign_cnn_base.safetensors")

    new_entries.append({
        "filepath": os.path.relpath(base_st_path, PROJECT_ROOT).replace("\\", "/"),
        "format": "safetensors",
        "ground_truth_label": "BENIGN",
        "attack_type": "none",
        "embedding_rate": 0.0,
        "perturbation_magnitude": 0.0,
        "notes": "Synthetic CNN base for LSB stego — clean reference for Layer 3a centroid",
    })

    weight_tensors = [v for v in base_state.values() if v.dtype == torch.float32]
    total_weights = sum(t.numel() for t in weight_tensors)
    print(f"  Base has {total_weights:,} float32 weights")

    for rate in EMBEDDING_RATES:
        rate_pct = int(rate * 100)
        out_name = f"stego_cnn_{rate_pct}pct.safetensors"
        out_path = os.path.join(FIXTURES_DIR, out_name)

        stego_tensors = x_lsb_fill_attack(weight_tensors, SENTINEL_PAYLOAD, x_bits=1, embedding_rate=rate)

        stego_state = {}
        stego_iter = iter(stego_tensors)
        for key in base_state:
            stego_state[key] = next(stego_iter)

        st_save_file(stego_state, out_path)
        print(f"  [OK] {out_name} ({rate_pct}% embedding rate)")

        new_entries.append({
            "filepath": os.path.relpath(out_path, PROJECT_ROOT).replace("\\", "/"),
            "format": "safetensors",
            "ground_truth_label": "MALICIOUS",
            "attack_type": "lsb_stego",
            "embedding_rate": rate,
            "perturbation_magnitude": 0.0,
            "payload_sha256": sha256_of_bytes(SENTINEL_PAYLOAD),
            "x_bits": 1,
            "notes": f"Synthetic CNN base, {rate_pct}% LSB capacity filled with SENTINEL_PAYLOAD",
        })


# ---------------------------------------------------------------------------
# Fixture: Weight-distribution backdoor (.pt format, for Layer 3b)
# ---------------------------------------------------------------------------

def generate_backdoor_fixtures(base_model_path: str, new_entries: list):
    """
    Generate synthetic weight-backdoor fixtures by perturbing specific layers
    of a clean resnet18 state dict. Three perturbation types across four configs.
    """
    if not HAS_TORCH:
        print("[SKIP] PyTorch unavailable — skipping backdoor fixtures.")
        return

    print("\n[BACKDOOR] Generating weight-distribution backdoor fixtures...")

    if not os.path.exists(base_model_path):
        m = models.resnet18(weights=None)
        torch.save(m.state_dict(), base_model_path)

    state_dict = torch.load(base_model_path, map_location="cpu", weights_only=True)

    backdoor_configs = [
        {
            "name": "backdoor_mean_shift_mild",
            "perturbation_magnitude": 3.0,
            "description": "Mean-shift on conv layers by 3 std — mild, detectable at N=3 threshold",
            "type": "mean_shift",
            "target_layers": ["layer1", "layer2"],
            "shift_std_multiplier": 3.0,
        },
        {
            "name": "backdoor_mean_shift_strong",
            "perturbation_magnitude": 6.0,
            "description": "Mean-shift on multiple layers by 6 std — strong, clearly detectable",
            "type": "mean_shift",
            "target_layers": ["layer1", "layer2", "layer3"],
            "shift_std_multiplier": 6.0,
        },
        {
            "name": "backdoor_outlier_injection",
            "perturbation_magnitude": 10.0,
            "description": "Inject outlier channels (5% of weights set to 10 std away) in fc layers",
            "type": "outlier_injection",
            "target_layers": ["fc"],
            "outlier_fraction": 0.05,
            "outlier_multiplier": 10.0,
        },
        {
            "name": "backdoor_variance_inflation",
            "perturbation_magnitude": 5.0,
            "description": "Variance inflation 5x on layer4 — weight distribution blown out",
            "type": "variance_inflation",
            "target_layers": ["layer4"],
            "scale_factor": 5.0,
        },
    ]

    rng = np.random.default_rng(seed=99)

    for cfg in backdoor_configs:
        out_name = f"{cfg['name']}.pt"
        out_path = os.path.join(FIXTURES_DIR, out_name)

        perturbed_state = {}
        for key, tensor in state_dict.items():
            if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32:
                perturbed_state[key] = tensor
                continue

            should_perturb = any(target in key for target in cfg["target_layers"])
            if not should_perturb:
                perturbed_state[key] = tensor
                continue

            t_np = tensor.cpu().numpy().astype(np.float32).copy()
            layer_std = float(t_np.std()) + 1e-8
            layer_mean = float(t_np.mean())

            if cfg["type"] == "mean_shift":
                shift = cfg["shift_std_multiplier"] * layer_std
                t_np += shift

            elif cfg["type"] == "outlier_injection":
                n_outliers = max(1, int(t_np.size * cfg["outlier_fraction"]))
                outlier_indices = rng.choice(t_np.size, size=n_outliers, replace=False)
                outlier_sign = rng.choice([-1, 1], size=n_outliers)
                t_np.flat[outlier_indices] = (
                    layer_mean + outlier_sign * cfg["outlier_multiplier"] * layer_std
                )

            elif cfg["type"] == "variance_inflation":
                t_np = layer_mean + (t_np - layer_mean) * cfg["scale_factor"]

            perturbed_state[key] = torch.from_numpy(t_np.reshape(tensor.shape))

        torch.save(perturbed_state, out_path)
        print(f"  [OK] {out_name} — {cfg['description']}")

        new_entries.append({
            "filepath": os.path.relpath(out_path, PROJECT_ROOT).replace("\\", "/"),
            "format": "pt",
            "ground_truth_label": "MALICIOUS",
            "attack_type": "weight_backdoor",
            "embedding_rate": 0.0,
            "perturbation_magnitude": cfg["perturbation_magnitude"],
            "backdoor_type": cfg["type"],
            "notes": cfg["description"],
        })


# ---------------------------------------------------------------------------
# Append to manifest
# ---------------------------------------------------------------------------

def append_to_manifest(new_entries: list):
    """Safely append new entries to the existing manifest.json without duplicates."""
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r") as f:
            existing = json.load(f)
    else:
        existing = []

    existing_paths = {e["filepath"] for e in existing}
    deduped_new = [e for e in new_entries if e["filepath"] not in existing_paths]

    if not deduped_new:
        print("\n[MANIFEST] No new entries to append (all already present).")
        return

    merged = existing + deduped_new
    with open(MANIFEST_PATH, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"\n[MANIFEST] Appended {len(deduped_new)} new entries -> {MANIFEST_PATH}")
    print(f"           Total entries now: {len(merged)}")


# ---------------------------------------------------------------------------
# Verify round-trip extraction (sanity check)
# ---------------------------------------------------------------------------

def verify_extraction(base_model_path: str):
    """
    Spot-check: embed the sentinel at 100% rate into resnet18,
    then extract it back and confirm the first bytes match.
    """
    if not HAS_TORCH:
        return

    print("\n[VERIFY] Round-trip LSB embed/extract check...")
    state_dict = torch.load(base_model_path, map_location="cpu", weights_only=True)
    weight_tensors = [
        v for v in state_dict.values()
        if isinstance(v, torch.Tensor) and v.dtype == torch.float32
    ]

    stego_tensors = x_lsb_fill_attack(weight_tensors, SENTINEL_PAYLOAD, x_bits=1, embedding_rate=1.0)
    extracted = x_lsb_extract(stego_tensors, x_bits=1, embedding_rate=1.0)

    expected_len = sum(t.numel() for t in weight_tensors) // 8
    tiled = (SENTINEL_PAYLOAD * (expected_len // len(SENTINEL_PAYLOAD) + 1))[:expected_len]

    if extracted[:50] == tiled[:50]:
        print("  [OK] First 50 bytes match — embed/extract pipeline is correct.")
    else:
        print("  [FAIL] MISMATCH — check x_lsb_fill_attack / x_lsb_extract!")
        print(f"         Expected: {tiled[:50]}")
        print(f"         Got:      {extracted[:50]}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_stego_and_backdoor_fixtures():
    os.makedirs(FIXTURES_DIR, exist_ok=True)

    base_pt_path = os.path.join(FIXTURES_DIR, "benign_resnet18_real.pt")
    new_entries = []

    print("=" * 60)
    print("  MIRAGE: Stego & Backdoor Fixture Generator")
    print("=" * 60)

    generate_pt_stego_fixtures(base_pt_path, new_entries)
    generate_safetensors_stego_fixtures(new_entries)
    generate_backdoor_fixtures(base_pt_path, new_entries)
    append_to_manifest(new_entries)
    verify_extraction(base_pt_path)

    print("\n[DONE] Fixture generation complete.")


if __name__ == "__main__":
    generate_stego_and_backdoor_fixtures()
