"""
fixtures/generate_patch_dataset.py
====================================
Dataset generation and LSB injection pipeline for Layer 3a steganalysis.

Key functions:
  - inject_xlsb: Dense X-LSB-Attack payload replacement across every weight's mantissa.
  - tensor_to_gf_fourpart_patch: Canonical four-byte float32 GF representation.
  - build_dataset: Builds train/val sets with tensor-level holdout.
  - build_dataset_leave_one_arch_out: Leaves all tensors of one architecture for val.
  - build_reference_support_set: Deterministic reference support set builder.
"""

import math
import random
import json
from pathlib import Path
import numpy as np
import torch
import torchvision.models as models

SENTINEL_BENIGN_PAYLOAD = b"SENTINEL_EICAR_BENIGN_MARKER_PAYLOAD_LSB_STEGO_MIRAGE_V1"
GF_PATCH_SIDE = 224
GF_QUADRANT_SIDE = GF_PATCH_SIDE // 2
GF_WEIGHTS_PER_PATCH = GF_QUADRANT_SIDE * GF_QUADRANT_SIDE
GF_REPRESENTATION_VERSION = "gf-fourpart-v2"
GF_IMAGE_SIZE = 100  # canonical OSL CNN input size (paper Section 4.3)


def model_to_gf_image(
    state_dict: dict,
    imsize: int = GF_IMAGE_SIZE,
) -> np.ndarray:
    """Convert model state_dict float32 weights into a 2x2 GF-Fourpart byte-plane image.

    Uses unified PyTorch bilinear interpolation with antialias=True across ALL model
    sizes to eliminate the 10,000-pixel skimage vs. torch downsampling discontinuity.

    Args:
        state_dict: PyTorch state dictionary (str -> Tensor).
        imsize:     Output spatial resolution (default: GF_IMAGE_SIZE).

    Returns:
        uint8 numpy array of shape (imsize, imsize).
    """
    parts = []
    for tensor in state_dict.values():
        if isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float32:
            parts.append(tensor.detach().cpu().numpy().astype("<f4").flatten())
    if not parts:
        raise ValueError("state_dict contains no float32 tensors")
    flat = np.concatenate(parts)

    n_weights = len(flat)
    closest_sq = int(np.ceil(np.sqrt(n_weights)) ** 2)
    sqrt_sq = int(np.sqrt(closest_sq))

    padded = np.pad(flat, (0, closest_sq - n_weights), mode="constant")
    byte_planes = padded.view(np.uint8).reshape(closest_sq, 4)[:, ::-1]  # (N_sq, 4) MSB->LSB

    tiles = [byte_planes[:, ch].reshape(sqrt_sq, sqrt_sq) for ch in range(4)]
    image_arr = np.block([[tiles[0], tiles[1]], [tiles[2], tiles[3]]])

    t = torch.from_numpy(image_arr).unsqueeze(0).unsqueeze(0).float()
    resized_t = torch.nn.functional.interpolate(
        t, size=(imsize, imsize), mode="bilinear", align_corners=False, antialias=True
    )
    return resized_t.squeeze().round().clamp(0, 255).byte().numpy()


def _build_xray_fill_payload(payload_seed: int, payload_len_bytes: int = 16384) -> np.ndarray:
    """Build a deterministic, seed-specific 16 KB binary payload buffer for
    Model X-Ray Algorithm 2 (X-LSB-Attack-Fill).

    - Even seeds (payload_seed % 2 == 0): Structured executable/sentinel payload
      (tests non-uniform byte distribution + 2D periodicity).
    - Odd seeds (payload_seed % 2 == 1): Encrypted/compressed uniform byte stream in [0, 255]
      with mean ~127.5 (forces the network to learn 2D periodic covariance and
      row-variance collapse when first-order mean shift is zero).
    """
    rng = np.random.default_rng(int(payload_seed) & 0xFFFFFFFFFFFFFFFF)
    sentinel = np.frombuffer(SENTINEL_BENIGN_PAYLOAD, dtype=np.uint8)

    if int(payload_seed) % 2 == 1:
        # High-entropy encrypted/compressed payload (Uniform [0, 255], mean ~127.5)
        buf = rng.integers(0, 256, size=payload_len_bytes, dtype=np.uint8)
        buf[: len(sentinel)] = sentinel
        return buf

    # Structured executable/rodata payload
    n_hdr = 1024
    hdr_reps = (n_hdr // len(sentinel)) + 1
    hdr = np.tile(sentinel, hdr_reps)[:n_hdr].copy()
    seed_tag = rng.integers(32, 126, size=64, dtype=np.uint8)
    hdr[64::128] = seed_tag[: len(hdr[64::128])]

    n_body = payload_len_bytes - n_hdr
    n_blocks = n_body // 16
    palette = np.concatenate([
        np.arange(0, 160, dtype=np.uint8),
        np.arange(32, 128, dtype=np.uint8),
        np.arange(192, 240, dtype=np.uint8),
    ])
    blocks = rng.choice(palette, size=(n_blocks, 16), replace=True)
    blocks[:, 0] = rng.choice(np.array([0x48, 0x55, 0x89, 0x8B, 0xE8, 0xFF, 0x0F, 0x31], dtype=np.uint8), size=n_blocks)
    blocks[:, 4] = blocks[:, 0] ^ np.uint8((payload_seed & 0x3F) + 0x11)
    blocks[:, 8] = (blocks[:, 2] >> 1) + np.uint8(0x20)
    blocks[:, 12] = np.uint8(0x00)

    body = blocks.reshape(-1)
    return np.concatenate([hdr, body]).astype(np.uint8)


def inject_model_xlsb(
    state_dict: dict,
    x_bits: int,
    payload_seed: int,
) -> dict:
    """Apply Model X-Ray Algorithm 2 (X-LSB-Attack-Fill) across all float32 parameters.

    Constructs a deterministic 16 KB seed-keyed payload m(payload_seed) and tiles it
    continuously across every float32 weight's bottom x_bits in state_dict.
    """
    payload_buf = _build_xray_fill_payload(payload_seed, payload_len_bytes=16384)
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




def inject_xlsb(
    tensor: torch.Tensor,
    x_bits: int,
    payload_type: str = "ascii",
    seed: int = 42,
) -> torch.Tensor:
    """
    Dense X-LSB-Attack: replaces the bottom x_bits of EVERY float32 weight's mantissa.
    Embedding rate ER = x_bits / 32.

    Args:
        tensor: float32 PyTorch tensor
        x_bits: number of LSB bits to overwrite per weight (e.g. 2, 4, 6, 8, 16)
        payload_type: 'ascii' (tiled low-entropy string) or 'random' (high-entropy random bits)
        seed: random seed for 'random' payload generation

    Returns:
        New float32 PyTorch tensor with dense LSB substitution.
    """
    assert tensor.dtype == torch.float32, "tensor must be float32"
    num_weights = tensor.numel()
    total_payload_bits = num_weights * x_bits
    total_bytes = math.ceil(total_payload_bits / 8)

    if payload_type == "ascii":
        payload_bytes = (SENTINEL_BENIGN_PAYLOAD * (total_bytes // len(SENTINEL_BENIGN_PAYLOAD) + 1))[:total_bytes]
        bit_array = np.unpackbits(np.frombuffer(payload_bytes, dtype=np.uint8))[:total_payload_bits]
    elif payload_type == "random":
        rng = np.random.default_rng(seed)
        random_bytes = rng.integers(0, 256, size=total_bytes, dtype=np.uint8)
        bit_array = np.unpackbits(random_bytes)[:total_payload_bits]
    else:
        raise ValueError(f"Unknown payload_type: {payload_type}")

    # Reshape bit_array to [num_weights, x_bits]
    bits_per_weight = bit_array[: num_weights * x_bits].reshape(num_weights, x_bits)

    # Convert x_bits to uint32 integer values per weight
    powers = (1 << np.arange(x_bits - 1, -1, -1, dtype=np.uint32))
    payload_uint32 = (bits_per_weight * powers).sum(axis=1).astype(np.uint32)

    # View float32 tensor as uint32
    weights_u32 = tensor.detach().cpu().numpy().astype(np.float32).flatten().view(np.uint32).copy()

    # Clear lower x_bits and OR payload bits
    mask = np.uint32(((1 << x_bits) - 1) ^ 0xFFFFFFFF)
    stego_u32 = (weights_u32 & mask) | payload_uint32

    # View back as float32 and reshape to original tensor shape
    stego_f32 = stego_u32.view(np.float32).reshape(tensor.shape)
    return torch.from_numpy(stego_f32)


def tensor_to_gf_patch(tensor: torch.Tensor, offset_idx: int = 0) -> torch.Tensor:
    """Compatibility alias for the canonical GF-Fourpart representation.

    Older experimental callers used this name for a byte-0-only image. Keeping the
    function name prevents silent import failures, but it now delegates to the
    locked four-byte representation so no active path can recreate the old feature.
    """
    return tensor_to_gf_fourpart_patch(tensor, offset_idx)


def tensor_to_gf_fourpart_patch(tensor: torch.Tensor, offset_idx: int = 0) -> torch.Tensor:
    """Convert a deterministic float32 window into a 224×224 GF-Fourpart patch.

    A float32 value supplies four byte planes.  Each plane becomes a 112×112
    grayscale quadrant; quadrants are arranged most-significant byte to
    least-significant byte in reading order.  The final 224×224 image therefore
    represents 12,544 consecutive float32 values while preserving byte position.

    This is deliberately not the legacy byte-0-only representation.  Explicit
    little-endian conversion makes the layout stable across hosts, and zero padding
    is deterministic for a short final window.
    """
    if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32:
        raise TypeError("GF-Fourpart requires a float32 torch.Tensor")
    if offset_idx < 0:
        raise ValueError("offset_idx must be non-negative")

    flat_f32 = np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype="<f4").reshape(-1)
    start = offset_idx * GF_WEIGHTS_PER_PATCH
    if start >= len(flat_f32):
        start = 0
    chunk = flat_f32[start : start + GF_WEIGHTS_PER_PATCH]
    if len(chunk) < GF_WEIGHTS_PER_PATCH:
        chunk = np.pad(chunk, (0, GF_WEIGHTS_PER_PATCH - len(chunk)), mode="constant")

    # Little-endian storage exposes bytes [LSB, ..., MSB]; reverse into the
    # documented GF order so sign/exponent information occupies the first quadrant.
    byte_planes = chunk.view(np.uint8).reshape(GF_WEIGHTS_PER_PATCH, 4)[:, ::-1]
    quadrants = [plane.reshape(GF_QUADRANT_SIDE, GF_QUADRANT_SIDE) for plane in byte_planes.T]
    image = np.block([[quadrants[0], quadrants[1]], [quadrants[2], quadrants[3]]])
    return torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32) / 255.0).unsqueeze(0)


def qualifying_tensors(model_dict: dict) -> list:
    """Return convolutional weight tensors large enough for a full byte-plane patch.

    PyTorch convolution weights are four-dimensional ``[out, in, kH, kW]``.
    Restricting to them prevents fully-connected matrices and normalization state
    from silently entering the Layer 3a corpus.
    """
    qualifying = []
    for name, param in model_dict.items():
        if isinstance(param, torch.Tensor) and param.dtype == torch.float32:
            if param.dim() == 4 and param.numel() >= GF_WEIGHTS_PER_PATCH:
                qualifying.append((name, param))
    return qualifying


def get_default_architecture_models():
    """Instantiates clean torchvision architecture instances with weights=None."""
    arch_models = {
        "resnet18": models.resnet18(weights=None),
        "resnet50": models.resnet50(weights=None),
        "densenet121": models.densenet121(weights=None),
        "mobilenet_v2": models.mobilenet_v2(weights=None),
    }
    return arch_models


def build_dataset(
    x_bits: int,
    val_frac: float = 0.3,
    offsets_per_tensor: int = 2,
    archs: list | None = None,
    seed: int = 42,
):
    """
    Builds clean/stego patch items with a tensor-level train/val split (prevents leakage).
    """
    if archs is None:
        archs = ["resnet18", "resnet50", "densenet121", "mobilenet_v2"]

    arch_models = get_default_architecture_models()
    rng = random.Random(seed)

    train_items = []
    val_items = []

    for arch_name in archs:
        model = arch_models[arch_name]
        qual = qualifying_tensors(model.state_dict())

        # Shuffle qualifying tensors deterministically per arch
        indices = list(range(len(qual)))
        rng.shuffle(indices)

        n_val = max(1, int(len(qual) * val_frac))
        val_indices = set(indices[:n_val])

        for idx, (t_name, tensor) in enumerate(qual):
            stego_tensor = inject_xlsb(tensor, x_bits=x_bits, payload_type="ascii", seed=seed)

            target_list = val_items if idx in val_indices else train_items

            max_offsets = tensor.numel() // (224 * 224)
            actual_offsets = min(offsets_per_tensor, max_offsets)

            for offset_idx in range(actual_offsets):
                clean_patch = tensor_to_gf_patch(tensor, offset_idx=offset_idx)
                stego_patch = tensor_to_gf_patch(stego_tensor, offset_idx=offset_idx)

                target_list.append({
                    "patch": clean_patch,
                    "label": 0,
                    "arch": arch_name,
                    "tensor_name": t_name,
                    "offset": offset_idx,
                })
                target_list.append({
                    "patch": stego_patch,
                    "label": 1,
                    "arch": arch_name,
                    "tensor_name": t_name,
                    "offset": offset_idx,
                })

    return train_items, val_items


def build_dataset_leave_one_arch_out(
    x_bits: int,
    held_out_arch: str,
    offsets_per_tensor: int = 2,
    seed: int = 42,
):
    """
    Architecture-level train/val split: all tensors from held_out_arch go to val.
    """
    all_archs = ["resnet18", "resnet50", "densenet121", "mobilenet_v2"]

    arch_models = get_default_architecture_models()
    train_items = []
    val_items = []

    for arch_name in all_archs:
        model = arch_models[arch_name]
        qual = qualifying_tensors(model.state_dict())
        target_list = val_items if arch_name == held_out_arch else train_items

        for t_name, tensor in qual:
            stego_tensor = inject_xlsb(tensor, x_bits=x_bits, payload_type="ascii", seed=seed)
            max_offsets = tensor.numel() // (224 * 224)
            actual_offsets = min(offsets_per_tensor, max_offsets)

            for offset_idx in range(actual_offsets):
                clean_patch = tensor_to_gf_patch(tensor, offset_idx=offset_idx)
                stego_patch = tensor_to_gf_patch(stego_tensor, offset_idx=offset_idx)

                target_list.append({
                    "patch": clean_patch,
                    "label": 0,
                    "arch": arch_name,
                    "tensor_name": t_name,
                    "offset": offset_idx,
                })
                target_list.append({
                    "patch": stego_patch,
                    "label": 1,
                    "arch": arch_name,
                    "tensor_name": t_name,
                    "offset": offset_idx,
                })

    return train_items, val_items


def build_reference_support_set(x_bits: int = 8, seed: int = 42, n_per_class: int = 6):
    """
    Builds a deterministic reference support set drawn only from the train pool.
    """
    train_items, _ = build_dataset(x_bits=x_bits, val_frac=0.3, offsets_per_tensor=2, seed=seed)

    clean_pool = [item for item in train_items if item["label"] == 0]
    stego_pool = [item for item in train_items if item["label"] == 1]

    rng = random.Random(seed)
    rng.shuffle(clean_pool)
    rng.shuffle(stego_pool)

    ref_clean = clean_pool[:n_per_class]
    ref_stego = stego_pool[:n_per_class]

    reference_set = {
        "clean_patches": torch.stack([item["patch"] for item in ref_clean]),  # [N, 1, 224, 224]
        "stego_patches": torch.stack([item["patch"] for item in ref_stego]),  # [N, 1, 224, 224]
        "labels": torch.tensor([0] * len(ref_clean) + [1] * len(ref_stego), dtype=torch.long),
        "seed": seed,
        "x_bits": x_bits,
    }
    return reference_set


def build_dataset_from_corpus(
    manifest_path: str,
    x_bits: int = 8,
    offsets_per_tensor: int = 2,
    holdout_base_id: str | None = None,
):
    """Load real grouped corpus artifacts and make patch items for Layer 3a.

    ``holdout_base_id`` is an explicit exploratory holdout for the current two-source
    demo. It keeps every clean/stego derivative of that base checkpoint together.
    The production benchmark should use manifest-level train/reference/dev/test splits.
    """
    manifest_file = Path(manifest_path).resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    project_root = manifest_file.parents[2] if len(manifest_file.parents) > 2 else manifest_file.parent
    train_items, holdout_items = [], []
    for record in manifest:
        attack = record.get("attack", {})
        if record.get("label") == "stego" and int(attack.get("x_bits", -1)) != x_bits:
            continue
        artifact_path = Path(record["filepath"])
        if not artifact_path.is_absolute():
            artifact_path = project_root / artifact_path
        state = torch.load(artifact_path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise ValueError(f"corpus artifact is not a state_dict: {record['filepath']}")
        target = holdout_items if holdout_base_id and record["base_checkpoint_id"] == holdout_base_id else train_items
        for tensor_name, tensor in qualifying_tensors(state):
            stego_flag = record["label"] == "stego"
            actual_offsets = min(offsets_per_tensor, max(1, tensor.numel() // (224 * 224)))
            for offset_idx in range(actual_offsets):
                target.append({
                    "patch": tensor_to_gf_fourpart_patch(tensor, offset_idx=offset_idx),
                    "label": int(stego_flag),
                    "arch": record["architecture_family"],
                    "tensor_name": tensor_name,
                    "offset": offset_idx,
                    "base_checkpoint_id": record["base_checkpoint_id"],
                    "artifact_id": record["artifact_id"],
                    "representation": GF_REPRESENTATION_VERSION,
                })
    if holdout_base_id and not holdout_items:
        raise ValueError(f"holdout base checkpoint not found in corpus: {holdout_base_id}")
    return train_items, holdout_items
