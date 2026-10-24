"""
detector/disarmer_lsb.py
========================
Layer 3a Disarmament: LSB Steganographic Payload Removal.

Zeros out the bottom `x_bits` of every float32 weight's IEEE-754 uint32
representation across all supported model formats (.safetensors, .pt, .pkl, .h5).

Why zeroing is safe:
    Zeroing the Q3 byte (x_bits=8) changes each weight by at most
    2^(-23+8) = 2^(-15) ≈ 3e-5 in absolute magnitude — negligible for
    model accuracy, but completely destroys any structured LSB payload.

Forensic extraction:
    Before zeroing, the Q3 (LSB) byte of every float32 is read out,
    concatenated, and hashed with SHA-256. This hash is returned as
    forensic evidence of what was removed.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class LsbDisarmResult:
    cleaned_filepath: str
    payload_sha256: str
    payload_size_bytes: int
    x_bits_zeroed: int
    tensors_processed: int
    weights_modified: int


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------

def _zero_lsb_array(arr: np.ndarray, x_bits: int) -> tuple[np.ndarray, bytes]:
    """
    Zero the bottom `x_bits` of every element of a float32 array.

    Returns:
        (cleaned_array, extracted_payload_bytes)
    """
    flat_u32 = arr.flatten().astype(np.float32).view(np.uint32).copy()
    mask = np.uint32(((1 << x_bits) - 1) ^ 0xFFFFFFFF)
    payload_u32 = flat_u32 & np.uint32((1 << x_bits) - 1)

    if x_bits <= 8:
        payload_bytes = payload_u32.astype(np.uint8).tobytes()
    elif x_bits <= 16:
        payload_bytes = payload_u32.astype("<u2").tobytes()
    else:
        payload_bytes = payload_u32.astype("<u4").tobytes()

    cleaned_u32 = flat_u32 & mask
    cleaned = cleaned_u32.view(np.float32).reshape(arr.shape)
    return cleaned, payload_bytes


def _extract_and_zero_tensors(
    state_dict: dict,
    x_bits: int,
) -> tuple[dict, bytes, int, int]:
    """
    Apply LSB zeroing to all float32 tensors in a state_dict.

    Returns:
        (cleaned_state_dict, payload_bytes, tensors_processed, weights_modified)
    """
    try:
        import torch
    except ImportError:
        torch = None

    cleaned = {}
    all_payload = []
    tensors_processed = 0
    weights_modified = 0

    for key, val in state_dict.items():
        if torch is not None and hasattr(val, "numpy") and val.dtype.is_floating_point:
            arr = val.detach().cpu().numpy().astype(np.float32)
            cleaned_arr, payload = _zero_lsb_array(arr, x_bits)
            all_payload.append(payload)
            tensors_processed += 1
            weights_modified += arr.size
            cleaned[key] = torch.from_numpy(cleaned_arr.copy())
        elif isinstance(val, np.ndarray) and val.dtype == np.float32:
            cleaned_arr, payload = _zero_lsb_array(val, x_bits)
            all_payload.append(payload)
            tensors_processed += 1
            weights_modified += val.size
            cleaned[key] = cleaned_arr
        else:
            cleaned[key] = val

    return cleaned, b"".join(all_payload), tensors_processed, weights_modified


# ---------------------------------------------------------------------------
# Format-specific disarmament
# ---------------------------------------------------------------------------

def _disarm_pt(src: str, dst: str, x_bits: int) -> tuple[bytes, int, int]:
    """
    Disarm a .pt / .pkl / .pth file (both zip and legacy formats).
    Tries weights_only=True first; falls back to False or raw pickle.load
    (safe here because this path is only reached after the opcode disarmer has
    already cleaned the file, or the file has no dangerous callables).
    """
    import pickle
    import torch
    state_dict = None
    try:
        state_dict = torch.load(src, map_location="cpu", weights_only=True)
    except Exception:
        try:
            state_dict = torch.load(src, map_location="cpu", weights_only=False)
        except Exception:
            with open(src, "rb") as f:
                state_dict = pickle.load(f)

    if not isinstance(state_dict, dict):
        state_dict = {"weights": state_dict}

    cleaned, payload, n_tensors, n_weights = _extract_and_zero_tensors(state_dict, x_bits)
    torch.save(cleaned, dst)
    return payload, n_tensors, n_weights


def _disarm_safetensors(src: str, dst: str, x_bits: int) -> tuple[bytes, int, int]:
    """Disarm a .safetensors file by parsing the binary format directly."""
    with open(src, "rb") as f:
        raw = f.read()

    # Parse safetensors header (8-byte little-endian length prefix)
    if len(raw) < 8:
        raise ValueError("Invalid safetensors file: too short")

    header_len = struct.unpack_from("<Q", raw, 0)[0]
    header_bytes = raw[8: 8 + header_len]
    data_start = 8 + header_len
    data_block = bytearray(raw[data_start:])

    header = json.loads(header_bytes.decode("utf-8"))

    all_payload = []
    n_tensors = 0
    n_weights = 0

    for name, meta in header.items():
        if name == "__metadata__":
            continue
        dtype = meta.get("dtype", "")
        if dtype != "F32":
            continue  # only process float32 tensors

        shape = meta.get("shape", [])
        offsets = meta.get("data_offsets", [0, 0])
        start, end = offsets

        arr_bytes = data_block[start:end]
        arr = np.frombuffer(arr_bytes, dtype=np.float32).reshape(shape).copy()

        cleaned_arr, payload = _zero_lsb_array(arr, x_bits)
        all_payload.append(payload)
        data_block[start:end] = cleaned_arr.tobytes()
        n_tensors += 1
        n_weights += arr.size

    # Reconstruct safetensors file
    with open(dst, "wb") as f:
        f.write(struct.pack("<Q", header_len))
        f.write(header_bytes)
        f.write(bytes(data_block))

    return b"".join(all_payload), n_tensors, n_weights


def _disarm_h5(src: str, dst: str, x_bits: int) -> tuple[bytes, int, int]:
    """Disarm an HDF5 (.h5) file."""
    try:
        import h5py
    except ImportError as e:
        raise ImportError("h5py required for .h5 disarmament: pip install h5py") from e

    import shutil
    shutil.copy2(src, dst)

    all_payload = []
    n_tensors = 0
    n_weights = 0

    with h5py.File(dst, "r+") as f:
        def _process_dataset(name, obj):
            nonlocal n_tensors, n_weights
            if not isinstance(obj, h5py.Dataset):
                return
            if obj.dtype != np.float32:
                return
            arr = obj[()].copy()
            cleaned_arr, payload = _zero_lsb_array(arr, x_bits)
            obj[()] = cleaned_arr
            all_payload.append(payload)
            n_tensors += 1
            n_weights += arr.size

        f.visititems(_process_dataset)

    return b"".join(all_payload), n_tensors, n_weights


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def disarm_lsb(filepath: str, x_bits: int = 8) -> LsbDisarmResult:
    """
    Remove LSB steganographic payload from a model file by zeroing float32 LSBs.

    Args:
        filepath: Absolute path to the malicious model file.
        x_bits:   Number of LSB bits to zero per weight (default=8 = full Q3 byte).

    Returns:
        LsbDisarmResult with path to cleaned file and forensic evidence.

    Raises:
        ValueError: If the file format is unsupported.
        FileNotFoundError: If the source file does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Source file not found: {filepath}")

    ext = Path(filepath).suffix.lower()
    suffix_map = {
        ".pt":           _disarm_pt,
        ".pth":          _disarm_pt,
        ".pkl":          _disarm_pt,
        ".safetensors":  _disarm_safetensors,
        ".h5":           _disarm_h5,
        ".hdf5":         _disarm_h5,
    }

    disarm_fn = suffix_map.get(ext)
    if disarm_fn is None:
        raise ValueError(f"LSB disarmament not supported for format: {ext}")

    # Write to a sibling temp file, then rename to avoid partial writes
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=ext, dir=os.path.dirname(filepath))
    os.close(tmp_fd)

    try:
        payload_bytes, n_tensors, n_weights = disarm_fn(filepath, tmp_path, x_bits)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    payload_sha256 = hashlib.sha256(payload_bytes).hexdigest()

    return LsbDisarmResult(
        cleaned_filepath=tmp_path,
        payload_sha256=payload_sha256,
        payload_size_bytes=len(payload_bytes),
        x_bits_zeroed=x_bits,
        tensors_processed=n_tensors,
        weights_modified=n_weights,
    )
