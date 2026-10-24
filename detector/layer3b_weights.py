"""
detector/layer3b_weights.py - Lightweight Within-Model Weight Anomaly Detector
===============================================================================
Layer 3b: Reference-free geometric and spectral backdoor/trojan detector.
No reference model, no training, no test data required.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any
import numpy as np

# Calibrated physical thresholds
INFLATED_NEURON_RATIO_THRESH: float = 15.0
INFLATED_NEURON_P90_THRESH: float   = 3.0
SPECTRAL_RATIO_THRESH: float        = 10.0
EXTREME_KURTOSIS_THRESH: float      = 100.0

# High-conviction (2x) single-signal dominance thresholds
DOMINANT_NEURON_RATIO_THRESH: float = 30.0
DOMINANT_SPECTRAL_THRESH: float     = 20.0
DOMINANT_KURTOSIS_THRESH: float     = 200.0

ACTIVE_NORM_FLOOR: float            = 1e-3
MIN_LAYER_WEIGHTS: int              = 64
MIN_NEURONS: int                    = 4
MIN_SVD_DIM: int                    = 32

_SKIP_KURTOSIS_KEYS = (
    "fc",
    "classifier",
    "head",
    "downsample",
    "shortcut",
    "se.",
    "squeeze",
    "excitation",
    "proj",
)


def _excess_kurtosis(arr: np.ndarray) -> float:
    if arr.size < 4:
        return 0.0
    sigma = float(arr.std())
    if sigma < 1e-12:
        return 0.0
    z = (arr - float(arr.mean())) / sigma
    return float(np.mean(z ** 4)) - 3.0


def _spectral_ratio(mat: np.ndarray) -> float:
    """Compute sigma_1 / sigma_2 on the active (non-pruned) 2D weight submatrix.

    Requires min(M_active, N_active) >= MIN_SVD_DIM (32) so low-rank Squeeze-and-Excitation
    and channel-bottleneck projections do not trigger false spectral collapse.
    """
    if mat.ndim != 2 or min(mat.shape) < MIN_SVD_DIM:
        return 1.0

    row_norms = np.linalg.norm(mat, axis=1)
    active_rows = row_norms >= ACTIVE_NORM_FLOOR
    if int(np.sum(active_rows)) < MIN_SVD_DIM:
        return 1.0
    mat = mat[active_rows]

    col_norms = np.linalg.norm(mat, axis=0)
    active_cols = col_norms >= ACTIVE_NORM_FLOOR
    if int(np.sum(active_cols)) < MIN_SVD_DIM:
        return 1.0
    mat = mat[:, active_cols]

    if mat.shape[0] > 512 or mat.shape[1] > 512:
        rng = np.random.default_rng(0)
        r = rng.choice(mat.shape[0], min(512, mat.shape[0]), replace=False)
        c = rng.choice(mat.shape[1], min(512, mat.shape[1]), replace=False)
        mat = mat[np.ix_(r, c)]
    try:
        sv = np.linalg.svd(mat, compute_uv=False)
        if len(sv) > 1 and sv[1] > 1e-12:
            return float(sv[0] / sv[1])
    except Exception:
        pass
    return 1.0


def _neuron_norm_ratio(tensor: np.ndarray) -> float:
    """Compute pruning-safe max/median neuron L2 norm ratio with P90 tail isolation."""
    if tensor.ndim < 2 or tensor.shape[0] < MIN_NEURONS:
        return 1.0
    norms = np.linalg.norm(tensor.reshape(tensor.shape[0], -1), axis=1)
    active = norms[norms >= ACTIVE_NORM_FLOOR]
    if active.size < MIN_NEURONS:
        return 1.0

    med = float(np.median(active))
    p90 = float(np.percentile(active, 90))
    max_norm = float(active.max())
    if med < 1e-12 or p90 < 1e-12:
        return 1.0

    # Require tail outlier isolation above the 90th percentile to ignore bimodal clean layers
    if (max_norm / p90) < INFLATED_NEURON_P90_THRESH:
        return 1.0
    return float(max_norm / med)


def _load_state_dict(model_path: Path) -> dict[str, np.ndarray] | None:
    ext = model_path.suffix.lower()

    # 1. SafeTensors support
    if ext == ".safetensors":
        try:
            from safetensors.torch import load_file
            import torch
            sd = load_file(str(model_path))
            out = {
                k: v.detach().cpu().numpy()
                for k, v in sd.items()
                if isinstance(v, torch.Tensor) and v.dtype == torch.float32
            }
            return out if out else None
        except Exception:
            return None

    # 2. HDF5 support
    if ext in (".h5", ".hdf5"):
        try:
            import h5py
            out: dict[str, np.ndarray] = {}

            def _visit(name: str, obj: Any) -> None:
                if isinstance(obj, h5py.Dataset) and obj.dtype == np.float32:
                    out[name] = np.array(obj, dtype=np.float32)

            with h5py.File(str(model_path), "r") as f:
                f.visititems(_visit)
            return out if out else None
        except Exception:
            return None

    # 3. PyTorch / Pickle checkpoint support
    try:
        import torch
        sd = torch.load(str(model_path), map_location="cpu", weights_only=True)
        if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
            sd = sd["state_dict"]
        if not isinstance(sd, dict):
            return None
        out = {}
        for k, v in sd.items():
            if hasattr(v, "dtype") and v.dtype == torch.float32:
                out[k] = v.detach().cpu().numpy()
        return out if out else None
    except Exception:
        return None


def scan(model_path: str | Path) -> dict[str, Any]:
    path = Path(model_path)
    tensors = _load_state_dict(path)
    if tensors is None:
        return {
            "layer": "3b",
            "verdict": False,
            "score": 0.0,
            "flagged_layers": [],
            "signals": {
                "inflated_neuron": [],
                "spectral_collapse": [],
                "extreme_kurtosis": [],
            },
            "coverage_note": "Could not load float32 tensors from file.",
        }

    signals: dict[str, list[str]] = {
        "inflated_neuron": [],
        "spectral_collapse": [],
        "extreme_kurtosis": [],
    }
    flagged_layer_details: list[tuple[str, list[str], bool]] = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, arr in tensors.items():
            if not (name.endswith(".weight") or name.endswith("/weight") or name.endswith(":0")):
                continue
            if arr.ndim < 2 or arr.size < MIN_LAYER_WEIGHTS or arr.shape[0] < MIN_NEURONS:
                continue

            active_signals: list[str] = []
            is_dominant: bool = False

            # 1. Pruning-safe inflated neuron check
            neuron_ratio = _neuron_norm_ratio(arr)
            if neuron_ratio >= INFLATED_NEURON_RATIO_THRESH:
                signals["inflated_neuron"].append(name)
                active_signals.append("inflated_neuron")
                if neuron_ratio >= DOMINANT_NEURON_RATIO_THRESH:
                    is_dominant = True

            # 2. Active-submatrix spectral collapse check
            mat = arr.reshape(arr.shape[0], -1)
            sr = _spectral_ratio(mat)
            if sr >= SPECTRAL_RATIO_THRESH:
                signals["spectral_collapse"].append(name)
                active_signals.append("spectral_collapse")
                if sr >= DOMINANT_SPECTRAL_THRESH:
                    is_dominant = True

            # 3. Spatial-kernel extreme kurtosis check (skip 1x1 pointwise & classifier/projection layers)
            name_lower = name.lower()
            skip_kurtosis = any(k in name_lower for k in _SKIP_KURTOSIS_KEYS)
            is_pointwise_conv = (arr.ndim == 4 and (arr.shape[2] * arr.shape[3]) == 1)
            if not skip_kurtosis and not is_pointwise_conv:
                ku = _excess_kurtosis(arr.ravel())
                if ku >= EXTREME_KURTOSIS_THRESH:
                    signals["extreme_kurtosis"].append(name)
                    active_signals.append("extreme_kurtosis")
                    if ku >= DOMINANT_KURTOSIS_THRESH:
                        is_dominant = True

            if active_signals:
                flagged_layer_details.append((name, active_signals, is_dominant))

    flagged_layer_names = sorted(set(item[0] for item in flagged_layer_details))
    has_high_conviction = any(
        (len(item[1]) >= 2) or item[2] for item in flagged_layer_details
    )

    if flagged_layer_names:
        raw_score = len(flagged_layer_names) * 0.35 + (0.40 if has_high_conviction else 0.0)
        score = float(np.clip(raw_score, 0.35, 1.0))
        verdict = True
    else:
        score = 0.0
        verdict = False

    return {
        "layer": "3b",
        "verdict": verdict,
        "score": round(score, 4),
        "flagged_layers": flagged_layer_names,
        "signals": signals,
        "coverage_note": None,
    }
