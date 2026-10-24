"""
detector/layer3b_weights.py - Lightweight Within-Model Weight Anomaly Detector
===============================================================================
Layer 3b: Reference-free heuristic for weight-space backdoor/trojan detection.
No reference model, no training, no test data required.

Three calibrated signals (all reference-free):
1. INFLATED_NEURON: Ratio of maximum neuron L2 norm to median neuron norm in a layer.
   In clean models, max/median norm ratio is strictly < 6.5 (typically 1.2 - 2.5).
   In trojaned models with neuron-level triggers, ratio easily exceeds 15.0 - 100.0+.
   Flag if max_norm / median_norm >= 15.0.

2. SPECTRAL_COLLAPSE: Singular value ratio (sigma_1 / sigma_2) of 2-D reshaped weight matrix.
   In clean convolutional and linear layers, singular values decay smoothly (sigma_1/sigma_2 <= 3.2).
   In rank-1 or low-rank trojan trigger injections, sigma_1/sigma_2 spikes >= 10.0+.
   Flag if sigma_1 / sigma_2 >= 10.0.

3. EXTREME_KURTOSIS: Excess kurtosis of conv/feature layer weight distributions.
   In clean weights, kurtosis is typically < 40.0.
   Bimodal backdoor distributions (clean weights + large injected trigger values)
   reach kurtosis >= 100.0+.
   Flag if excess kurtosis >= 100.0 (excluding final classification heads).

Output contract (matches backend/fusion.py expectations):
    {
        "layer":          "3b",
        "verdict":        bool,
        "score":          float,        # 0.0 - 1.0
        "flagged_layers": list[str],
        "signals":        dict[str, list[str]],
        "coverage_note":  str | None,
    }
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any
import numpy as np

# Calibrated thresholds (empirically validated across ResNet, SqueezeNet, ShuffleNet)
INFLATED_NEURON_RATIO_THRESH: float = 15.0
SPECTRAL_RATIO_THRESH: float        = 10.0
EXTREME_KURTOSIS_THRESH: float      = 100.0
MIN_LAYER_WEIGHTS: int              = 64
MIN_NEURONS: int                    = 4


def _excess_kurtosis(arr: np.ndarray) -> float:
    if arr.size < 4:
        return 0.0
    sigma = arr.std()
    if sigma < 1e-12:
        return 0.0
    z = (arr - arr.mean()) / sigma
    return float(np.mean(z ** 4)) - 3.0


def _spectral_ratio(mat: np.ndarray) -> float:
    if mat.ndim != 2 or min(mat.shape) < 2:
        return 1.0
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
    if tensor.ndim < 2 or tensor.shape[0] < MIN_NEURONS:
        return 1.0
    norms = np.linalg.norm(tensor.reshape(tensor.shape[0], -1), axis=1)
    med = float(np.median(norms))
    if med < 1e-12:
        return 1.0
    return float(norms.max() / med)


def _load_state_dict(model_path: Path) -> dict[str, np.ndarray] | None:
    try:
        import torch
        sd = torch.load(str(model_path), map_location="cpu", weights_only=True)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        if not isinstance(sd, dict):
            return None
        out: dict[str, np.ndarray] = {}
        for k, v in sd.items():
            if hasattr(v, "dtype") and v.dtype == torch.float32:
                out[k] = v.detach().cpu().numpy()
        return out if out else None
    except Exception:
        return None


def scan(model_path: str | Path) -> dict[str, Any]:
    """
    Run Layer 3b within-model weight anomaly detection.
    """
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
    flagged_layer_details: list[tuple[str, list[str]]] = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, arr in tensors.items():
            if not name.endswith(".weight") or arr.ndim < 2 or arr.size < MIN_LAYER_WEIGHTS:
                continue
            if arr.shape[0] < MIN_NEURONS:
                continue

            active_signals: list[str] = []

            # 1. Inflated neuron check
            neuron_ratio = _neuron_norm_ratio(arr)
            if neuron_ratio >= INFLATED_NEURON_RATIO_THRESH:
                signals["inflated_neuron"].append(name)
                active_signals.append("inflated_neuron")

            # 2. Spectral collapse check
            mat = arr.reshape(arr.shape[0], -1)
            sr = _spectral_ratio(mat)
            if sr >= SPECTRAL_RATIO_THRESH:
                signals["spectral_collapse"].append(name)
                active_signals.append("spectral_collapse")

            # 3. Extreme kurtosis (skip final fc/classifier layers where class sparsity is normal)
            is_classifier = any(cls_key in name.lower() for cls_key in ("fc", "classifier", "head"))
            if not is_classifier:
                ku = _excess_kurtosis(arr.ravel())
                if ku >= EXTREME_KURTOSIS_THRESH:
                    signals["extreme_kurtosis"].append(name)
                    active_signals.append("extreme_kurtosis")

            if active_signals:
                flagged_layer_details.append((name, active_signals))

    flagged_layer_names = sorted(set(item[0] for item in flagged_layer_details))
    has_consensus = any(len(item[1]) >= 2 for item in flagged_layer_details)

    if flagged_layer_names:
        raw_score = len(flagged_layer_names) * 0.35 + (0.40 if has_consensus else 0.0)
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


if __name__ == "__main__":
    import argparse, json as _json
    parser = argparse.ArgumentParser(description="Layer 3b weight anomaly scan")
    parser.add_argument("model_path", type=Path)
    args = parser.parse_args()
    print(_json.dumps(scan(args.model_path), indent=2))
