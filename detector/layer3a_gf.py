"""Layer 3a: GF-Fourpart + OSLEmbeddingNet nearest-centroid detector.

Loads model weights, prototypes, and calibration threshold from detector/models/.
Pipeline:
  1. Extract float32 weights from model file (.pt, .safetensors, .h5, .bin).
  2. Map weights to 512x512 GF-Fourpart byte-plane image.
  3. OSLEmbeddingNet forward pass -> 128-D L2-normalized embedding.
  4. Cosine-centroid classification against calibrated prototypes.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------

_torch = None
_safetensors = None
_h5py = None


def _import_torch():
    global _torch
    if _torch is None:
        import torch as _t
        _torch = _t
    return _torch


def _import_safetensors():
    global _safetensors
    if _safetensors is None:
        try:
            import safetensors.torch as _st
            _safetensors = _st
        except ImportError:
            _safetensors = None
    return _safetensors


def _import_h5py():
    global _h5py
    if _h5py is None:
        try:
            import h5py as _h
            _h5py = _h
        except ImportError:
            _h5py = None
    return _h5py


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DETECTOR_DIR = Path(__file__).resolve().parent
_MODELS_DIR = _DETECTOR_DIR / "models"
_NET_PATH = _MODELS_DIR / "layer3a_gf_fourpart_embedding_net.pt"
_PROTO_PATH = _MODELS_DIR / "layer3a_gf_fourpart_prototypes.npz"
_CAL_PATH = _MODELS_DIR / "layer3a_gf_fourpart_calibration.json"

_PROJECT_ROOT = _DETECTOR_DIR.parent

# Ensure detector package imports work when called from backend/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Step 1 — Weight extraction (all four formats)
# ---------------------------------------------------------------------------

def _extract_weights_flat(filepath: str) -> np.ndarray | None:
    """
    Extract and concatenate all float32 weight tensors from a model file.

    Returns a flat float32 numpy array, or None if no float32 weights found.
    Supports: .pt / .pth / .pkl / .bin, .safetensors, .h5 / .hdf5.
    """
    torch = _import_torch()
    ext = os.path.splitext(filepath)[1].lower()
    parts = []

    try:
        if ext in (".pt", ".pth", ".pkl", ".bin"):
            obj = torch.load(filepath, map_location="cpu", weights_only=True)
            if isinstance(obj, dict):
                for v in obj.values():
                    if isinstance(v, torch.Tensor) and v.dtype == torch.float32 and v.numel() > 0:
                        parts.append(v.detach().cpu().numpy().astype("<f4").flatten())
            elif isinstance(obj, torch.Tensor) and obj.dtype == torch.float32:
                parts.append(obj.detach().cpu().numpy().astype("<f4").flatten())

        elif ext == ".safetensors":
            st = _import_safetensors()
            if st is None:
                return None
            sd = st.load_file(filepath)
            for v in sd.values():
                if isinstance(v, torch.Tensor) and v.dtype == torch.float32 and v.numel() > 0:
                    parts.append(v.detach().cpu().numpy().astype("<f4").flatten())

        elif ext in (".h5", ".hdf5"):
            h5py = _import_h5py()
            if h5py is None:
                return None
            with h5py.File(filepath, "r") as f:
                def _visit(name, node):
                    if isinstance(node, h5py.Dataset):
                        arr = node[()]
                        if arr.dtype.kind == "f":
                            parts.append(arr.astype(np.float32).flatten())
                f.visititems(_visit)

    except Exception:
        pass

    if not parts:
        return None
    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# Step 2 — GF-Fourpart whole-model image (paper's SCZ preprocessing)
# ---------------------------------------------------------------------------

def _model_to_gf_image(flat_f32: np.ndarray, imsize: int = 100) -> np.ndarray:
    """
    Convert a flat float32 weight vector into the GF-Fourpart 100×100 image
    used by the paper's SCZ experiments and produced by
    ``fixtures/generate_patch_dataset.py::model_to_gf_image()``.

    All four IEEE-754 bytes of each weight are used (MSB→LSB tile layout),
    then the full-resolution image is anti-aliased bicubic-resized to
    ``imsize × imsize``.
    """
    try:
        from skimage.transform import resize as skimage_resize
    except ImportError as exc:
        raise RuntimeError(
            "_model_to_gf_image requires scikit-image; install requirements.txt"
        ) from exc

    n_weights = len(flat_f32)
    closest_sq = int(np.ceil(np.sqrt(n_weights)) ** 2)
    sqrt_sq = int(np.sqrt(closest_sq))

    # Pad to nearest square
    padded = np.pad(flat_f32, (0, closest_sq - n_weights), mode="constant")

    # Little-endian bytes, flip to MSB-first: index 0 = MSB (sign+exp)
    byte_planes = padded.view(np.uint8).reshape(closest_sq, 4)[:, ::-1]  # (N_sq, 4)

    # 2×2 tile layout [[Q0, Q1], [Q2, Q3]]
    tiles = [byte_planes[:, ch].reshape(sqrt_sq, sqrt_sq) for ch in range(4)]
    image_arr = np.block([[tiles[0], tiles[1]], [tiles[2], tiles[3]]])

    # Avoid skimage's float64 working buffer for >10k×10k planes (ConvNeXt
    # and similarly deep models).  Area interpolation is continuous pooling,
    # retaining the anti-aliasing behaviour needed for downsampling.
    if image_arr.shape[0] > 10_000:
        torch = _import_torch()
        image_tensor = torch.from_numpy(np.ascontiguousarray(image_arr)).unsqueeze(0).unsqueeze(0).float()
        resized_tensor = torch.nn.functional.interpolate(
            image_tensor, size=(imsize, imsize), mode="area"
        )
        return resized_tensor.squeeze(0).squeeze(0).round().clamp(0, 255).to(torch.uint8).numpy()

    resized = skimage_resize(
        image_arr,
        output_shape=(imsize, imsize),
        order=1,  # bilinear (matches generate_patch_dataset and model_xray procedures)
        preserve_range=True,
        anti_aliasing=True,
        clip=True,
    )
    return np.rint(resized).clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Step 3 — Bundle loading
# ---------------------------------------------------------------------------

_bundle_cache: dict | None = None  # cached after first successful load


def _load_bundle() -> dict | str:
    """
    Load the v4 training bundle from detector/models/.

    Returns a dict with keys:
        model       — OSLEmbeddingNet (eval mode, on CPU)
        proto_clean — np.ndarray (emb_dim,)
        proto_stego — np.ndarray (emb_dim,)
        calibration — dict (full calibration JSON)

    Returns a string (error/coverage message) if the bundle is missing or
    not validated.
    """
    global _bundle_cache
    if _bundle_cache is not None:
        return _bundle_cache

    # Check files exist
    for p in (_NET_PATH, _PROTO_PATH, _CAL_PATH):
        if not p.exists():
            return f"Layer 3a bundle not present ({p.name} missing); run train_layer3a_gf.py first"

    # Check calibration status
    try:
        calibration = json.loads(_CAL_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        return f"Layer 3a calibration JSON unreadable: {exc}"

    if calibration.get("status") not in ("validated", "completed"):
        status = calibration.get("status", "unknown")
        dev_ba = calibration.get("dev_centroid_balanced_accuracy", "?")
        gate = calibration.get("min_dev_balanced_accuracy_gate", "?")
        return (
            f"Layer 3a bundle status={status!r} (dev BA={dev_ba}, gate={gate}); "
            f"retrain until bundle is validated before production use"
        )

    # Load the embedding network
    torch = _import_torch()
    from detector.models.gf_contrastive_net import OSLEmbeddingNet

    try:
        state = torch.load(str(_NET_PATH), map_location="cpu", weights_only=True)
        emb_dim = int(state["fc.weight"].shape[0]) if "fc.weight" in state else int(calibration.get("emb_dim", 128))
        model = OSLEmbeddingNet(emb_dim=emb_dim)
        model.load_state_dict(state)
        model.eval()
    except Exception as exc:
        return f"Layer 3a embedding net failed to load: {exc}"

    # Load prototypes
    try:
        npz = np.load(str(_PROTO_PATH))
        proto_clean = (npz["proto_clean"] if "proto_clean" in npz else npz["clean_prototype"]).astype(np.float32)
        proto_stego = (npz["proto_stego"] if "proto_stego" in npz else npz["stego_prototype"]).astype(np.float32)
        mean_train = npz["mean_train"].astype(np.float32) if "mean_train" in npz else None
    except Exception as exc:
        return f"Layer 3a prototypes failed to load: {exc}"

    _bundle_cache = {
        "model": model,
        "proto_clean": proto_clean,
        "proto_stego": proto_stego,
        "mean_train": mean_train,
        "calibration": calibration,
    }
    return _bundle_cache


def _invalidate_bundle_cache() -> None:
    """Force the next call to _load_bundle() to reload from disk."""
    global _bundle_cache
    _bundle_cache = None


# ---------------------------------------------------------------------------
# Step 4 — Embed one GF image using the loaded bundle
# ---------------------------------------------------------------------------

def _embed(gf_image: np.ndarray, model) -> np.ndarray:
    """Run the OSLEmbeddingNet on a 100×100 GF-Fourpart image."""
    torch = _import_torch()

    # Shape: [1, 1, H, W] as uint8 (model expects raw uint8 values, normalizes internally)
    tensor = torch.from_numpy(gf_image.astype(np.uint8)).unsqueeze(0).unsqueeze(0)  # [1,1,100,100]

    with torch.no_grad():
        emb = model(tensor)  # [1, emb_dim]
    return emb.squeeze(0).cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Step 5 — Nearest-centroid classification
# ---------------------------------------------------------------------------

def _cl2_cosine_margin(
    emb: np.ndarray,
    proto_clean: np.ndarray,
    proto_stego: np.ndarray,
    mean_train: np.ndarray | None,
) -> float:
    """Return the calibrated CL2 cosine margin used by the training export."""
    query = emb.astype(np.float32, copy=False)
    if mean_train is not None:
        query = query - mean_train
    query = query / max(float(np.linalg.norm(query)), 1e-7)
    clean = proto_clean / max(float(np.linalg.norm(proto_clean)), 1e-7)
    stego = proto_stego / max(float(np.linalg.norm(proto_stego)), 1e-7)
    return float(np.dot(query, stego) - np.dot(query, clean))


def _size_aware_adjustment(calibration: dict, parameter_count: int) -> float:
    """Return an opt-in log-parameter correction; disabled bundles return zero."""
    policy = calibration.get("size_aware_correction", {})
    if not isinstance(policy, dict) or not policy.get("enabled", False) or parameter_count <= 0:
        return 0.0
    try:
        if policy.get("version") != "log10_parameter_count_v1":
            return 0.0
        slope = float(policy["slope"])
        reference = float(policy["reference_log10_parameter_count"])
        if not np.isfinite(slope) or not np.isfinite(reference):
            return 0.0
        return slope * (np.log10(parameter_count) - reference)
    except (KeyError, TypeError, ValueError):
        return 0.0


def _calibrated_score(margin: float, threshold: float) -> float:
    """Training-compatible confidence centered on the calibrated boundary."""
    return float(1.0 / (1.0 + np.exp(-4.0 * (margin - threshold))))


def _forensic_band(score: float, calibration: dict) -> str:
    bands = calibration.get("forensic_bands", {})
    suspicious_min = float(bands.get("suspicious_band_min", 0.40))
    malicious_min = float(bands.get("malicious_band_min", 0.70))
    if score >= malicious_min:
        return "malicious"
    if score >= suspicious_min:
        return "suspicious"
    return "clean"


# ---------------------------------------------------------------------------
# Main detector class
# ---------------------------------------------------------------------------

class Layer3aGFDetector:
    """
    Layer 3a production detector: GF-Fourpart + OSLEmbeddingNet nearest-centroid.

    Loads the bundle produced by ``detector/train_layer3a_gf.py``. Returns
    ``verdict: False`` with a ``coverage_note`` if no validated bundle exists.
    """

    SUPPORTED_EXTENSIONS = {".pt", ".pth", ".pkl", ".bin", ".safetensors", ".h5", ".hdf5"}

    def _unvalidated_response(self, note: str) -> dict:
        return {
            "layer": "3a",
            "verdict": False,
            "distance": 0.0,
            "calibrated_distance": 0.0,
            "threshold": 0.0,
            "score": 0.0,
            "extracted_hash": None,
            "forensic_band": "clean",
            "coverage_note": note,
        }

    def predict(self, filepath: str) -> dict:
        ext = os.path.splitext(filepath)[1].lower()

        if ext not in self.SUPPORTED_EXTENSIONS:
            return self._unvalidated_response(
                f"Layer 3a does not support {ext}"
            )

        # Load bundle (cached after first call)
        bundle = _load_bundle()
        if isinstance(bundle, str):
            # bundle is an error/coverage string
            return self._unvalidated_response(bundle)

        # Extract weights
        try:
            flat_f32 = _extract_weights_flat(filepath)
        except Exception as exc:
            return self._unvalidated_response(f"Weight extraction failed: {exc}")

        if flat_f32 is None or len(flat_f32) == 0:
            return self._unvalidated_response(f"No float32 weights found in {ext} file")

        # Build GF image (512x512 resolution matching training corpus)
        try:
            imsize = int(bundle["calibration"].get("imsize", 512))
            gf_image = _model_to_gf_image(flat_f32, imsize=imsize)
        except Exception as exc:
            return self._unvalidated_response(f"GF image construction failed: {exc}")

        # Embed
        try:
            emb = _embed(gf_image, bundle["model"])
        except Exception as exc:
            return self._unvalidated_response(f"Embedding failed: {exc}")

        # Classify
        parameter_count = int(len(flat_f32))
        margin = _cl2_cosine_margin(
            emb, bundle["proto_clean"], bundle["proto_stego"], bundle["mean_train"]
        )
        calibrated_margin = margin - _size_aware_adjustment(bundle["calibration"], parameter_count)
        threshold = float(bundle["calibration"].get("threshold", 0.0))
        verdict = calibrated_margin > threshold
        score = _calibrated_score(calibrated_margin, threshold)
        band = _forensic_band(score, bundle["calibration"])

        return {
            "layer": "3a",
            "verdict": verdict,
            "distance": round(float(margin), 6),
            "calibrated_distance": round(float(calibrated_margin), 6),
            "threshold": round(threshold, 6),
            "score": round(score, 4),
            "extracted_hash": None,
            "forensic_band": band,
            "coverage_note": None,
        }

    def reload_bundle(self) -> None:
        """Force the bundle to be reloaded from disk on the next predict() call."""
        _invalidate_bundle_cache()


# ---------------------------------------------------------------------------
# CLI for quick manual testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python detector/layer3a_gf.py <path/to/model> [<path/to/model> ...]")
        sys.exit(1)

    detector = Layer3aGFDetector()
    for path in sys.argv[1:]:
        result = detector.predict(path)
        verdict_str = "STEGO" if result["verdict"] else "CLEAN"
        print(
            f"{os.path.basename(path)}: {verdict_str}  "
            f"margin={result['distance']:.4f}  score={result['score']:.4f}"
            + (f"  [{result['coverage_note']}]" if result["coverage_note"] else "")
        )
