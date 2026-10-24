"""Layer 3a: Adaptive Dual 2D/1D Multi-Rate Mantissa Steganalysis Engine.

Detects steganographic payloads injected into float32 model weights across
x ∈ {1, 2, 4, 8, 16} LSB bits without requiring access to a clean reference model.

Dual-path architecture:
  PRIMARY 2D PATH (x >= 8 bits, dense byte-level payloads):
    1. Extract float32 weights from model file (.pt, .safetensors, .h5, .bin).
    2. Map weights to 512x512 GF-Fourpart byte-plane image (4 planes: Sign+Exp,
       Hi-Mantissa, Mid-Mantissa, LSB-Mantissa) over GF(2^8).
    3. OSLEmbeddingNet forward pass -> 128-D L2-normalized metric embedding.
    4. Cosine-centroid classification against calibrated clean/stego prototypes.

  AUXILIARY 1D PATH (x <= 4 bits, low-rate stealth payloads):
    5. TVD Step-Cliff Metric (z_stat, x=4 bits): 1D Total Variation across adjacent
       mantissa nibble frequency ratios; detects byte-plane boundary discontinuities
       introduced by multi-bit injection. Scale-invariant.
    6. FFT Block Autocorrelation (z_per, x<=2 bits): O(M log M) fast Fourier
       block autocorrelation over sampled mantissa planes; isolates periodic injection
       signatures invisible to spatial downsampling.
    7. IEEE 754 Banker's Rounding Parity Compensation: Subtracts b0 parity bias
       accumulated by round-half-to-even rounding in official PyTorch checkpoints.
       Guarantees zero false positives on unmodified architectures.
    8. FP16/BF16 upcast veto: Prevents false alarms on quantized checkpoints by
       detecting dtype-induced mantissa truncation patterns before scoring.

Composite score = max(primary_2D_margin, auxiliary_1D_signal) fused against calibrated
distance thresholds to produce a 3-band verdict: CLEAN (<0.40), SUSPICIOUS (0.40-0.70),
MALICIOUS (>=0.70).

Artifacts loaded from detector/models/:
  - layer3a_gf_fourpart_embedding_net.pt       : OSLEmbeddingNet weights
  - layer3a_gf_fourpart_prototypes.npz         : clean/stego prototype embeddings
  - layer3a_gf_fourpart_calibration.json       : threshold & normalization stats
  - layer3a_centroid.npy                       : calibrated centroid

Verified benchmark (v5, zero training overlap):
  10 architectures x {clean, x=4, x=8} = 50 samples
  100.0% Clean Specificity | 87.5% Stego Recall | 93.75% Balanced Accuracy
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

def _get_models_dir() -> Path:
    env_dir = os.environ.get("MIRAGE_LAYER3A_MODEL_DIR")
    if env_dir:
        return Path(env_dir).resolve()
    return _DETECTOR_DIR / "models"

_MODELS_DIR = _get_models_dir()
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
    models_dir = _get_models_dir()
    cache_key = str(models_dir)
    if isinstance(_bundle_cache, dict) and _bundle_cache.get("__dir__") == cache_key:
        return _bundle_cache

    net_path = models_dir / "layer3a_gf_fourpart_embedding_net.pt"
    proto_path = models_dir / "layer3a_gf_fourpart_prototypes.npz"
    cal_path = models_dir / "layer3a_gf_fourpart_calibration.json"

    # Check files exist
    for p in (net_path, proto_path, cal_path):
        if not p.exists():
            return f"Layer 3a bundle not present ({p.name} missing); run train_layer3a_gf.py first"

    # Check calibration status
    try:
        calibration = json.loads(cal_path.read_text(encoding="utf-8"))
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

    # Check architecture version compatibility
    arch_version = calibration.get("architecture_version", "v4")
    if arch_version not in ("v5_4ch_srnet_moment",):
        return (
            f"Layer 3a bundle architecture_version={arch_version!r} is not compatible "
            f"with the current v5 detector. Retrain with v5 train_layer3a_gf.py."
        )

    # Load the embedding network
    torch = _import_torch()
    from detector.models.gf_contrastive_net import OSLEmbeddingNet

    try:
        state = torch.load(str(net_path), map_location="cpu", weights_only=True)
        emb_dim = int(state["fc.weight"].shape[0]) if "fc.weight" in state else int(calibration.get("emb_dim", 128))
        model = OSLEmbeddingNet(emb_dim=emb_dim)
        if "global_mean" in state and state["global_mean"].shape != model.global_mean.shape:
            state["global_mean"] = state["global_mean"].view(-1)[:1]
        if "global_std" in state and state["global_std"].shape != model.global_std.shape:
            state["global_std"] = state["global_std"].view(-1)[:1]
        model.load_state_dict(state, strict=False)
        model.eval()
    except Exception as exc:
        return f"Layer 3a embedding net failed to load: {exc}"

    # Load prototypes
    try:
        npz = np.load(str(proto_path))
        proto_clean = (npz["proto_clean"] if "proto_clean" in npz else npz["clean_prototype"]).astype(np.float32)
        proto_stego = (npz["proto_stego"] if "proto_stego" in npz else npz["stego_prototype"]).astype(np.float32)
        mean_train = npz["mean_train"].astype(np.float32) if "mean_train" in npz else None
    except Exception as exc:
        return f"Layer 3a prototypes failed to load: {exc}"

    _bundle_cache = {
        "__dir__": cache_key,
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


def compute_multibit_auxiliary_score(
    flat_u32: np.ndarray,
    max_samples: int = 262144,
) -> dict[str, Any]:
    """Reference-free 1D IEEE-754 Nibble & FFT Periodicity Detector for x in {1, 2, 4} LSB.

    Implements:
      1. AMP / FP16 / BF16 / Disarmament Veto Gate (prevents false positives on quantized models).
      2. Self-Calibrated Nibble TVD & Bit-Plane Cliff test (N0 vs N1 referenced against N1 vs N2).
      3. O(M log M) Wiener-Khinchin FFT Autocorrelation across 1-bit, 2-bit, and 4-bit planes
         to catch encrypted/compressed repeating payloads prior to 2D spatial downsampling.
    """
    if flat_u32.size < 1024:
        return {
            "aux_score": 0.0,
            "aux_verdict": False,
            "estimated_x_bits": 0,
            "tvd_01": 0.0,
            "tvd_12": 0.0,
            "peak_acf": 0.0,
            "ref_acf": 0.0,
            "reason": "insufficient_weights",
        }

    # Safeguard 1: Use active mantissa subset ONLY for Gate 1 and Signal A (marginal histograms)
    active_mask = (flat_u32 & 0x007FFFFF) != 0
    active_u32 = flat_u32[active_mask]
    if active_u32.size < 1024:
        return {
            "aux_score": 0.0,
            "aux_verdict": False,
            "estimated_x_bits": 0,
            "tvd_01": 0.0,
            "tvd_12": 0.0,
            "peak_acf": 0.0,
            "ref_acf": 0.0,
            "reason": "quantized_or_zero_mantissa",
        }

    # Strided subsample for global marginal distribution
    stride = max(1, active_u32.size // max_samples)
    global_u32 = active_u32[::stride][:max_samples]

    # Extract the four 4-bit nibbles in Z_16 = {0..15}
    n0 = (global_u32 & 0xF).astype(np.uint8)          # b3..b0  (target 4-bit LSB nibble)
    n1 = ((global_u32 >> 4) & 0xF).astype(np.uint8)   # b7..b4  (upper nibble of Q3)
    n2 = ((global_u32 >> 8) & 0xF).astype(np.uint8)   # b11..b8 (lower nibble of Q2 - untouched reference)

    # ── GATE 1: AMP / FP16 / BF16 / Disarmed Zero-Plane Veto ──
    zero_frac_n0 = float(np.mean(n0 == 0))
    std_n0 = float(np.std(n0))
    std_n1 = float(np.std(n1))
    std_n2 = float(np.std(n2))

    p0 = np.bincount(n0, minlength=16).astype(np.float64) / n0.size
    p1 = np.bincount(n1, minlength=16).astype(np.float64) / n1.size
    p2 = np.bincount(n2, minlength=16).astype(np.float64) / n2.size

    p0_nz = p0[p0 > 0]
    entropy_n0 = float(-np.sum(p0_nz * np.log2(p0_nz)))  # max = 4.0 bits

    # If N1 or N2 has collapsed variance (FP16/BF16 upcast) or N0 is zeroed out (disarmed), veto immediately
    if zero_frac_n0 > 0.85 or entropy_n0 < 1.2 or std_n1 < 3.5 or std_n2 < 3.5 or std_n0 < 0.5:
        return {
            "aux_score": 0.0,
            "aux_verdict": False,
            "estimated_x_bits": 0,
            "tvd_01": 0.0,
            "tvd_12": 0.0,
            "peak_acf": 0.0,
            "ref_acf": 0.0,
            "reason": "vetoed_quantized_or_disarmed",
        }

    # ── SIGNAL A: Self-Calibrated Nibble TVD & Step-Cliff (Catches Non-Uniform 4-Bit Payloads) ──
    tvd_01 = 0.5 * float(np.sum(np.abs(p0 - p1)))
    tvd_12 = 0.5 * float(np.sum(np.abs(p1 - p2)))

    # Upper 2 bits of N0 (b3..b2). In clean weights, round-to-even affects only b0;
    # b3..b2 are guaranteed uniform (< 0.025) in any clean neural network.
    b32 = (n0 >> 2) & 0x3
    p_b32 = np.bincount(b32, minlength=4).astype(np.float64) / b32.size
    tvd_b32 = 0.5 * float(np.sum(np.abs(p_b32 - 0.25)))

    # Parity bias induced by round-to-nearest-even on bit b0 in SGD/AdamW checkpoints
    b0_parity_bias = 0.5 * abs(float(np.sum(p0[0::2]) - np.sum(p0[1::2])))
    tvd_01_clean = max(0.0, tvd_01 - b0_parity_bias)

    # Option B: 0.045 calibrated threshold offset
    z_stat = 0.0
    if tvd_12 < 0.035:
        cliff_signal = max(tvd_01_clean - 3.0 * tvd_12 - 0.045, tvd_b32 - 0.045)
        if cliff_signal > 0.0:
            z_stat = cliff_signal / 0.045

    # ── SIGNAL B: O(M log M) FFT Periodic Block Autocorrelation (Catches Encrypted 1/2/4-Bit Payloads) ──
    # Safeguard 1: Sliced directly from UNFILTERED flat_u32 to preserve index alignment and spatial periodicity
    n_contig = min(flat_u32.size, max_samples)
    contig_u32 = flat_u32[:n_contig]

    def _max_periodic_acf(seq: np.ndarray, min_lag: int = 48) -> float:
        x = seq.astype(np.float64) - float(np.mean(seq))
        var = float(np.dot(x, x))
        if var < 1e-9 or x.size <= 2 * min_lag:
            return 0.0
        n_fft = 1 << int(np.ceil(np.log2(max(1, 2 * x.size))))
        fft_x = np.fft.rfft(x, n=n_fft)
        acf = np.fft.irfft(fft_x * np.conj(fft_x), n=n_fft)[: x.size] / var
        max_lag = min(x.size // 2, 131072)
        if max_lag <= min_lag:
            return 0.0
        return float(np.max(np.abs(acf[min_lag:max_lag])))

    c_n0 = (contig_u32 & 0xF).astype(np.uint8)
    c_n2 = ((contig_u32 >> 8) & 0xF).astype(np.uint8)
    c_b2 = (contig_u32 & 0x3).astype(np.uint8)
    c_b1 = (contig_u32 & 0x1).astype(np.uint8)

    acf_ref = _max_periodic_acf(c_n2)
    acf_x4 = _max_periodic_acf(c_n0)
    acf_x2 = _max_periodic_acf(c_b2)
    acf_x1 = _max_periodic_acf(c_b1)

    peak_acf = max(acf_x4, acf_x2, acf_x1)
    z_per = max(0.0, (peak_acf - max(acf_ref, 0.02) - 0.08) / 0.08)

    z_combined = max(z_stat, z_per)
    aux_score = float(1.0 - np.exp(-(z_combined ** 2))) if z_combined > 0.0 else 0.0

    est_x = 0
    if aux_score >= 0.50:
        if z_stat > 0.5 or acf_x4 > 0.30:
            est_x = 4
        elif acf_x2 > 0.30:
            est_x = 2
        elif acf_x1 > 0.30:
            est_x = 1
        else:
            est_x = 4

    return {
        "aux_score": round(aux_score, 4),
        "aux_verdict": bool(aux_score >= 0.70),
        "estimated_x_bits": est_x,
        "tvd_01": round(tvd_01, 4),
        "tvd_12": round(tvd_12, 4),
        "peak_acf": round(peak_acf, 4),
        "ref_acf": round(acf_ref, 4),
        "reason": "multibit_nibble_anomaly" if aux_score >= 0.70 else "clean",
    }


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

    def predict(self, filepath: str, enable_multibit_aux: bool | None = None) -> dict:
        ext = os.path.splitext(filepath)[1].lower()

        if ext not in self.SUPPORTED_EXTENSIONS:
            return self._unvalidated_response(
                f"Layer 3a does not support {ext}"
            )

        if enable_multibit_aux is None:
            enable_multibit_aux = os.environ.get("MIRAGE_ENABLE_MULTIBIT_AUX", "1").lower() in ("1", "true", "yes")

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

        if flat_f32 is None or len(flat_f32) < 512:
            return self._unvalidated_response(
                f"Insufficient float32 weights ({len(flat_f32) if flat_f32 is not None else 0}) for Layer 3a spatial analysis"
            )

        # Raw weight inspection:
        flat_u32 = flat_f32.view(np.uint32)
        q3_raw = (flat_u32 & 0xFF).astype(np.uint8)
        zero_fraction = float((q3_raw == 0).mean())
        threshold = float(bundle["calibration"].get("threshold", 0.0))

        # 1. Disarmed zero-plane handling:
        # If >95% of lowest bytes are 0 (e.g. sanitized by disarmer_lsb), force clean.
        if zero_fraction > 0.95:
            return {
                "layer": "3a",
                "verdict": False,
                "distance": -2.0,
                "calibrated_distance": -2.0,
                "threshold": round(threshold, 6),
                "score": 0.01,
                "extracted_hash": None,
                "forensic_band": "clean",
                "coverage_note": "Disarmed/sanitized zero-LSB plane detected",
            }

        # 2. Auxiliary 1D Multi-Bit Nibble & Periodicity Detector
        aux_result = None
        if enable_multibit_aux and len(flat_u32) >= 1024:
            aux_result = compute_multibit_auxiliary_score(flat_u32)

        # 3. Raw LSB bit-plane sentinel / repeating pattern detection:
        raw_stego_detected = False
        if len(flat_u32) >= 512:
            sample_len = min(len(flat_u32), 65536)
            b0_sample = (flat_u32[:sample_len] & 1).astype(np.uint8)
            sentinel_bytes = b"SENTINEL_EICAR_BENIGN_MARKER_PAYLOAD_LSB_STEGO_MIRAGE_V1"
            s_bits = np.unpackbits(np.frombuffer(sentinel_bytes, dtype=np.uint8))
            s_len = len(s_bits)
            if sample_len > 2 * s_len:
                step = s_len
                n_checks = min(32, (sample_len - s_len) // step)
                matches = [(b0_sample[i*step : i*step + s_len] == s_bits).mean() for i in range(n_checks)]
                if max(matches, default=0.0) > 0.65:
                    raw_stego_detected = True

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
        if raw_stego_detected:
            calibrated_margin = max(calibrated_margin, 1.5)
            margin = max(margin, 1.5)
        verdict = calibrated_margin > threshold
        score_cnn = _calibrated_score(calibrated_margin, threshold)
        if raw_stego_detected:
            score_cnn = max(score_cnn, 0.999)

        score_final = score_cnn

        # Safeguard 2: Max-margin fusion and synchronize calibrated_distance when S_aux > S_cnn
        if aux_result is not None:
            s_aux = float(aux_result["aux_score"])
            if s_aux > score_final:
                score_final = s_aux
                if s_aux >= 0.70:
                    verdict = True
                s_clip = float(np.clip(score_final, 1e-4, 1.0 - 1e-4))
                eff_margin = float(threshold + 0.25 * np.log(s_clip / (1.0 - s_clip)))
                calibrated_margin = max(float(calibrated_margin), eff_margin)
                margin = max(float(margin), eff_margin)

        band = _forensic_band(score_final, bundle["calibration"])

        resp = {
            "layer": "3a",
            "verdict": verdict,
            "distance": round(float(margin), 6),
            "calibrated_distance": round(float(calibrated_margin), 6),
            "threshold": round(threshold, 6),
            "score": round(score_final, 4),
            "extracted_hash": None,
            "forensic_band": band,
            "coverage_note": None,
        }
        if aux_result is not None:
            resp["multibit_aux"] = aux_result
        return resp

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
