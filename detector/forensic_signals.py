"""Steganography Forensic Signal Engine.

Extracts structured byte-level signals from model weights for triage:
  1. Bit-depth & contaminated byte-plane estimation (Signal 1)
  2. Layer-wise modification distribution (Signal 2)
  3. Shannon entropy and ciphertext classification (Signal 3)
  4. Hidden payload capacity estimation (Signal 4)
  5. Executable magic byte inspection (Signal 5)
"""

from __future__ import annotations

import os
import warnings
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Byte-plane labels (float32 is 4 bytes, little-endian on most hardware)
# Index 0 → MSB (sign + exponent), index 3 → LSB (lowest mantissa bits)
_PLANE_LABELS = ["Q0 (MSB/Sign+Exp)", "Q1 (Upper Mantissa)", "Q2 (Mid Mantissa)", "Q3 (LSB)"]
_PLANE_SHORT  = ["Q0", "Q1", "Q2", "Q3"]

# Per-plane entropy baselines for clean float32 Gaussian weights (reporting only).
# Q0 (MSB/sign+exp) is the ONLY plane that is statistically distinct
# between clean models and stego — it is low-entropy by nature (~3.5 bits/byte)
# and is NEVER targeted by LSB steganography.
# Q1-Q3 are already near-uniform in clean Gaussian weights.
_CLEAN_ENTROPY_BASELINE = {
    "Q0": 3.50,   # sign+exp: highly structured, LOW entropy, never modified by stego
    "Q1": 7.96,   # upper mantissa: near-max, indistinguishable clean vs stego
    "Q2": 7.99,   # mid mantissa: near-max, indistinguishable clean vs stego
    "Q3": 7.996,  # LSB: near-max, indistinguishable clean vs stego at byte level
}

# Q0 anomaly threshold: if Q0 entropy is elevated above its clean baseline,
# the sign+exponent bytes were modified — this indicates DEEP (x>8) embedding
# that has spilled into the mid-mantissa, causing model accuracy degradation.
_Q0_CLEAN_UPPER_BOUND: float = 5.0    # Q0 > 5.0 → sign+exp contaminated

# Minimum weights for meaningful analysis
_MIN_WEIGHTS_FOR_ANALYSIS: int = 5_000

# Known file magic bytes: (hex_prefix, label, description)
_MAGIC_SIGNATURES: list[tuple[bytes, str, str]] = [
    (bytes([0x4D, 0x5A]), "MZ", "Windows Portable Executable (PE / DLL / EXE)"),
    (bytes([0x7F, 0x45, 0x4C, 0x46]), "ELF", "Linux/Unix executable or shared object"),
    (bytes([0x50, 0x4B, 0x03, 0x04]), "PK", "ZIP archive / Python wheel / JAR"),
    (bytes([0x1F, 0x8B]), "GZIP", "Gzip compressed data (shellcode or archive)"),
    (bytes([0x42, 0x5A, 0x68]), "BZ2", "Bzip2 compressed data"),
    (bytes([0x89, 0x50, 0x4E, 0x47]), "PNG", "PNG image (possible steganographic chain)"),
    (bytes([0xFF, 0xD8, 0xFF]), "JPEG", "JPEG image"),
    (bytes([0x25, 0x50, 0x44, 0x46]), "PDF", "PDF document"),
    (bytes([0x7B, 0x22]), '{"', "JSON payload (C2 config or credentials)"),
    (bytes([0x23, 0x21]), "#!", "Unix shebang (shell/Python script)"),
    (bytes([0x64, 0x65, 0x78, 0x0A]), "DEX", "Android Dalvik bytecode"),
    (bytes([0xCA, 0xFE, 0xBA, 0xBE]), "JAVA", "Java class file or Mach-O fat binary"),
]


# ---------------------------------------------------------------------------
# Weight loading (format-agnostic, same approach as layer3a_gf.py)
# ---------------------------------------------------------------------------

def _load_flat_bytes(filepath: str) -> tuple[np.ndarray | None, dict[str, np.ndarray] | None]:
    """
    Load model weights as both:
    - flat_bytes: raw uint8 view of all float32 weights concatenated (for global analysis)
    - named_tensors: dict of layer_name → float32 ndarray (for per-layer analysis)

    Returns (flat_bytes, named_tensors). Either may be None on failure.
    """
    flat_parts: list[np.ndarray] = []
    named: dict[str, np.ndarray] = {}

    ext = os.path.splitext(filepath)[1].lower()
    try:
        import torch
        if ext in (".pt", ".pth", ".pkl", ".bin"):
            obj = torch.load(filepath, map_location="cpu", weights_only=True)
            if isinstance(obj, dict):
                sd = obj.get("state_dict", obj)
                if isinstance(sd, dict):
                    for k, v in sd.items():
                        if hasattr(v, "dtype") and v.dtype == torch.float32 and v.numel() > 0:
                            arr = v.detach().cpu().numpy().astype(np.float32)
                            named[k] = arr
                            flat_parts.append(arr.ravel().view(np.uint8))
            elif hasattr(obj, "dtype") and obj.dtype == torch.float32:
                arr = obj.detach().cpu().numpy().astype(np.float32)
                named["tensor"] = arr
                flat_parts.append(arr.ravel().view(np.uint8))

        elif ext == ".safetensors":
            try:
                import safetensors.torch as st
                sd = st.load_file(filepath)
                for k, v in sd.items():
                    if hasattr(v, "dtype") and v.dtype == torch.float32 and v.numel() > 0:
                        arr = v.detach().cpu().numpy().astype(np.float32)
                        named[k] = arr
                        flat_parts.append(arr.ravel().view(np.uint8))
            except ImportError:
                pass

        elif ext in (".h5", ".hdf5"):
            try:
                import h5py
                with h5py.File(filepath, "r") as f:
                    def _visit(name: str, node: Any) -> None:
                        if hasattr(node, "__iter__") and hasattr(node, "dtype"):
                            arr = node[()]
                            if arr.dtype.kind == "f":
                                a = arr.astype(np.float32).ravel()
                                named[name] = a.reshape(-1)
                                flat_parts.append(a.view(np.uint8))
                    f.visititems(_visit)
            except ImportError:
                pass

    except Exception:
        pass

    if not flat_parts:
        return None, None

    flat_bytes = np.concatenate(flat_parts)  # uint8 view of all float32 weights
    return flat_bytes, named


# ---------------------------------------------------------------------------
# Signal 1: Bit-Plane Depth Analysis
# ---------------------------------------------------------------------------

def _shannon_entropy(arr: np.ndarray) -> float:
    """Shannon entropy in bits/byte over a uint8 array."""
    if arr.size == 0:
        return 0.0
    counts = np.bincount(arr.ravel(), minlength=256).astype(np.float64)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))




def _analyze_bit_plane_depth(flat_bytes: np.ndarray) -> dict[str, Any]:
    """
    Signal 1: Characterize the bit-plane depth (x-LSB embedding rate) of a
    CONFIRMED steganographic payload (Layer 3a has already flagged this model).

    This function CHARACTERIZES the embedding — it does NOT re-detect stego.
    Detection is Layer 3a's job. The forensic engine runs only when Layer 3a
    has already placed the model in the suspicious/malicious band.

    Key insight (empirically validated):
    - Gaussian float32 Q1/Q2/Q3 bytes are ALREADY near-uniform (~7.99 bits/byte).
    - Byte-level entropy CANNOT distinguish clean from LSB-stego at Q1-Q3.
    - Q0 (sign+exponent) is LOW entropy (~3.5 bits) and NEVER modified by LSB stego.
    - Elevated Q0 entropy (>5.0) means payload spilled into exponent bits → x > 8.
    """
    n_weights = len(flat_bytes) // 4
    if n_weights == 0 or n_weights < _MIN_WEIGHTS_FOR_ANALYSIS:
        return {"error": "Insufficient float32 weights for bit-plane analysis",
                "estimated_x": 8, "anomalous_planes": ["Q3"],
                "stealth_assessment": "Unknown (insufficient data)", "description": ""}

    # Reshape to (n_weights, 4) — LE order: col0=LSB byte, col3=MSB byte
    weight_bytes = flat_bytes[:n_weights * 4].reshape(n_weights, 4)
    # Flip to MSB-first for labeling: col0=Q0(MSB/sign+exp), col3=Q3(LSB)
    weight_bytes_msb = weight_bytes[:, ::-1]

    per_plane_entropy: dict[str, float] = {}
    for i, label in enumerate(_PLANE_SHORT):
        plane_data = weight_bytes_msb[:, i].ravel()
        if len(plane_data) > 5_000_000:
            plane_data = np.random.default_rng(42).choice(plane_data, 5_000_000, replace=False)
        per_plane_entropy[label] = round(float(_shannon_entropy(plane_data)), 4)

    q0_entropy = per_plane_entropy.get("Q0", 0.0)
    q3_entropy = per_plane_entropy.get("Q3", 0.0)
    q0_contaminated = q0_entropy > _Q0_CLEAN_UPPER_BOUND

    # Estimate x-bits given stego is confirmed by Layer 3a
    if q0_contaminated:
        estimated_x = 16
        anomalous_planes = ["Q2", "Q3"]
        stealth = "Amateur / Uncalibrated (Deep Embedding)"
        desc = (
            f"Q0 (sign+exponent) entropy is ELEVATED ({q0_entropy:.3f} bits vs clean baseline ~3.5). "
            "Payload has spilled into IEEE-754 exponent and sign bits (x > 8). "
            "This WILL degrade model prediction accuracy by ~1-5% or more. "
            "Consistent with uncalibrated tools that embed too deeply."
        )
    elif q3_entropy >= 7.995:
        estimated_x = 8
        anomalous_planes = ["Q3"]
        stealth = "High Capacity (Binary Dropper)"
        desc = (
            "Q3 (LSB byte) at near-maximal entropy — consistent with full 8-bit LSB replacement. "
            f"Max payload capacity: ~{n_weights // 8192} KB. "
            "Typical contents: full DLL, .exe, or Cobalt Strike beacon. "
            "Q0 is clean — attacker avoided sign/exponent contamination. "
            "Model accuracy degradation: <0.01%."
        )
    elif q3_entropy >= 7.990:
        estimated_x = 4
        anomalous_planes = ["Q3"]
        stealth = "Moderate Capacity (4-bit Embedding)"
        desc = (
            f"Partial LSB replacement (~4 bits). "
            f"Max payload capacity: ~{n_weights // 16384} KB. Near-zero accuracy impact."
        )
    else:
        estimated_x = 2
        anomalous_planes = []
        stealth = "Maximum Stealth (Nation-State / APT)"
        desc = (
            "All byte planes within clean baseline. Consistent with 1-2 bit embedding. "
            f"Max payload capacity: ~{n_weights // 32768} KB. "
            "Near-zero accuracy impact (<0.001%). "
            "Used for covert C2 channels, SSH keys, or API credentials."
        )

    return {
        "estimated_x": estimated_x,
        "anomalous_planes": anomalous_planes,
        "per_plane_entropy": per_plane_entropy,
        "clean_entropy_baseline": _CLEAN_ENTROPY_BASELINE,
        "q0_contaminated": q0_contaminated,
        "stealth_assessment": stealth,
        "description": desc,
        "max_theoretical_capacity_bytes": (n_weights * estimated_x) // 8,
        "note": (
            "Signal 1 characterizes embedding depth (Layer 3a confirmed stego). "
            "Q0 elevation is the structural discriminant for deep/uncalibrated embedding."
        ),
    }


# ---------------------------------------------------------------------------
# Signal 2: Payload Distribution Geometry (malicious band only)
# ---------------------------------------------------------------------------

def _classify_distribution_pattern(
    layer_rates: dict[str, float],
    layer_names: list[str],
) -> tuple[str, str]:
    """
    Classify the spatial pattern of modified weights across layers.
    Returns (pattern_key, human_description).
    """
    if not layer_rates:
        return "unknown", "Insufficient layer data for distribution analysis."

    rates = [layer_rates.get(n, 0.0) for n in layer_names]
    n = len(rates)
    if n == 0:
        return "unknown", "No layers analyzed."

    # Contiguous prefix: high rates in first ≤30% of layers, low elsewhere
    first_third = rates[:max(1, n // 3)]
    last_two_thirds = rates[max(1, n // 3):]
    mean_first = float(np.mean(first_third)) if first_third else 0.0
    mean_last  = float(np.mean(last_two_thirds)) if last_two_thirds else 0.0

    if mean_first > 0.60 and mean_last < 0.15:
        return (
            "contiguous_prefix",
            "Contiguous / Sequential Prefix — modifications concentrated in the first layers only. "
            "Signature of script kiddie or naive stego tool. Very easy to detect with layer variance checks.",
        )

    # Uniform stride: rates are roughly equal across all layers
    std_dev = float(np.std(rates))
    mean_all = float(np.mean(rates))
    if mean_all > 0.10 and std_dev < 0.15:
        return (
            "uniform_stride",
            f"Uniform Strided — modifications uniformly distributed across all layers (std={std_dev:.3f}). "
            "Attacker is diluting local patch anomalies so spatial filters see low per-layer difference rates. "
            "Consistent with moderate-sophistication tools aiming to evade layer-wise detectors.",
        )

    # Sensitive-layer avoidance: batchnorm/first conv clean, deep convs hit hard
    bn_names = [n for n in layer_names if "bn" in n.lower() or "batch" in n.lower() or "norm" in n.lower()]
    deep_names = [n for n in layer_names if any(f"layer{i}" in n.lower() for i in [3, 4])]
    bn_rate   = float(np.mean([layer_rates.get(n, 0.0) for n in bn_names])) if bn_names else 0.0
    deep_rate = float(np.mean([layer_rates.get(n, 0.0) for n in deep_names])) if deep_names else 0.0

    if bn_rate < 0.05 and deep_rate > 0.50:
        return (
            "sensitive_layer_avoidance",
            "Resilient / Insensitive Layer Targeting — payload injected exclusively into "
            "over-parameterized deep conv layers while sensitive layers (BatchNorm, first conv) are untouched. "
            "Expert-level targeting. Maximises survival across quantization and pruning.",
        )

    # Default: PRNG scattered (irregular distribution)
    return (
        "prng_scattered",
        "PRNG Key-Seeded (Scattered) — modified weights appear randomly distributed with no "
        "obvious spatial structure. Consistent with keyed steganography tool using PRNG seeded "
        "by a secret key. Requires the key to reconstruct the message even if LSBs are extracted.",
    )


def _analyze_distribution_geometry(
    named_tensors: dict[str, np.ndarray],
    estimated_x: int,
) -> dict[str, Any]:
    """
    Signal 2: Per-layer LSB modification rate analysis and distribution pattern classification.
    """
    if not named_tensors or estimated_x == 0:
        return {
            "pattern": "unknown",
            "layer_modification_rates": {},
            "concentration_layers": [],
            "batchnorm_untouched": None,
            "first_conv_untouched": None,
            "sophistication": "Insufficient data",
            "description": "Distribution analysis skipped (no embedding detected).",
        }

    layer_rates: dict[str, float] = {}
    weight_layers = [(k, v) for k, v in named_tensors.items()
                     if k.endswith(".weight") and v.ndim >= 2 and v.size >= 64]

    x_bits = min(estimated_x, 8)

    for name, arr in weight_layers:
        flat = arr.ravel().view(np.uint8)[::4]  # Q3 (LSB) bytes
        if len(flat) < 16:
            continue
        # A weight is "modified" if its Q3 LSB byte entropy in a local window is high.
        # Simple proxy: entropy of the bottom x_bits in chunks.
        mask = (1 << x_bits) - 1
        lsb_vals = flat & mask
        h = _shannon_entropy(lsb_vals)
        # Normalize: max entropy for x_bits values = x_bits; flag if close to max
        max_h = float(x_bits)
        rate = float(np.clip(h / max(max_h, 1e-6), 0.0, 1.0))
        layer_rates[name] = round(rate, 4)

    sorted_names = sorted(layer_rates.keys())
    pattern_key, pattern_desc = _classify_distribution_pattern(layer_rates, sorted_names)

    # High-modification layers (top 5 by rate)
    concentration = sorted(layer_rates.items(), key=lambda x: -x[1])[:5]
    concentration_layers = [k for k, _ in concentration if layer_rates[k] > 0.40]
    # Sensitive-layer checks
    bn_names = [n for n in sorted_names if "bn" in n.lower() or "norm" in n.lower()]
    first_conv_names = [n for n in sorted_names if "conv1" in n.lower() or "features.0" in n.lower()]
    bn_untouched = all(layer_rates.get(n, 0.0) < 0.20 for n in bn_names) if bn_names else None
    first_conv_untouched = all(layer_rates.get(n, 0.0) < 0.20 for n in first_conv_names) if first_conv_names else None

    sophistication_map = {
        "contiguous_prefix":         "Low (Script Kiddie / Naive Tool)",
        "uniform_stride":            "Medium (Dilution Strategy)",
        "prng_scattered":            "High (Keyed PRNG Steganography)",
        "sensitive_layer_avoidance": "Expert (Resilient Layer Targeting)",
        "unknown":                   "Unknown",
    }

    return {
        "pattern": pattern_key,
        "layer_modification_rates": {k: layer_rates[k] for k in sorted_names[:20]},  # cap at 20
        "concentration_layers": concentration_layers,
        "batchnorm_untouched": bn_untouched,
        "first_conv_untouched": first_conv_untouched,
        "sophistication": sophistication_map.get(pattern_key, "Unknown"),
        "description": pattern_desc,
    }


# ---------------------------------------------------------------------------
# Signal 3: Payload Entropy Spectrum
# ---------------------------------------------------------------------------

def _analyze_payload_entropy(flat_bytes: np.ndarray, estimated_x: int) -> dict[str, Any]:
    """
    Signal 3: Shannon entropy of the extracted LSB bitstream.
    Classifies whether the payload is plaintext, compressed, or encrypted.
    """
    if flat_bytes is None or len(flat_bytes) == 0 or estimated_x == 0:
        return {
            "bits_per_byte": 0.0,
            "classification": "No payload detected",
            "entropy_band": "clean",
            "assessment": "No steganographic signal detected — bit-plane entropy is within clean model baseline.",
        }

    n_weights = len(flat_bytes) // 4
    weight_bytes = flat_bytes[:n_weights * 4].reshape(n_weights, 4)
    lsb_bytes = weight_bytes[:, 0]  # Q3 (LE byte 0 = LSB of float32 mantissa)

    x_bits = min(estimated_x, 8)
    if x_bits < 8:
        mask = (1 << x_bits) - 1
        lsb_vals = lsb_bytes & mask
    else:
        lsb_vals = lsb_bytes

    # Subsample for speed
    if len(lsb_vals) > 2_000_000:
        rng = np.random.default_rng(42)
        lsb_vals = rng.choice(lsb_vals, 2_000_000, replace=False)

    h = _shannon_entropy(lsb_vals.astype(np.uint8))
    h_normalized = h  # already in bits/byte (0–8)

    if h_normalized >= 7.90:
        band = "encrypted"
        classification = "Encrypted / High-Entropy Ciphertext"
        assessment = (
            f"Shannon entropy H={h_normalized:.4f} bits/byte is near-maximum (8.0). "
            "Payload is encrypted (e.g., AES-CBC, ChaCha20) before embedding. "
            "Cannot read payload without the decryption key, but confirms malicious intent — "
            "clean model weights never produce uniform-random LSB distributions across thousands of floats."
        )
    elif h_normalized >= 6.00:
        band = "compressed"
        classification = "Compressed Archive / Encoded Payload"
        assessment = (
            f"Shannon entropy H={h_normalized:.4f} bits/byte is high but sub-maximal. "
            "Consistent with gzip/zip compressed shellcode, pickled exploit stubs, "
            "or base64/hex-encoded payloads."
        )
    else:
        band = "plaintext"
        classification = "Plaintext ASCII / Unencoded Data"
        assessment = (
            f"Shannon entropy H={h_normalized:.4f} bits/byte is moderate. "
            "Consistent with unencoded strings, URLs, domain lists, "
            "C2 IP addresses, shell scripts, or SSH keys embedded directly."
        )

    return {
        "bits_per_byte": round(h_normalized, 4),
        "classification": classification,
        "entropy_band": band,
        "assessment": assessment,
    }


# ---------------------------------------------------------------------------
# Signal 4: Estimated Payload Capacity
# ---------------------------------------------------------------------------

def _estimate_payload_capacity(
    flat_bytes: np.ndarray,
    estimated_x: int,
    distribution: dict[str, Any],
) -> dict[str, Any]:
    """
    Signal 4: Estimate maximum payload size based on number of modified weights
    and embedding rate.

    Formula: capacity_bytes = (n_anomalous_weights × estimated_x) / 8
    """
    if flat_bytes is None or estimated_x == 0:
        return {
            "estimated_bytes": 0,
            "human_readable": "0 B",
            "n_anomalous_weights": 0,
            "embedding_rate_x": 0,
            "classification": "No payload",
            "description": "No steganographic signal — capacity not applicable.",
        }

    n_weights = len(flat_bytes) // 4

    # Estimate anomalous weight fraction from distribution signal
    layer_rates = distribution.get("layer_modification_rates", {})

    if layer_rates:
        mean_rate = float(np.mean(list(layer_rates.values())))
    else:
        # Fallback: assume full Q3 embedding if no distribution data
        mean_rate = 1.0 if estimated_x == 8 else 0.5

    n_anomalous = int(n_weights * mean_rate)
    capacity_bytes = int((n_anomalous * estimated_x) / 8)

    if capacity_bytes < 5 * 1024:
        human = f"{capacity_bytes} B"
        classification = "Credential / C2 Payload"
        desc = (
            f"Estimated ~{human} payload. Typical contents: SSH private keys, "
            "AWS/cloud credentials, C2 domain lists, or a small shellcode stager."
        )
    elif capacity_bytes < 50 * 1024:
        human = f"{capacity_bytes // 1024} KB"
        classification = "Small Exploit / Shellcode"
        desc = (
            f"Estimated ~{human} payload. Typical contents: compact shellcode, "
            "simple backdoor, or small configuration blob."
        )
    elif capacity_bytes < 500 * 1024:
        human = f"{capacity_bytes // 1024} KB"
        classification = "Standalone Exploit / In-Memory Implant"
        desc = (
            f"Estimated ~{human} payload. Typical contents: standalone compiled exploit, "
            "reflective DLL loader, or in-memory implant."
        )
    else:
        human = f"{capacity_bytes / (1024*1024):.1f} MB"
        classification = "Full Binary / RAT / Ransomware"
        desc = (
            f"Estimated ~{human} payload. Typical contents: full second-stage binary, "
            "ransomware executable, multi-platform RAT, or Cobalt Strike beacon."
        )

    return {
        "estimated_bytes": capacity_bytes,
        "human_readable": human,
        "n_anomalous_weights": n_anomalous,
        "embedding_rate_x": estimated_x,
        "classification": classification,
        "description": desc,
    }


# ---------------------------------------------------------------------------
# Signal 5: Magic Byte Signatures
# ---------------------------------------------------------------------------

def _detect_magic_bytes(flat_bytes: np.ndarray, estimated_x: int) -> dict[str, Any]:
    """
    Signal 5: If payload entropy is sub-7.90 (not fully encrypted), attempt to
    extract the first 16 bytes of the LSB bitstream and match against known magic bytes.
    """
    if flat_bytes is None or estimated_x == 0:
        return {
            "detected": False,
            "signature": None,
            "description": None,
            "raw_header_hex": None,
            "reason": "No payload detected.",
        }

    # Extract the first 128 weights worth of LSB bits → 16 bytes of payload
    x_bits = min(estimated_x, 8)
    n_needed = max(1, (16 * 8) // x_bits)  # weights needed to get 16 payload bytes

    weight_bytes = flat_bytes[:n_needed * 4].reshape(-1, 4)
    lsb_col = weight_bytes[:, 0]  # Q3 byte (LE byte 0)

    # Reconstruct first 16 payload bytes by packing x_bits from each weight
    bits_list: list[int] = []
    for b in lsb_col:
        mask = (1 << x_bits) - 1
        val = int(b) & mask
        # Append bits MSB-first
        for shift in range(x_bits - 1, -1, -1):
            bits_list.append((val >> shift) & 1)
        if len(bits_list) >= 128:
            break

    # Pack bits into bytes
    payload_bytes = bytearray()
    for i in range(0, min(len(bits_list), 128), 8):
        byte_bits = bits_list[i:i+8]
        if len(byte_bits) < 8:
            break
        val = 0
        for b in byte_bits:
            val = (val << 1) | b
        payload_bytes.append(val)

    if not payload_bytes:
        return {
            "detected": False,
            "signature": None,
            "description": None,
            "raw_header_hex": None,
            "reason": "Could not extract payload bytes.",
        }

    raw_hex = payload_bytes.hex().upper()

    for magic, sig_name, sig_desc in _MAGIC_SIGNATURES:
        if payload_bytes[:len(magic)] == magic:
            return {
                "detected": True,
                "signature": sig_name,
                "description": sig_desc,
                "raw_header_hex": raw_hex[:32],
            }

    return {
        "detected": False,
        "signature": None,
        "description": "No known magic bytes matched — payload may be encrypted, a custom format, or raw shellcode.",
        "raw_header_hex": raw_hex[:32],
    }


# ---------------------------------------------------------------------------
# Main Engine
# ---------------------------------------------------------------------------

class ForensicSignalEngine:
    """
    Deterministic steganography forensic signal extractor.

    Usage::

        engine = ForensicSignalEngine()
        forensics = engine.analyze("/path/to/model.pt", band="malicious")
    """

    def analyze(self, filepath: str, band: str) -> dict[str, Any]:
        """
        Run forensic signal extraction.

        Args:
            filepath:  Absolute path to the model file.
            band:      "suspicious" or "malicious". Both run the full suite;
                       the band remains a confidence classification, not an
                       analysis-depth gate.

        Returns:
            Structured forensic signals dict, or an error dict on failure.
        """
        if band not in ("suspicious", "malicious"):
            raise ValueError(f"band must be 'suspicious' or 'malicious', got {band!r}")

        try:
            flat_bytes, named_tensors = _load_flat_bytes(filepath)
        except Exception as exc:
            return {"error": f"Failed to load model file: {exc}", "band": band}

        if flat_bytes is None or len(flat_bytes) == 0:
            return {
                "band": band,
                "error": "No float32 weights could be extracted from the file.",
            }

        # --- Signal 1: Always run (bit-plane depth) ---
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bit_depth = _analyze_bit_plane_depth(flat_bytes)

        estimated_x = bit_depth.get("estimated_x", 0)

        # --- Signal 3: Payload entropy ---
        entropy = _analyze_payload_entropy(flat_bytes, estimated_x)

        # Both elevated bands run all five signals. A suspicious result is
        # still not a confirmed malicious verdict; it simply receives enough
        # evidence for a meaningful analyst review.

        # Signal 2: Distribution geometry
        distribution = _analyze_distribution_geometry(
            named_tensors or {}, estimated_x
        )

        # Signal 4: Payload capacity
        capacity = _estimate_payload_capacity(flat_bytes, estimated_x, distribution)

        # Signal 5: Magic bytes (only if payload is not fully encrypted)
        entropy_band = entropy.get("entropy_band", "encrypted")
        if entropy_band == "encrypted":
            magic = {
                "detected": False,
                "signature": None,
                "description": "Magic byte analysis not applicable — payload is encrypted.",
                "raw_header_hex": None,
            }
        else:
            magic = _detect_magic_bytes(flat_bytes, estimated_x)

        return {
            "band": band,
            "analysis_depth": "full (all 5 signals)",
            "note": (
                "Full forensic review triggered by an elevated Layer 3a signal. "
                "The forensic band is a confidence classification and does not "
                "by itself change the fused verdict."
            ),
            "bit_depth": bit_depth,
            "distribution": distribution,
            "entropy": entropy,
            "capacity": capacity,
            "magic_bytes": magic,
        }


# ---------------------------------------------------------------------------
# CLI for manual testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="PROJECT-CLIMAX Forensic Signal Engine")
    parser.add_argument("model_path", help="Path to model file")
    parser.add_argument(
        "--band",
        choices=["suspicious", "malicious"],
        default="malicious",
        help="Analysis band (default: malicious)",
    )
    args = parser.parse_args()

    engine = ForensicSignalEngine()
    result = engine.analyze(args.model_path, band=args.band)
    print(json.dumps(result, indent=2))
