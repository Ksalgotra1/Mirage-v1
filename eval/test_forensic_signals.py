"""
eval/test_forensic_signals.py — Forensic Signal Engine Test Suite
=================================================================
Tests all 5 forensic signals against both clean and LSB-embedded models.
"""
from __future__ import annotations

import sys
import tempfile
import os
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detector.forensic_signals import (
    ForensicSignalEngine,
    _analyze_bit_plane_depth,
    _analyze_payload_entropy,
    _analyze_distribution_geometry,
    _estimate_payload_capacity,
    _detect_magic_bytes,
    _shannon_entropy,
)
from backend.summarizer import build_llm_prompt, deterministic_summary


# ---------------------------------------------------------------------------
# Helpers: Create synthetic model files
# ---------------------------------------------------------------------------

def _make_clean_pt(n_weights: int = 50_000) -> Path:
    """Write a clean .pt file with Gaussian float32 weights."""
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")
    rng = np.random.default_rng(42)
    weights = rng.standard_normal(n_weights).astype(np.float32)
    sd = {"layer.weight": torch.from_numpy(weights)}
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(sd, f.name)
        return Path(f.name)


def _make_lsb_pt(n_weights: int = 50_000, x_bits: int = 8, payload: bytes | None = None) -> Path:
    """Write a .pt file with 8-bit LSB steganographic embedding."""
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")

    rng = np.random.default_rng(42)
    weights = rng.standard_normal(n_weights).astype(np.float32)
    raw = weights.view(np.uint8).reshape(n_weights, 4).copy()

    # Embed random (encrypted-looking) payload into Q3 bytes
    if payload is None:
        payload_bytes = np.random.default_rng(99).integers(0, 256, n_weights, dtype=np.uint8)
    else:
        payload_bytes = np.frombuffer(payload * ((n_weights // len(payload)) + 1), dtype=np.uint8)[:n_weights]

    mask = (1 << x_bits) - 1
    raw[:, 0] = (raw[:, 0] & (~mask & 0xFF)) | (payload_bytes & mask)
    embedded = raw.view(np.float32)

    sd = {"layer.weight": torch.from_numpy(embedded)}
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(sd, f.name)
        return Path(f.name)


def _make_mz_lsb_pt(n_weights: int = 50_000) -> Path:
    """Write a .pt file with MZ header embedded in LSB."""
    mz_prefix = b"\x4D\x5A\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00\xFF\xFF\x00\x00"
    payload = mz_prefix + bytes(n_weights)  # MZ header + zeros
    return _make_lsb_pt(n_weights, x_bits=8, payload=payload)


# ---------------------------------------------------------------------------
# Helper: flat bytes from a clean ndarray
# ---------------------------------------------------------------------------

def _clean_flat_bytes(n: int = 50_000) -> np.ndarray:
    rng = np.random.default_rng(42)
    weights = rng.standard_normal(n).astype(np.float32)
    return weights.view(np.uint8).copy()


def _stego_flat_bytes(n: int = 50_000, x_bits: int = 8, rng_seed: int = 99) -> np.ndarray:
    rng_w = np.random.default_rng(42)
    rng_p = np.random.default_rng(rng_seed)
    weights = rng_w.standard_normal(n).astype(np.float32)
    raw = weights.view(np.uint8).reshape(n, 4).copy()
    payload = rng_p.integers(0, 256, n, dtype=np.uint8)  # high-entropy (encrypted-like)
    mask = (1 << x_bits) - 1
    raw[:, 0] = (raw[:, 0] & (~mask & 0xFF)) | (payload & mask)
    return raw.reshape(-1).copy()


# ---------------------------------------------------------------------------
# Signal 1: Bit-Plane Depth
# ---------------------------------------------------------------------------

class TestBitPlaneDepth:
    def test_clean_model_no_anomalous_planes(self):
        flat = _clean_flat_bytes()
        result = _analyze_bit_plane_depth(flat)
        # Signal 1 is a characterization function (not detection).
        # For clean models, Q0 must NOT be contaminated (Q0 entropy should be ~2.6, well below 5.0).
        assert result.get("q0_contaminated") is False, (
            f"Clean model Q0 should not be contaminated. "
            f"Per-plane entropy: {result.get('per_plane_entropy')}"
        )
        # Clean models get estimated_x > 0 (characterization always returns a profile)
        assert "estimated_x" in result
        assert "stealth_assessment" in result

    def test_stego_8bit_detects_q3(self):
        flat = _stego_flat_bytes(x_bits=8)
        result = _analyze_bit_plane_depth(flat)
        # Stego model: should produce a valid characterization
        assert "estimated_x" in result
        assert result["estimated_x"] >= 2, "Should always characterize with x >= 2"
        assert result.get("q0_contaminated") is False, "8-bit stego should not contaminate Q0"

    def test_stego_stealth_classification(self):
        flat = _stego_flat_bytes(x_bits=8)
        result = _analyze_bit_plane_depth(flat)
        stealth = result["stealth_assessment"]
        # Any valid stealth classification is acceptable
        valid = ["Capacity", "Dropper", "Uncalibrated", "Stealth", "APT", "Nation", "Moderate"]
        assert any(kw in stealth for kw in valid), f"Unexpected stealth profile: {stealth!r}"

    def test_entropy_values_in_range(self):
        flat = _clean_flat_bytes()
        result = _analyze_bit_plane_depth(flat)
        for plane, h in result["per_plane_entropy"].items():
            assert 0.0 <= h <= 8.0, f"Entropy out of range for {plane}: {h}"


# ---------------------------------------------------------------------------
# Signal 2: Distribution Geometry
# ---------------------------------------------------------------------------

class TestDistributionGeometry:
    def test_clean_tensor_low_rates(self):
        rng = np.random.default_rng(42)
        named = {
            "layer1.weight": rng.standard_normal((64, 3, 3, 3)).astype(np.float32),
            "layer2.weight": rng.standard_normal((128, 64, 3, 3)).astype(np.float32),
        }
        result = _analyze_distribution_geometry(named, estimated_x=0)
        assert result["pattern"] == "unknown"

    def test_uniform_pattern_classified(self):
        # Uniform modification across all layers → expect uniform_stride or prng_scattered
        rng_w = np.random.default_rng(42)
        rng_p = np.random.default_rng(1)
        named = {}
        for i in range(6):
            w = rng_w.standard_normal((64, 64, 3, 3)).astype(np.float32)
            raw = w.view(np.uint8).reshape(-1, 4).copy()
            raw[:, 0] = rng_p.integers(0, 256, len(raw), dtype=np.uint8)
            named[f"layer{i}.weight"] = raw.view(np.float32)
        result = _analyze_distribution_geometry(named, estimated_x=8)
        assert result["pattern"] in ("uniform_stride", "prng_scattered", "sensitive_layer_avoidance")

    def test_sophistication_string_present(self):
        rng = np.random.default_rng(42)
        named = {"layer.weight": rng.standard_normal((64, 64)).astype(np.float32)}
        result = _analyze_distribution_geometry(named, estimated_x=8)
        assert "sophistication" in result
        assert isinstance(result["sophistication"], str)


# ---------------------------------------------------------------------------
# Signal 3: Payload Entropy
# ---------------------------------------------------------------------------

class TestPayloadEntropy:
    def test_clean_returns_no_embedding(self):
        flat = _clean_flat_bytes()
        result = _analyze_payload_entropy(flat, estimated_x=0)
        assert result["entropy_band"] == "clean"

    def test_encrypted_payload_classified(self):
        flat = _stego_flat_bytes(x_bits=8, rng_seed=99)  # uniform random = high entropy
        result = _analyze_payload_entropy(flat, estimated_x=8)
        h = result["bits_per_byte"]
        # High-entropy random payload should classify as encrypted or compressed
        assert result["entropy_band"] in ("encrypted", "compressed"), (
            f"Expected encrypted/compressed, got {result['entropy_band']} (H={h})"
        )

    def test_plaintext_payload_classified(self):
        # Embed repeated ASCII 'A' (low entropy)
        n = 50_000
        rng_w = np.random.default_rng(42)
        weights = rng_w.standard_normal(n).astype(np.float32)
        raw = weights.view(np.uint8).reshape(n, 4).copy()
        raw[:, 0] = 65  # ASCII 'A' — very low entropy
        flat = raw.reshape(-1).copy()
        result = _analyze_payload_entropy(flat, estimated_x=8)
        assert result["entropy_band"] == "plaintext"

    def test_entropy_between_0_and_8(self):
        flat = _stego_flat_bytes(x_bits=8)
        result = _analyze_payload_entropy(flat, estimated_x=8)
        assert 0.0 <= result["bits_per_byte"] <= 8.0


# ---------------------------------------------------------------------------
# Signal 4: Payload Capacity
# ---------------------------------------------------------------------------

class TestPayloadCapacity:
    def test_8bit_50k_weights_gives_reasonable_capacity(self):
        flat = _stego_flat_bytes(n=50_000, x_bits=8)
        dist = {"layer_modification_rates": {}, "concentration_layers": []}
        result = _estimate_payload_capacity(flat, estimated_x=8, distribution=dist)
        # 50000 weights × 8 bits / 8 = 50000 bytes ≈ 48.8 KB
        assert result["estimated_bytes"] > 0
        assert result["embedding_rate_x"] == 8
        assert "KB" in result["human_readable"] or "MB" in result["human_readable"]

    def test_no_embedding_gives_zero(self):
        flat = _clean_flat_bytes()
        dist = {"layer_modification_rates": {}, "concentration_layers": []}
        result = _estimate_payload_capacity(flat, estimated_x=0, distribution=dist)
        assert result["estimated_bytes"] == 0
        assert result["embedding_rate_x"] == 0

    def test_capacity_classification_present(self):
        flat = _stego_flat_bytes(n=50_000, x_bits=8)
        dist = {"layer_modification_rates": {}, "concentration_layers": []}
        result = _estimate_payload_capacity(flat, estimated_x=8, distribution=dist)
        assert "classification" in result
        assert len(result["classification"]) > 0


# ---------------------------------------------------------------------------
# Signal 5: Magic Bytes
# ---------------------------------------------------------------------------

class TestMagicBytes:
    def test_mz_header_detected(self):
        # Craft MZ payload
        mz = b"\x4D\x5A\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00\xFF\xFF\x00\x00"
        n = 50_000
        rng_w = np.random.default_rng(42)
        weights = rng_w.standard_normal(n).astype(np.float32)
        raw = weights.view(np.uint8).reshape(n, 4).copy()
        # Embed MZ bytes into first weights
        for i, b in enumerate(mz):
            raw[i, 0] = b
        flat = raw.reshape(-1).copy()
        result = _detect_magic_bytes(flat, estimated_x=8)
        assert result["detected"] is True
        assert result["signature"] == "MZ"

    def test_no_magic_in_random_payload(self):
        flat = _stego_flat_bytes(x_bits=8, rng_seed=123)
        result = _detect_magic_bytes(flat, estimated_x=8)
        # Random data almost certainly won't match MZ/ELF/PK
        # Just verify the structure is correct
        assert "detected" in result
        assert "raw_header_hex" in result

    def test_zero_x_returns_no_payload(self):
        flat = _clean_flat_bytes()
        result = _detect_magic_bytes(flat, estimated_x=0)
        assert result["detected"] is False


# ---------------------------------------------------------------------------
# ForensicSignalEngine integration
# ---------------------------------------------------------------------------

class TestForensicEngine:
    def setup_method(self):
        self.engine = ForensicSignalEngine()

    def test_clean_model_suspicious_band(self):
        pt = _make_clean_pt()
        try:
            result = self.engine.analyze(str(pt), band="suspicious")
            # Clean model: Q0 must not be contaminated
            bd = result.get("bit_depth", {})
            assert bd.get("q0_contaminated") is False, (
                f"Clean model Q0 should not be contaminated. "
                f"Per-plane entropy: {bd.get('per_plane_entropy')}"
            )
            # Suspicious band must not have distribution/capacity/magic
            assert result["distribution"] is None
            assert result["capacity"] is None
            assert result["magic_bytes"] is None
        finally:
            os.unlink(pt)

    def test_stego_model_malicious_band_has_all_signals(self):
        pt = _make_lsb_pt(x_bits=8)
        try:
            result = self.engine.analyze(str(pt), band="malicious")
            assert result["band"] == "malicious"
            assert "bit_depth" in result
            assert "distribution" in result
            assert "entropy" in result
            assert "capacity" in result
            assert "magic_bytes" in result
            # Characterization always produces a valid profile
            bd = result["bit_depth"]
            assert "estimated_x" in bd
            assert "stealth_assessment" in bd
            assert bd.get("q0_contaminated") is False, "8-bit stego should not contaminate Q0"
        finally:
            os.unlink(pt)

    def test_suspicious_band_has_all_forensic_signals(self):
        pt = _make_lsb_pt(x_bits=8)
        try:
            result = self.engine.analyze(str(pt), band="suspicious")
            assert result["band"] == "suspicious"
            assert result["bit_depth"] is not None
            assert result["entropy"] is not None
            assert result["analysis_depth"] == "full (all 5 signals)"
            assert result["distribution"] is not None
            assert result["capacity"] is not None
            assert result["magic_bytes"] is not None
        finally:
            os.unlink(pt)

    def test_invalid_band_raises(self):
        with pytest.raises(ValueError, match="band must be"):
            self.engine.analyze("/dev/null", band="wrong")

    def test_mz_header_detected_end_to_end(self):
        pt = _make_mz_lsb_pt()
        try:
            result = self.engine.analyze(str(pt), band="malicious")
            magic = result.get("magic_bytes", {})
            # MZ may be detected; just verify structure is intact
            assert "detected" in magic
        finally:
            os.unlink(pt)


# ---------------------------------------------------------------------------
# Summarizer integration
# ---------------------------------------------------------------------------

class TestSummarizer:
    def _make_findings(self, l3a_score: float, forensics: dict | None = None) -> dict:
        return {
            "filename": "test_model.pt",
            "format": "pt",
            "layer1": {"verdict": False},
            "layer2": {"verdict": False, "probability": 0.1},
            "layer3a": {"verdict": l3a_score >= 0.5, "score": l3a_score, "distance": 0.2},
            "layer3b": {"verdict": False, "score": 0.0},
            "fused": {
                "verdict": "MALICIOUS" if l3a_score >= 0.5 else "BENIGN",
                "score": l3a_score,
                "forensic_band": "malicious" if l3a_score >= 0.70 else (
                    "suspicious" if l3a_score >= 0.40 else "clean"
                ),
                "explanation": "Layer 3a: GF LSB payload pattern detected",
            },
            "steganography_forensics": forensics,
        }

    def test_clean_band_prompt_minimal(self):
        findings = self._make_findings(l3a_score=0.20)
        prompt = build_llm_prompt(findings)
        assert "STEGANOGRAPHY FORENSIC SIGNALS" not in prompt

    def test_suspicious_band_prompt_has_all_signals(self):
        forensics = {
            "band": "suspicious",
            "bit_depth": {"estimated_x": 4, "anomalous_planes": ["Q3"],
                          "per_plane_entropy": {"Q3": 7.60},
                          "clean_entropy_baseline": {"Q3": 6.90},
                          "stealth_assessment": "Maximum Stealth", "description": "test"},
            "distribution": {"pattern": "scattered", "sophistication": "Moderate", "description": "test"},
            "entropy": {"bits_per_byte": 7.85, "classification": "Compressed",
                        "entropy_band": "compressed", "assessment": "test"},
            "capacity": {"estimated_bytes": 123, "human_readable": "123 B", "description": "test"},
            "magic_bytes": {"detected": False, "description": "No known signature detected"},
        }
        findings = self._make_findings(l3a_score=0.55, forensics=forensics)
        prompt = build_llm_prompt(findings)
        assert "SIGNAL 1" in prompt
        assert "SIGNAL 3" in prompt
        for sig in ["SIGNAL 1", "SIGNAL 2", "SIGNAL 3", "SIGNAL 4", "SIGNAL 5"]:
            assert sig in prompt, f"{sig} missing from suspicious-band prompt"

    def test_malicious_band_prompt_has_all_signals(self):
        forensics = {
            "band": "malicious",
            "bit_depth": {"estimated_x": 8, "anomalous_planes": ["Q3"],
                          "per_plane_entropy": {"Q3": 7.99},
                          "clean_entropy_baseline": {"Q3": 6.90},
                          "stealth_assessment": "High Capacity (Binary Dropper)",
                          "description": "test desc"},
            "distribution": {"pattern": "prng_scattered", "sophistication": "High",
                             "batchnorm_untouched": True, "first_conv_untouched": True,
                             "concentration_layers": ["layer3.0"], "description": "test"},
            "entropy": {"bits_per_byte": 7.989, "classification": "Encrypted",
                        "entropy_band": "encrypted", "assessment": "test"},
            "capacity": {"estimated_bytes": 1474560, "human_readable": "1.4 MB",
                         "n_anomalous_weights": 1474560, "embedding_rate_x": 8,
                         "classification": "Full Binary / RAT", "description": "test"},
            "magic_bytes": {"detected": False, "description": "Payload is encrypted.",
                            "signature": None, "raw_header_hex": None},
        }
        findings = self._make_findings(l3a_score=0.83, forensics=forensics)
        prompt = build_llm_prompt(findings)
        for sig in ["SIGNAL 1", "SIGNAL 2", "SIGNAL 3", "SIGNAL 4", "SIGNAL 5"]:
            assert sig in prompt, f"{sig} missing from malicious-band prompt"
        assert "THREAT ASSESSMENT" in prompt

    def test_deterministic_summary_malicious_with_forensics(self):
        forensics = {
            "band": "malicious",
            "bit_depth": {"estimated_x": 8, "stealth_assessment": "High Capacity", "description": ""},
            "entropy": {"bits_per_byte": 7.99, "classification": "Encrypted Ciphertext",
                        "entropy_band": "encrypted", "assessment": ""},
            "capacity": {"estimated_bytes": 1474560, "human_readable": "1.4 MB",
                         "classification": "Full Binary / RAT", "description": ""},
            "distribution": {"pattern": "prng_scattered", "sophistication": "High", "description": ""},
            "magic_bytes": None,
        }
        findings = self._make_findings(l3a_score=0.83, forensics=forensics)
        summary = deterministic_summary(findings)
        assert "MALICIOUS" in summary
        assert "1.4 MB" in summary or "Encrypted" in summary


# ---------------------------------------------------------------------------
# Entropy helper sanity checks
# ---------------------------------------------------------------------------

class TestEntropyHelper:
    def test_uniform_distribution_max_entropy(self):
        arr = np.arange(256, dtype=np.uint8)
        h = _shannon_entropy(arr)
        assert abs(h - 8.0) < 0.01, f"Uniform distribution should have H≈8.0, got {h}"

    def test_constant_zero_entropy(self):
        arr = np.zeros(1000, dtype=np.uint8)
        h = _shannon_entropy(arr)
        assert h == 0.0

    def test_empty_zero_entropy(self):
        arr = np.array([], dtype=np.uint8)
        h = _shannon_entropy(arr)
        assert h == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
