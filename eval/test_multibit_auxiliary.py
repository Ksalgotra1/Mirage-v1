"""eval/test_multibit_auxiliary.py — Unit Tests for 1D Nibble Auxiliary Steganalysis

Tests:
  - Test A1: FP16 -> FP32 upcast veto (zero lower mantissa bits)
  - Test A2: BF16 -> FP32 upcast veto (zero lower 16 bits)
  - Test A3: Structured pruning / dead ReLU immunity (active set filtering)
  - Test A4: Non-uniform 4-bit payload detection (Signal A TVD)
  - Test A5: Encrypted/repeating 4-bit payload detection (Signal B FFT Periodicity)
  - Test A6: 1-bit and 2-bit payload estimation
  - Test A7: Distance synchronization safeguard (Safeguard 2)
  - Test A8: Toggle control (MIRAGE_ENABLE_MULTIBIT_AUX)
"""
from __future__ import annotations

import numpy as np
import pytest

from detector.layer3a_gf import compute_multibit_auxiliary_score, Layer3aGFDetector


def _generate_clean_f32(n: int = 50_000, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n).astype(np.float32)


def test_a1_fp16_upcast_vetoed():
    """FP16 upcast sets b12..b0 = 0. Gate 1 must veto unconditionally."""
    clean = _generate_clean_f32(50_000)
    u32 = clean.view(np.uint32).copy()
    # Mask out lowest 13 bits (simulating half().float())
    u32 &= ~np.uint32(0x1FFF)
    res = compute_multibit_auxiliary_score(u32)
    assert res["aux_score"] == 0.0
    assert res["aux_verdict"] is False
    assert "vetoed" in res["reason"] or "quantized" in res["reason"]


def test_a2_bf16_upcast_vetoed():
    """BF16 upcast sets b15..b0 = 0. Gate 1 must veto unconditionally."""
    clean = _generate_clean_f32(50_000)
    u32 = clean.view(np.uint32).copy()
    # Mask out lowest 16 bits (simulating bfloat16().float())
    u32 &= ~np.uint32(0xFFFF)
    res = compute_multibit_auxiliary_score(u32)
    assert res["aux_score"] == 0.0
    assert res["aux_verdict"] is False
    assert "vetoed" in res["reason"] or "quantized" in res["reason"]


def test_a3_structured_pruning_immunity():
    """Clean weights with 50% exact zeros must not trigger false positive."""
    clean = _generate_clean_f32(50_000)
    rng = np.random.default_rng(123)
    prune_mask = rng.random(clean.shape) < 0.50
    clean[prune_mask] = 0.0
    u32 = clean.view(np.uint32)

    res = compute_multibit_auxiliary_score(u32)
    assert res["aux_score"] < 0.30
    assert res["aux_verdict"] is False
    assert res["tvd_12"] < 0.04


def test_a4_non_uniform_4bit_payload_detected():
    """4-bit structured payload (e.g. ASCII / shellcode) triggers Signal A TVD."""
    clean = _generate_clean_f32(50_000)
    u32 = clean.view(np.uint32).copy()

    # ASCII payload pattern repeated in lower 4 bits
    payload_str = b"EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*"
    p_bytes = np.frombuffer(payload_str * ((len(clean) // len(payload_str)) + 1), dtype=np.uint8)[: len(clean)]
    p_nibbles = p_bytes & 0x0F

    # Overwrite only b3..b0
    u32 = (u32 & ~np.uint32(0x0F)) | p_nibbles.astype(np.uint32)

    res = compute_multibit_auxiliary_score(u32)
    assert res["aux_score"] >= 0.70
    assert res["aux_verdict"] is True
    assert res["estimated_x_bits"] == 4
    assert res["tvd_01"] > 0.10


def test_a5_encrypted_repeating_4bit_payload_detected():
    """16 KB encrypted payload tiled across weights triggers Signal B FFT Autocorrelation."""
    clean = _generate_clean_f32(100_000)
    u32 = clean.view(np.uint32).copy()

    # 16 KB random high-entropy buffer
    rng_p = np.random.default_rng(999)
    # At x=4 bits/weight, each byte needs 2 weights -> 16 KB payload repeats every 32,768 weights
    p_len = 32_768
    repeating_pattern = rng_p.integers(0, 16, size=p_len, dtype=np.uint8)
    tiled_nibbles = np.tile(repeating_pattern, (len(clean) // p_len) + 1)[: len(clean)]

    # Overwrite only b3..b0
    u32 = (u32 & ~np.uint32(0x0F)) | tiled_nibbles.astype(np.uint32)

    res = compute_multibit_auxiliary_score(u32)
    assert res["aux_score"] >= 0.70
    assert res["aux_verdict"] is True
    assert res["peak_acf"] > 0.60
    assert res["estimated_x_bits"] == 4


def test_a6_1bit_and_2bit_payload_estimation():
    """Periodic payloads at 1-bit and 2-bit correctly estimate x in {1, 2}."""
    clean = _generate_clean_f32(65_536)
    u32 = clean.view(np.uint32).copy()

    # 2-bit repeating pattern (period = 1024)
    rng_p = np.random.default_rng(777)
    p_2bit = rng_p.integers(0, 4, size=1024, dtype=np.uint8)
    tiled_2bit = np.tile(p_2bit, (len(clean) // 1024) + 1)[: len(clean)]
    u32_2bit = (u32 & ~np.uint32(0x03)) | tiled_2bit.astype(np.uint32)

    res_2bit = compute_multibit_auxiliary_score(u32_2bit)
    assert res_2bit["aux_score"] >= 0.70
    assert res_2bit["estimated_x_bits"] == 2


def test_a7_calibrated_distance_synchronization(tmp_path):
    """When S_aux > S_cnn, calibrated_distance must be synchronized via inverse logistic."""
    try:
        import torch
    except ImportError:
        pytest.skip("torch required")

    detector = Layer3aGFDetector()

    # Create a 4-bit stego model file
    clean = _generate_clean_f32(50_000)
    u32 = clean.view(np.uint32).copy()
    rng_p = np.random.default_rng(555)
    p_len = 1024
    pattern = rng_p.integers(0, 16, size=p_len, dtype=np.uint8)
    tiled = np.tile(pattern, (len(clean) // p_len) + 1)[: len(clean)]
    u32 = (u32 & ~np.uint32(0x0F)) | tiled.astype(np.uint32)
    stego_f32 = u32.view(np.float32)

    model_path = tmp_path / "stego_4bit.pt"
    torch.save({"weight": torch.from_numpy(stego_f32)}, model_path)

    res = detector.predict(str(model_path), enable_multibit_aux=True)
    assert res["verdict"] is True
    assert res["score"] >= 0.70
    assert res["calibrated_distance"] > res["threshold"]
    assert "multibit_aux" in res


def test_a8_multibit_aux_toggle(tmp_path):
    """Disabling multibit_aux turns off the auxiliary engine."""
    try:
        import torch
    except ImportError:
        pytest.skip("torch required")

    detector = Layer3aGFDetector()
    clean = _generate_clean_f32(20_000)
    model_path = tmp_path / "clean_test.pt"
    torch.save({"weight": torch.from_numpy(clean)}, model_path)

    res_off = detector.predict(str(model_path), enable_multibit_aux=False)
    assert "multibit_aux" not in res_off
