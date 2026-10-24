from __future__ import annotations

import numpy as np

from detector.layer3a_gf import (
    Layer3aGFDetector,
    _calibrated_score,
    _cl2_cosine_margin,
    _forensic_band,
    _load_bundle,
    _size_aware_adjustment,
)


def test_installed_bundle_loads_and_has_calibrated_threshold():
    bundle = _load_bundle()
    threshold = bundle["calibration"]["threshold"]
    assert isinstance(threshold, float)
    assert -1.5 <= threshold <= 1.5
    assert bundle["mean_train"] is not None


def test_cl2_margin_uses_centering_and_cosine_prototypes():
    margin = _cl2_cosine_margin(
        np.array([3.0, 1.0], dtype=np.float32),
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
        np.array([1.0, 1.0], dtype=np.float32),
    )
    assert margin < -0.99


def test_score_is_centered_on_threshold_and_bands_are_monotonic():
    threshold = 0.07223621010780334
    assert _calibrated_score(threshold, threshold) == 0.5
    calibration = {"forensic_bands": {"suspicious_band_min": 0.4, "malicious_band_min": 0.7}}
    assert _forensic_band(0.39, calibration) == "clean"
    assert _forensic_band(0.40, calibration) == "suspicious"
    assert _forensic_band(0.70, calibration) == "malicious"


def test_size_correction_is_opt_in_and_versioned():
    assert _size_aware_adjustment({}, 1_000_000) == 0.0
    policy = {"size_aware_correction": {
        "enabled": True, "version": "log10_parameter_count_v1",
        "slope": 0.02, "reference_log10_parameter_count": 6,
    }}
    assert _size_aware_adjustment(policy, 10_000_000) == 0.02


def test_unsupported_format_returns_stable_clean_response():
    result = Layer3aGFDetector().predict("checkpoint.unknown")
    assert result["verdict"] is False
    assert result["forensic_band"] == "clean"
    assert result["coverage_note"]
