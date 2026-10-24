from backend.fusion import FusionEngine


def _clean_layers():
    return {"verdict": False}, {"verdict": False, "probability": 0.0}, {"verdict": False, "score": 0.0}, {"verdict": False, "score": 0.0}


def test_safetensors_uses_weight_space_layers_only():
    l1, l2, l3a, l3b = _clean_layers()
    result = FusionEngine().fuse("safetensors", l1, l2, l3a, l3b)
    assert result["weights_applied"] == {"l1": 0.0, "l2": 0.0, "l3a": 0.6, "l3b": 0.4}


def test_layer3a_calibrated_band_is_preserved():
    l1, l2, _, l3b = _clean_layers()
    result = FusionEngine().fuse("pt", l1, l2, {"verdict": False, "score": 0.41, "forensic_band": "clean"}, l3b)
    assert result["forensic_band"] == "clean"


def test_legacy_score_still_sets_forensic_boundaries():
    l1, l2, _, l3b = _clean_layers()
    assert FusionEngine().fuse("pt", l1, l2, {"verdict": False, "score": 0.40}, l3b)["forensic_band"] == "suspicious"
    assert FusionEngine().fuse("pt", l1, l2, {"verdict": False, "score": 0.70}, l3b)["forensic_band"] == "malicious"


def test_any_positive_layer_overrides_low_fused_score():
    l1, l2, _, l3b = _clean_layers()
    result = FusionEngine().fuse("pt", l1, l2, {"verdict": True, "score": 0.1}, l3b)
    assert result["verdict"] == "MALICIOUS"


def test_clean_safetensors_evaluates_benign():
    l1, l2, l3a, l3b = _clean_layers()
    # Fully calibrated clean Layer 3a result
    l3a_cal = {"verdict": False, "calibrated_distance": -1.912, "threshold": -1.2676, "forensic_band": "clean", "score": 0.07}
    result = FusionEngine().fuse("safetensors", l1, l2, l3a_cal, l3b)
    assert result["verdict"] == "BENIGN"
    assert result["score"] < 0.15
    assert result["forensic_band"] == "clean"


def test_clean_h5_evaluates_benign():
    l1, l2, l3a, l3b = _clean_layers()
    l3a_cal = {"verdict": False, "calibrated_distance": -1.912, "threshold": -1.2676, "forensic_band": "clean", "score": 0.07}
    result = FusionEngine().fuse("h5", l1, l2, l3a_cal, l3b)
    assert result["verdict"] == "BENIGN"
    assert result["score"] < 0.15
    assert result["forensic_band"] == "clean"


def test_clean_pt_evaluates_benign():
    l1, l2, l3a, l3b = _clean_layers()
    l2_benign = {"verdict": False, "probability": 0.075}
    l3a_cal = {"verdict": False, "calibrated_distance": -1.912, "threshold": -1.2676, "forensic_band": "clean", "score": 0.07}
    result = FusionEngine().fuse("pt", l1, l2_benign, l3a_cal, l3b)
    assert result["verdict"] == "BENIGN"
    assert result["score"] < 0.15
    assert result["forensic_band"] == "clean"


def test_stego_safetensors_evaluates_malicious():
    l1, l2, _, l3b = _clean_layers()
    l3a_stego = {"verdict": True, "calibrated_distance": 1.45, "threshold": -1.2676, "forensic_band": "malicious", "score": 1.0}
    result = FusionEngine().fuse("safetensors", l1, l2, l3a_stego, l3b)
    assert result["verdict"] == "MALICIOUS"
    assert result["score"] >= 0.85
    assert result["forensic_band"] == "malicious"

