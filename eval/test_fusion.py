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
