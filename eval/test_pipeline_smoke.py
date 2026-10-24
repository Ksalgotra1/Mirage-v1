from __future__ import annotations

from unittest.mock import Mock, patch

import backend.main as api
from backend.summarizer import build_llm_prompt
from eval.run_layer3a_pipeline_smoke import _validate_prompt


def test_forensic_dispatch_skips_clean_and_selects_requested_band():
    engine = Mock()
    engine.analyze.return_value = {"band": "suspicious"}
    with patch.object(api, "_forensic_engine", engine):
        assert api._run_forensic_analysis("model.pt", "clean") is None
        assert api._run_forensic_analysis("model.pt", "suspicious") == {"band": "suspicious"}
    engine.analyze.assert_called_once_with("model.pt", band="suspicious")


def test_forensic_dispatch_isolates_engine_failure():
    engine = Mock()
    engine.analyze.side_effect = RuntimeError("fixture failure")
    with patch.object(api, "_forensic_engine", engine):
        result = api._run_forensic_analysis("model.pt", "malicious")
    assert result == {"error": "Forensic engine error: fixture failure", "band": "malicious"}


def test_band_specific_prompt_context_is_complete_and_safe():
    base = {
        "filename": "sample.pt", "format": "pt",
        "layer3a": {"score": 0.8, "extracted_hash": "secret"},
        "fused": {"verdict": "MALICIOUS"},
    }
    suspicious = {
        **base,
        "fused": {"verdict": "MALICIOUS", "forensic_band": "suspicious"},
        "steganography_forensics": {
            "band": "suspicious", "bit_depth": {"estimated_x": 8},
            "entropy": {"bits_per_byte": 7.9}, "distribution": None,
            "capacity": None, "magic_bytes": None,
        },
    }
    malicious = {
        **base,
        "fused": {"verdict": "MALICIOUS", "forensic_band": "malicious"},
        "steganography_forensics": {
            "band": "malicious", "bit_depth": {"estimated_x": 8},
            "entropy": {"bits_per_byte": 7.9}, "distribution": {"pattern": "dense"},
            "capacity": {"estimated_bytes": 12}, "magic_bytes": {"detected": False},
        },
    }
    assert all(_validate_prompt(build_llm_prompt(suspicious), "suspicious").values())
    assert all(_validate_prompt(build_llm_prompt(malicious), "malicious").values())
    assert suspicious["fused"]["verdict"] == "MALICIOUS"
