"""Tests for Layer 2 (XGBoost) edge cases and multi-layer triangulation.

Verifies:
  1. Adversarial frequency dilution (XGBoost False Negative -> Layer 1 catch).
  2. Benign custom functional reducers (XGBoost False Positive -> Layer 1 clean pass).
  3. Fusion Engine multi-layer threat resolution.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from backend.fusion import FusionEngine
from detector.layer1_modelscan import Layer1ModelScan
from detector.layer2_classifier import Layer2Classifier
from fixtures.generate_edge_case_fixtures import (
    generate_benign_complex_reducer,
    generate_padded_exploit,
)


def test_adversarial_padding_frequency_dilution():
    """Case 1: Adversarially padded exploit dilutes opcode frequencies below XGBoost threshold,
    but Layer 1 AST AST inspection intercepts the dangerous callable.
    """
    l1 = Layer1ModelScan()
    l2 = Layer2Classifier()
    fusion = FusionEngine()

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "padded_exploit.pkl"
        generate_padded_exploit(p)

        res_l1 = l1.scan(str(p))
        res_l2 = l2.predict(str(p))
        fused = fusion.fuse(".pkl", res_l1, res_l2, None, None)

        # Layer 2 misses due to frequency dilution (P < 0.10)
        assert not res_l2["verdict"], "XGBoost should miss due to statistical dilution"
        assert res_l2["probability"] < 0.15, "XGBoost probability should be diluted"

        # Layer 1 AST catches the dangerous posix.system callable
        assert res_l1["verdict"], "Layer 1 AST must catch dangerous callable"
        assert any("system" in str(issue) for issue in res_l1.get("issues", []))

        # Fusion engine resolves overall file as MALICIOUS
        assert fused["verdict"] == "MALICIOUS"
        assert "Layer 1" in fused["explanation"]


def test_benign_custom_functional_reducer():
    """Case 2: Benign protocol 0 functional reducer produces elevated REDUCE/BUILD opcodes,
    triggering an XGBoost false alarm (FP), while Layer 1 confirms zero dangerous symbols.
    """
    l1 = Layer1ModelScan()
    l2 = Layer2Classifier()

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "benign_reducer.pkl"
        generate_benign_complex_reducer(p)

        res_l1 = l1.scan(str(p))
        res_l2 = l2.predict(str(p))

        # Layer 1 sees zero dangerous global symbols
        assert not res_l1["verdict"], "Layer 1 must pass benign standard library reducer"
        assert len(res_l1.get("issues", [])) == 0

        # Layer 2 flags high reduce/build structural anomaly (False Alarm)
        assert res_l2["verdict"], "XGBoost flags structural anomaly on protocol 0 functional reducers"
        assert res_l2["probability"] > 0.80
