"""
eval/test_pipeline_resilience.py — Chaos & Layer Break Test Suite
===================================================================
Comprehensive fault-injection and ablation testing for PROJECT-CLIMAX:

1. Single-Layer Fault Injection (Breaking 1 layer at a time):
   - Break Layer 1 (ModelScan crash / exception)
   - Break Layer 2 (XGBoost model missing / crash)
   - Break Layer 3a (GF-CNN VRAM OOM / tensor failure)
   - Break Layer 3b (Weight distribution failure)

2. Pair-Wise Layer Fault Injection (Breaking 2 layers simultaneously):
   - Break [L1 + L2] (All opcode analysis offline)
   - Break [L1 + L3a] (Rule bouncer + GF LSB offline)
   - Break [L1 + L3b] (Rule bouncer + Weight distribution offline)
   - Break [L2 + L3a] (XGBoost + GF LSB offline)
   - Break [L2 + L3b] (XGBoost + Weight distribution offline)
   - Break [L3a + L3b] (All weight-space analysis offline)

3. Dynamic Fusion & Evasion Matrix:
   - L1 evasion: Obfuscated opcode malware bypasses L1 -> caught by L2
   - L1+L2 evasion: Stego in .safetensors bypasses opcode layers -> caught by L3a
"""

import json
import os
import sys
from unittest.mock import patch

from fastapi.testclient import TestClient

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from backend.main import app
from backend.fusion import FusionEngine

client = TestClient(app)

BENIGN_PT = os.path.join(PROJECT_ROOT, "fixtures", "benign_resnet18_real.pt")
MALICIOUS_OPCODE_PT = os.path.join(PROJECT_ROOT, "fixtures", "malicious_pickle_exec_direct_system_var0.pt")
OBFUSCATED_PKL = os.path.join(PROJECT_ROOT, "fixtures", "malicious_pickle_exec_importlib_var0.pt")
STEGO_SAFETENSORS = os.path.join(PROJECT_ROOT, "fixtures", "stego_cnn_50pct.safetensors")


def _scan_file(filepath: str):
    """Helper to upload a file to the POST /scan endpoint."""
    filename = os.path.basename(filepath)
    with open(filepath, "rb") as f:
        response = client.post("/scan", files={"file": (filename, f, "application/octet-stream")})
    return response


# ===========================================================================
# 1. SINGLE-LAYER FAULT INJECTION (Breaking 1 layer at a time)
# ===========================================================================

def test_break_layer1_alone():
    """Simulate Layer 1 raising a fatal exception (e.g. ModelScan crash)."""
    print("\n--- [TEST] Breaking Layer 1 Alone ---")
    with patch("backend.main.l1_scanner.scan", side_effect=RuntimeError("Simulated ModelScan crash")):
        # 1. Scan malicious opcode model -> L2 should still catch it!
        res = _scan_file(MALICIOUS_OPCODE_PT)
        assert res.status_code == 200, f"Expected 200 OK, got {res.status_code}"
        data = res.json()
        assert "execution failure" in data["layer1"].get("coverage_note", "")
        assert data["layer2"]["verdict"] is True
        assert data["fused"]["verdict"] == "MALICIOUS"
        print("  [OK] L1 crashed -> L2 successfully caught opcode malware (fused=MALICIOUS)")

        # 2. Scan stego model -> L3a should still catch it!
        res_stego = _scan_file(STEGO_SAFETENSORS)
        assert res_stego.status_code == 200
        data_stego = res_stego.json()
        assert data_stego["layer3a"]["verdict"] is True
        assert data_stego["fused"]["verdict"] == "MALICIOUS"
        print("  [OK] L1 crashed -> L3a successfully caught stego malware (fused=MALICIOUS)")


def test_break_layer2_alone():
    """Simulate Layer 2 raising a fatal exception (e.g. XGBoost model corrupted)."""
    print("\n--- [TEST] Breaking Layer 2 Alone ---")
    with patch("backend.main.l2_classifier.predict", side_effect=ValueError("Simulated XGBoost corrupted")):
        # 1. Scan direct system exploit -> L1 should catch it!
        res = _scan_file(MALICIOUS_OPCODE_PT)
        assert res.status_code == 200
        data = res.json()
        assert "execution failure" in data["layer2"].get("coverage_note", "")
        assert data["layer1"]["verdict"] is True
        assert data["fused"]["verdict"] == "MALICIOUS"
        print("  [OK] L2 crashed -> L1 successfully caught direct callable (fused=MALICIOUS)")

        # 2. Scan stego model -> L3a should catch it!
        res_stego = _scan_file(STEGO_SAFETENSORS)
        assert res_stego.status_code == 200
        data_stego = res_stego.json()
        assert data_stego["layer3a"]["verdict"] is True
        assert data_stego["fused"]["verdict"] == "MALICIOUS"
        print("  [OK] L2 crashed -> L3a successfully caught stego payload (fused=MALICIOUS)")


def test_break_layer3a_alone():
    """Simulate Layer 3a raising a fatal GPU/CUDA Out of Memory error."""
    print("\n--- [TEST] Breaking Layer 3a Alone ---")
    with patch("backend.main._run_layer3a_detector", side_effect=RuntimeError("CUDA out of memory in GF-CNN")):
        # 1. Scan malicious opcode model -> L1 and L2 still work!
        res = _scan_file(MALICIOUS_OPCODE_PT)
        assert res.status_code == 200
        data = res.json()
        assert "execution failure" in data["layer3a"].get("coverage_note", "")
        assert data["fused"]["verdict"] == "MALICIOUS"
        print("  [OK] L3a OOM -> L1/L2 successfully caught opcode malware (fused=MALICIOUS)")

        # 2. Scan clean model -> doesn't crash, stays benign
        res_clean = _scan_file(BENIGN_PT)
        assert res_clean.status_code == 200
        data_clean = res_clean.json()
        assert data_clean["fused"]["verdict"] == "BENIGN"
        print("  [OK] L3a OOM -> Clean model scan safely returns BENIGN")


def test_break_layer3b_alone():
    """Simulate Layer 3b raising a fatal exception."""
    print("\n--- [TEST] Breaking Layer 3b Alone ---")
    with patch("backend.main._run_layer3b_detector", side_effect=KeyError("Reference stats missing")):
        res = _scan_file(STEGO_SAFETENSORS)
        assert res.status_code == 200
        data = res.json()
        assert "execution failure" in data["layer3b"].get("coverage_note", "")
        assert data["layer3a"]["verdict"] is True
        assert data["fused"]["verdict"] == "MALICIOUS"
        print("  [OK] L3b crashed -> L3a successfully caught stego payload (fused=MALICIOUS)")


# ===========================================================================
# 2. PAIR-WISE LAYER FAULT INJECTION (Breaking 2 layers simultaneously)
# ===========================================================================

def test_break_pair_l1_and_l2():
    """Break [L1 + L2]: All Opcode analysis layers are offline simultaneously."""
    print("\n--- [TEST] Breaking Pair: [Layer 1 + Layer 2] ---")
    with patch("backend.main.l1_scanner.scan", side_effect=RuntimeError("L1 down")), \
         patch("backend.main.l2_classifier.predict", side_effect=RuntimeError("L2 down")):

        # Stego detection via L3a should still work 100%
        res = _scan_file(STEGO_SAFETENSORS)
        assert res.status_code == 200
        data = res.json()
        assert data["layer3a"]["verdict"] is True
        assert data["fused"]["verdict"] == "MALICIOUS"
        assert "execution failure" in data["layer1"]["coverage_note"]
        assert "execution failure" in data["layer2"]["coverage_note"]
        print("  [OK] Both opcode layers down -> L3a successfully detects stego in .safetensors")

        # Benign model scan does not crash
        res_clean = _scan_file(BENIGN_PT)
        assert res_clean.status_code == 200
        assert res_clean.json()["fused"]["verdict"] == "BENIGN"
        print("  [OK] Both opcode layers down -> Benign model gracefully defaults to BENIGN")


def test_break_pair_l1_and_l3a():
    """Break [L1 + L3a]: Rule-based scanner + GF LSB detector offline."""
    print("\n--- [TEST] Breaking Pair: [Layer 1 + Layer 3a] ---")
    with patch("backend.main.l1_scanner.scan", side_effect=RuntimeError("L1 down")), \
         patch("backend.main._run_layer3a_detector", side_effect=RuntimeError("L3a down")):

        # L2 should still catch opcode anomalies
        res = _scan_file(MALICIOUS_OPCODE_PT)
        assert res.status_code == 200
        data = res.json()
        assert data["layer2"]["verdict"] is True
        assert data["fused"]["verdict"] == "MALICIOUS"
        print("  [OK] L1 and L3a down -> Layer 2 (XGBoost) successfully catches exploit")


def test_break_pair_l1_and_l3b():
    """Break [L1 + L3b]: Rule bouncer + Weight distribution detector offline."""
    print("\n--- [TEST] Breaking Pair: [Layer 1 + Layer 3b] ---")
    with patch("backend.main.l1_scanner.scan", side_effect=RuntimeError("L1 down")), \
         patch("backend.main._run_layer3b_detector", side_effect=RuntimeError("L3b down")):

        # L2 catches opcode attacks, L3a catches stego
        res_op = _scan_file(MALICIOUS_OPCODE_PT)
        assert res_op.status_code == 200
        assert res_op.json()["layer2"]["verdict"] is True

        res_stego = _scan_file(STEGO_SAFETENSORS)
        assert res_stego.status_code == 200
        assert res_stego.json()["layer3a"]["verdict"] is True
        print("  [OK] L1 and L3b down -> L2 catches opcodes, L3a catches stego")


def test_break_pair_l2_and_l3a():
    """Break [L2 + L3a]: XGBoost classifier + GF LSB detector offline."""
    print("\n--- [TEST] Breaking Pair: [Layer 2 + Layer 3a] ---")
    with patch("backend.main.l2_classifier.predict", side_effect=RuntimeError("L2 down")), \
         patch("backend.main._run_layer3a_detector", side_effect=RuntimeError("L3a down")):

        # L1 should still catch known dangerous callables
        res = _scan_file(MALICIOUS_OPCODE_PT)
        assert res.status_code == 200
        assert res.json()["layer1"]["verdict"] is True
        assert res.json()["fused"]["verdict"] == "MALICIOUS"
        print("  [OK] L2 and L3a down -> Layer 1 catches known callable")


def test_break_pair_l2_and_l3b():
    """Break [L2 + L3b]: XGBoost classifier + Weight distribution detector offline."""
    print("\n--- [TEST] Breaking Pair: [Layer 2 + Layer 3b] ---")
    with patch("backend.main.l2_classifier.predict", side_effect=RuntimeError("L2 down")), \
         patch("backend.main._run_layer3b_detector", side_effect=RuntimeError("L3b down")):

        # L1 catches direct callables, L3a catches stego
        res_l1 = _scan_file(MALICIOUS_OPCODE_PT)
        assert res_l1.json()["layer1"]["verdict"] is True

        res_l3a = _scan_file(STEGO_SAFETENSORS)
        assert res_l3a.json()["layer3a"]["verdict"] is True
        print("  [OK] L2 and L3b down -> L1 catches callables, L3a catches stego")


def test_break_pair_l3a_and_l3b():
    """Break [L3a + L3b]: Both weight-space layers offline."""
    print("\n--- [TEST] Breaking Pair: [Layer 3a + Layer 3b] ---")
    with patch("backend.main._run_layer3a_detector", side_effect=RuntimeError("L3a down")), \
         patch("backend.main._run_layer3b_detector", side_effect=RuntimeError("L3b down")):

        # Opcode attacks are still caught by L1 + L2
        res = _scan_file(MALICIOUS_OPCODE_PT)
        assert res.status_code == 200
        data = res.json()
        assert data["layer1"]["verdict"] is True or data["layer2"]["verdict"] is True
        assert data["fused"]["verdict"] == "MALICIOUS"
        print("  [OK] Both weight layers down -> L1 and L2 catch opcode malware")


# ===========================================================================
# 3. MULTI-LAYER EVASION / TARGETED ATTACK BYPASS MATRIX
# ===========================================================================

def test_evasion_matrix():
    """Verify defense-in-depth: When attacks evade one layer, other layers catch them."""
    print("\n--- [TEST] Evasion & Defense-in-Depth Matrix ---")

    # 1. Obfuscated payload (bypasses L1 string rules -> caught by L2 XGBoost)
    res_obf = _scan_file(OBFUSCATED_PKL)
    data_obf = res_obf.json()
    assert data_obf["layer2"]["verdict"] is True, "L2 should catch obfuscated structural anomaly"
    assert data_obf["fused"]["verdict"] == "MALICIOUS"
    print("  [OK] L1 Evasion: Obfuscated payload caught by Layer 2 XGBoost")

    # 2. Stego in Safetensors (bypasses L1 & L2 opcode formats -> caught by Layer 3a)
    res_st = _scan_file(STEGO_SAFETENSORS)
    data_st = res_st.json()
    assert data_st["layer1"]["verdict"] is False
    assert data_st["layer2"]["verdict"] is False
    assert data_st["layer3a"]["verdict"] is True
    assert data_st["fused"]["verdict"] == "MALICIOUS"
    print("  [OK] L1/L2 Evasion: Safetensors stego caught by Layer 3a GF-LSB detector")


# ===========================================================================
# Main Runner
# ===========================================================================

def run_all_resilience_tests():
    print("=" * 75)
    print("  PROJECT-CLIMAX: CHAOS & LAYER-BREAK RESILIENCE TEST SUITE")
    print("=" * 75)

    # 1. Single Layer Breaks
    test_break_layer1_alone()
    test_break_layer2_alone()
    test_break_layer3a_alone()
    test_break_layer3b_alone()

    # 2. Pair-Wise Layer Breaks (All 6 combinations)
    test_break_pair_l1_and_l2()
    test_break_pair_l1_and_l3a()
    test_break_pair_l1_and_l3b()
    test_break_pair_l2_and_l3a()
    test_break_pair_l2_and_l3b()
    test_break_pair_l3a_and_l3b()

    # 3. Evasion & Defense in Depth
    test_evasion_matrix()

    print("\n" + "=" * 75)
    print("  ALL 11 RESILIENCE & CHAOS TESTS PASSED! PIPELINE IS FAULT-TOLERANT.")
    print("=" * 75)


if __name__ == "__main__":
    run_all_resilience_tests()
