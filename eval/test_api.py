import json
import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _disable_external_summary(monkeypatch):
    """Endpoint tests isolate HTTP orchestration from slow external scanners."""
    monkeypatch.setattr(
        "backend.main.summarize_findings",
        lambda findings: {"summary": "test", "provider": "test"},
    )
    monkeypatch.setattr("backend.main.l1_scanner.scan", lambda _: {"layer": 1, "verdict": False, "issues": []})
    monkeypatch.setattr("backend.main.l2_classifier.predict", lambda _: {"layer": 2, "verdict": False, "probability": 0.0})
    monkeypatch.setattr(
        "backend.main._run_layer3a_detector",
        lambda *_: {"layer": "3a", "verdict": False, "distance": -0.1, "calibrated_distance": -0.1,
                     "threshold": 0.072236, "score": 0.2, "forensic_band": "clean"},
    )
    monkeypatch.setattr(
        "backend.main._run_layer3b_detector",
        lambda *_: {"layer": "3b", "verdict": False, "score": 0.0},
    )


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    print("[OK] Health check response:", res.json())


def test_scan_clean_pt():
    clean_pt = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fixtures", "benign_resnet18_real.pt")
    if os.path.exists(clean_pt):
        with open(clean_pt, "rb") as f:
            res = client.post("/scan", files={"file": ("benign_resnet18_real.pt", f, "application/octet-stream")})
        assert res.status_code == 200
        data = res.json()
        print("\n[OK] Scan Clean PT:", json.dumps(data, indent=2))
        assert data["fused"]["verdict"] == "BENIGN"
        assert data["layer3a"]["verdict"] is False


def test_scan_stego_safetensors():
    stego_st = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fixtures", "stego_cnn_50pct.safetensors")
    if os.path.exists(stego_st):
        with open(stego_st, "rb") as f:
            res = client.post("/scan", files={"file": ("stego_cnn_50pct.safetensors", f, "application/octet-stream")})
        assert res.status_code == 200
        data = res.json()
        print("\n[OK] Scan Stego Safetensors 50%:", json.dumps(data, indent=2))
        assert {"distance", "calibrated_distance", "threshold", "score", "forensic_band"} <= data["layer3a"].keys()
        assert data["layer3a"]["forensic_band"] in ("clean", "suspicious", "malicious")


def test_fixtures_eval():
    res = client.get("/fixtures/eval")
    assert res.status_code == 200
    data = res.json()
    print(f"\n[OK] Fixtures eval endpoint: {data['results_count']} results returned")


if __name__ == "__main__":
    test_health()
    test_scan_clean_pt()
    test_scan_stego_safetensors()
    test_fixtures_eval()
    print("\nAll FastAPI backend integration tests passed successfully!")
