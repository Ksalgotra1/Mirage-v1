import json
import os
import sys
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.main import app

client = TestClient(app)


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    print("Health check response:", res.json())


def test_scan_clean_pkl():
    clean_pkl = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fixtures", "clean_model.pkl")
    with open(clean_pkl, "rb") as f:
        res = client.post("/scan", files={"file": ("clean_model.pkl", f, "application/octet-stream")})
    assert res.status_code == 200
    data = res.json()
    print("\nScan Clean PKL:", json.dumps(data, indent=2))
    assert data["fused"]["verdict"] == "BENIGN"


def test_scan_obfuscated_pkl():
    obf_pkl = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fixtures", "obfuscated_fixture.pkl")
    with open(obf_pkl, "rb") as f:
        res = client.post("/scan", files={"file": ("obfuscated_fixture.pkl", f, "application/octet-stream")})
    assert res.status_code == 200
    data = res.json()
    print("\nScan Obfuscated PKL:", json.dumps(data, indent=2))
    assert data["fused"]["verdict"] == "MALICIOUS"
    assert data["layer2"]["verdict"] is True


def test_scan_safetensors():
    st_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fixtures", "clean_model.safetensors")
    with open(st_path, "rb") as f:
        res = client.post("/scan", files={"file": ("clean_model.safetensors", f, "application/octet-stream")})
    assert res.status_code == 200
    data = res.json()
    print("\nScan Safetensors:", json.dumps(data, indent=2))
    assert data["layer1"]["coverage_note"] is not None
    assert "safetensors" in data["layer1"]["coverage_note"]


if __name__ == "__main__":
    test_health()
    test_scan_clean_pkl()
    test_scan_obfuscated_pkl()
    test_scan_safetensors()
    print("\nAll FastAPI endpoint tests passed successfully!")
