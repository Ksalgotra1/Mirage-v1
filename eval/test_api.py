import json
import os
import sys
import time
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
    monkeypatch.setattr("backend.main._ASYNC_STAGE_MIN_SECONDS", 0.0)


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


def test_scan_job_sse_events_are_ordered_and_reconnectable():
    """The asynchronous endpoint publishes the real pipeline stages to SSE."""
    res = client.post(
        "/scans",
        files={"file": ("telemetry.pt", b"test model bytes", "application/octet-stream")},
    )
    assert res.status_code == 202
    job = res.json()
    assert job["scan_id"]
    assert job["stream_url"].endswith("/events")

    # The mocked pipeline is quick, but wait briefly for the worker to terminally emit.
    time.sleep(0.05)
    stream = client.get(job["stream_url"])
    assert stream.status_code == 200
    assert stream.headers["content-type"].startswith("text/event-stream")
    payload = stream.text
    assert "event: complete" in payload
    assert '"stage": "layer1"' in payload
    assert '"stage": "forensics", "status": "skipped"' in payload
    assert '"stage": "output"' in payload

    # A resumed stream after the final event must be empty rather than duplicate output.
    last_event_id = max(
        int(line.split(":", 1)[1].strip())
        for line in payload.splitlines()
        if line.startswith("id:")
    )
    resumed = client.get(job["stream_url"], headers={"Last-Event-ID": str(last_event_id)})
    assert resumed.status_code == 200
    assert resumed.text == ""


def test_scan_job_paces_executed_stages(monkeypatch):
    monkeypatch.setattr("backend.main._ASYNC_STAGE_MIN_SECONDS", 0.02)
    response = client.post("/scans", files={"file": ("paced.pt", b"bytes", "application/octet-stream")})
    assert response.status_code == 202
    stream = client.get(response.json()["stream_url"])
    events = [json.loads(line[6:]) for line in stream.text.splitlines() if line.startswith("data: ")]
    starts = {event["stage"]: event for event in events if event["status"] == "started"}
    completes = {event["stage"]: event for event in events if event["status"] == "completed"}
    for stage in ("upload", "layer1", "layer2", "layer3a", "layer3b", "fusion", "summary", "output"):
        assert completes[stage]["timestamp"] - starts[stage]["timestamp"] >= 0.015


def test_synchronous_scan_is_not_paced(monkeypatch):
    sleeps = []
    monkeypatch.setattr("backend.main._ASYNC_STAGE_MIN_SECONDS", 1.5)
    monkeypatch.setattr("backend.main.time.sleep", lambda seconds: sleeps.append(seconds))
    response = client.post("/scan", files={"file": ("sync.pt", b"bytes", "application/octet-stream")})
    assert response.status_code == 200
    assert sleeps == []


@pytest.mark.parametrize(
    ("band", "expected_status", "expected_depth"),
    [
        ("clean", "skipped", None),
        ("suspicious", "completed", "full (all 5 signals)"),
        ("malicious", "completed", "full (all 5 signals)"),
    ],
)
def test_scan_job_forensic_band_events(monkeypatch, band, expected_status, expected_depth):
    monkeypatch.setattr(
        "backend.main._run_layer3a_detector",
        lambda *_: {"layer": "3a", "verdict": band == "malicious", "score": 0.8 if band == "malicious" else 0.5,
                    "forensic_band": band},
    )
    forensic_calls = []

    def forensic_stub(_path, supplied_band):
        forensic_calls.append(supplied_band)
        return {"band": supplied_band, "analysis_depth": expected_depth}

    monkeypatch.setattr("backend.main._run_forensic_analysis", forensic_stub)
    response = client.post("/scans", files={"file": ("bands.pt", b"bytes", "application/octet-stream")})
    assert response.status_code == 202
    time.sleep(0.05)
    stream = client.get(response.json()["stream_url"])
    events = [json.loads(line[6:]) for line in stream.text.splitlines() if line.startswith("data: ")]
    forensic = [event for event in events if event["stage"] == "forensics"]
    assert forensic[-1]["status"] == expected_status
    assert forensic[-1]["metadata"]["forensic_band"] == band
    if band != "clean":
        assert forensic[-1]["metadata"]["analysis_depth"] == "full"
    assert forensic_calls == ([] if band == "clean" else [band])


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
