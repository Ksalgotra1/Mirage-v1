"""
eval/test_disarmament.py
========================
Full test suite for the /disarm endpoint and supporting modules.

Covers:
  - Unit tests for ScanSessionStore
  - Unit tests for disarmer_lsb (all formats)
  - Unit tests for disarmer_opcode (Phase 1 + Phase 2)
  - Integration tests for POST /scan → POST /disarm full flow
  - Edge cases: zero-weight model, non-float32 tensors, rate limiting, etc.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import struct
import tempfile
import time
import uuid
from pathlib import Path
from typing import Generator
from unittest.mock import patch

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.main import app
from backend.session_store import ScanSessionStore, SessionEntry
from detector.disarmer_lsb import disarm_lsb, _zero_lsb_array
from detector.disarmer_opcode import (
    disarm_opcodes,
    DisarmamentNotPossibleError,
    DANGEROUS_MODULES,
)
from detector.disarmament_engine import (
    DisarmamentEngine,
    DisarmamentVerificationError,
)

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
SESSIONS_DIR = Path(__file__).resolve().parent.parent / "sessions"

client = TestClient(app)


def _make_benign_state_dict() -> dict:
    """Small in-memory state dict for unit tests."""
    rng = np.random.default_rng(seed=0)
    return {
        "layer1.weight": torch.tensor(rng.normal(0, 0.02, (8, 4)), dtype=torch.float32),
        "layer1.bias":   torch.tensor(rng.normal(0, 0.01, (8,)),   dtype=torch.float32),
        "layer2.weight": torch.tensor(rng.normal(0, 0.02, (4, 8)), dtype=torch.float32),
        "layer2.bias":   torch.tensor(rng.normal(0, 0.01, (4,)),   dtype=torch.float32),
    }


def _inject_x_bits(state_dict: dict, x_bits: int, seed: int = 42) -> dict:
    """Inject random LSB payload into a state dict (mirrors training corpus method)."""
    from fixtures.generate_patch_dataset import inject_xlsb
    result = {}
    for k, v in state_dict.items():
        if v.dtype == torch.float32:
            result[k] = inject_xlsb(v, x_bits=x_bits, payload_type="random", seed=seed)
        else:
            result[k] = v
    return result


@pytest.fixture(autouse=True)
def reset_rate_limits():
    from backend.main import _rate_counters
    _rate_counters.clear()
    yield
    _rate_counters.clear()


@pytest.fixture
def tmp_dir() -> Generator[str, None, None]:
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def benign_pt(tmp_dir) -> str:
    path = os.path.join(tmp_dir, "benign.pt")
    torch.save(_make_benign_state_dict(), path)
    return path


@pytest.fixture
def stego_pt(tmp_dir) -> str:
    path = os.path.join(tmp_dir, "stego.pt")
    stego = _inject_x_bits(_make_benign_state_dict(), x_bits=8)
    torch.save(stego, path)
    return path


@pytest.fixture
def stego_safetensors(tmp_dir) -> str:
    from safetensors.torch import save_file
    path = os.path.join(tmp_dir, "stego.safetensors")
    stego = _inject_x_bits(_make_benign_state_dict(), x_bits=8)
    save_file(stego, path)
    return path


@pytest.fixture
def malicious_pt_direct():
    return str(FIXTURES_DIR / "malicious_pickle_exec_direct_system_var0.pt")


@pytest.fixture
def malicious_pt_importlib():
    return str(FIXTURES_DIR / "malicious_pickle_exec_importlib_var0.pt")


@pytest.fixture
def real_stego_safetensors():
    return str(FIXTURES_DIR / "stego_cnn_50pct.safetensors")


# ===========================================================================
# 1. ScanSessionStore — Unit Tests
# ===========================================================================

class TestScanSessionStore:

    def test_save_and_get(self, benign_pt):
        store = ScanSessionStore()
        scan_result = {"verdict": "MALICIOUS"}
        scan_id = store.save(benign_pt, scan_result)
        assert isinstance(scan_id, str)
        # Validate UUID4 format
        uuid.UUID(scan_id, version=4)

        entry = store.get(scan_id)
        assert entry is not None
        assert entry.scan_id == scan_id
        assert os.path.exists(entry.filepath)

    def test_get_returns_none_for_unknown_id(self):
        store = ScanSessionStore()
        assert store.get("nonexistent-id") is None

    def test_expired_session_returns_none(self, benign_pt):
        store = ScanSessionStore()
        scan_id = store.save(benign_pt, {})
        with store._lock:
            store._store[scan_id].ttl_seconds = -1  # force expire
        assert store.get(scan_id) is None

    def test_delete_removes_file_and_entry(self, benign_pt):
        store = ScanSessionStore()
        scan_id = store.save(benign_pt, {})
        entry = store.get(scan_id)
        stored_path = entry.filepath
        result = store.delete(scan_id)
        assert result is True
        assert store.get(scan_id) is None
        assert not os.path.exists(stored_path)

    def test_delete_nonexistent_returns_false(self):
        store = ScanSessionStore()
        assert store.delete("does-not-exist") is False

    def test_purge_removes_only_expired(self, tmp_dir):
        store = ScanSessionStore()
        alive_path = os.path.join(tmp_dir, "alive.pt")
        dead_path = os.path.join(tmp_dir, "dead.pt")
        torch.save({}, alive_path)
        torch.save({}, dead_path)

        alive_id = store.save(alive_path, {})
        dead_id = store.save(dead_path, {})

        with store._lock:
            store._store[dead_id].ttl_seconds = -1

        purged = store.purge_expired()
        assert purged == 1
        assert store.get(alive_id) is not None
        assert store.get(dead_id) is None

    def test_scan_ids_are_unique(self, benign_pt):
        store = ScanSessionStore()
        ids = [store.save(benign_pt, {}) for _ in range(20)]
        assert len(set(ids)) == 20

    def test_session_seconds_remaining(self, benign_pt):
        store = ScanSessionStore()
        scan_id = store.save(benign_pt, {})
        entry = store.get(scan_id)
        assert 3550 < entry.seconds_remaining <= 3600


# ===========================================================================
# 2. disarmer_lsb — Unit Tests
# ===========================================================================

class TestLsbDisarmer:

    def test_zero_lsb_array_clears_q3_byte(self):
        """After zeroing x_bits=8, every LSB byte must be 0x00."""
        arr = np.random.default_rng(1).random(100).astype(np.float32)
        cleaned, _ = _zero_lsb_array(arr, x_bits=8)
        flat = cleaned.flatten().view(np.uint32)
        assert np.all((flat & 0xFF) == 0), "LSB byte (bits 0-7) must be zeroed"

    def test_lsb_payload_extraction_is_correct(self):
        """Extracted payload must equal the original LSB bytes."""
        arr = np.ones(64, dtype=np.float32)
        lsb_before = (arr.flatten().astype(np.float32).view(np.uint32) & 0xFF).astype(np.uint8).tobytes()
        _, payload = _zero_lsb_array(arr, x_bits=8)
        assert payload == lsb_before

    def test_weight_accuracy_preserved_after_lsb_zeroing(self, stego_pt):
        """Max weight delta after zeroing must be negligible (< 1e-4)."""
        original = torch.load(stego_pt, map_location="cpu", weights_only=True)
        result = disarm_lsb(stego_pt, x_bits=8)
        cleaned = torch.load(result.cleaned_filepath, map_location="cpu", weights_only=True)

        for key in original:
            if original[key].dtype == torch.float32:
                delta = (original[key] - cleaned[key]).abs().max().item()
                assert delta < 1e-4, f"Weight delta too large in {key}: {delta}"
        os.remove(result.cleaned_filepath)

    def test_pt_format_roundtrip(self, stego_pt):
        """Disarmed .pt must load without error and have same keys."""
        result = disarm_lsb(stego_pt, x_bits=8)
        loaded = torch.load(result.cleaned_filepath, map_location="cpu", weights_only=True)
        original = torch.load(stego_pt, map_location="cpu", weights_only=True)
        assert set(loaded.keys()) == set(original.keys())
        assert result.tensors_processed > 0
        os.remove(result.cleaned_filepath)

    def test_safetensors_format_roundtrip(self, stego_safetensors):
        """Disarmed .safetensors must parse correctly and preserve shapes."""
        from safetensors.torch import load_file
        original = load_file(stego_safetensors)
        result = disarm_lsb(stego_safetensors, x_bits=8)
        cleaned = load_file(result.cleaned_filepath)
        for key in original:
            assert cleaned[key].shape == original[key].shape
            assert cleaned[key].dtype == original[key].dtype
        os.remove(result.cleaned_filepath)

    def test_payload_sha256_is_hex_string(self, stego_pt):
        result = disarm_lsb(stego_pt, x_bits=8)
        assert len(result.payload_sha256) == 64
        int(result.payload_sha256, 16)  # must be valid hex
        os.remove(result.cleaned_filepath)

    def test_payload_size_bytes_is_positive(self, stego_pt):
        result = disarm_lsb(stego_pt, x_bits=8)
        assert result.payload_size_bytes > 0
        os.remove(result.cleaned_filepath)

    def test_disarm_clean_file_is_noop_for_lsb(self, benign_pt):
        """Disarming a clean file should produce an output (no error), not crash."""
        result = disarm_lsb(benign_pt, x_bits=8)
        assert os.path.exists(result.cleaned_filepath)
        assert result.tensors_processed > 0
        os.remove(result.cleaned_filepath)

    def test_unsupported_format_raises_value_error(self, tmp_dir):
        bad_path = os.path.join(tmp_dir, "model.onnx")
        with open(bad_path, "wb") as f:
            f.write(b"fakecontent")
        with pytest.raises(ValueError, match="not supported"):
            disarm_lsb(bad_path)

    def test_missing_file_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            disarm_lsb("/nonexistent/path/model.pt")


# ===========================================================================
# 3. disarmer_opcode — Unit Tests
# ===========================================================================

class TestOpcodeDisarmer:

    def test_phase1_strips_direct_os_system(self, malicious_pt_direct):
        result = disarm_opcodes(malicious_pt_direct)
        assert result.cleaned_filepath
        # The cleaned file must load safely with weights_only=True
        loaded = torch.load(result.cleaned_filepath, map_location="cpu", weights_only=True)
        assert isinstance(loaded, dict)
        os.remove(result.cleaned_filepath)

    def test_phase1_strips_importlib_payload(self, malicious_pt_importlib):
        result = disarm_opcodes(malicious_pt_importlib)
        assert result.cleaned_filepath
        loaded = torch.load(result.cleaned_filepath, map_location="cpu", weights_only=True)
        assert isinstance(loaded, dict)
        os.remove(result.cleaned_filepath)

    def test_disarmed_opcode_file_is_loadable(self, malicious_pt_direct):
        """Cleaned file must be loadable without executing any system call."""
        result = disarm_opcodes(malicious_pt_direct)
        # Should NOT raise
        loaded = torch.load(result.cleaned_filepath, map_location="cpu", weights_only=True)
        os.remove(result.cleaned_filepath)

    def test_unsupported_format_raises_value_error(self, tmp_dir):
        bad = os.path.join(tmp_dir, "model.safetensors")
        with open(bad, "wb") as f:
            f.write(b"fakecontent")
        with pytest.raises(ValueError):
            disarm_opcodes(bad)

    def test_missing_file_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            disarm_opcodes("/nonexistent/model.pt")

    def test_benign_pt_disarm_produces_clean_output(self, benign_pt):
        """Disarming a clean .pt must not crash."""
        result = disarm_opcodes(benign_pt)
        assert os.path.exists(result.cleaned_filepath)
        os.remove(result.cleaned_filepath)


# ===========================================================================
# 4. DisarmamentEngine — Unit Tests
# ===========================================================================

class TestDisarmamentEngine:

    def test_lsb_disarm_report_has_all_fields(self, stego_safetensors):
        engine = DisarmamentEngine()
        scan_result = {"layer3a": {"verdict": True}, "layer1": {"verdict": False},
                       "layer2": {"verdict": False}, "layer3b": {"verdict": False}}
        report = engine.disarm(stego_safetensors, scan_result)
        assert report.cleaned_filepath
        assert report.original_sha256
        assert report.cleaned_sha256
        assert "lsb" in report.phases_applied
        assert report.payload_sha256 is not None
        assert report.verification_passed
        os.remove(report.cleaned_filepath)

    def test_opcode_disarm_report_has_all_fields(self, malicious_pt_direct):
        engine = DisarmamentEngine()
        scan_result = {"layer1": {"verdict": True}, "layer2": {"verdict": False},
                       "layer3a": {"verdict": False}, "layer3b": {"verdict": False}}
        report = engine.disarm(malicious_pt_direct, scan_result)
        assert "opcode" in report.phases_applied
        assert report.verification_passed
        os.remove(report.cleaned_filepath)

    def test_verification_failure_raises_error(self, stego_safetensors):
        engine = DisarmamentEngine()
        scan_result = {"layer3a": {"verdict": True}, "layer1": {"verdict": False},
                       "layer2": {"verdict": False}, "layer3b": {"verdict": False}}
        # Mock _verify to always return MALICIOUS
        with patch.object(engine, "_verify", return_value={"fused": {"verdict": "MALICIOUS"}}):
            with pytest.raises(DisarmamentVerificationError):
                engine.disarm(stego_safetensors, scan_result)

    def test_original_and_cleaned_sha256_differ(self, stego_safetensors):
        """The cleaned file must have a different hash from the original."""
        engine = DisarmamentEngine()
        scan_result = {"layer3a": {"verdict": True}, "layer1": {"verdict": False},
                       "layer2": {"verdict": False}, "layer3b": {"verdict": False}}
        report = engine.disarm(stego_safetensors, scan_result)
        assert report.original_sha256 != report.cleaned_sha256
        os.remove(report.cleaned_filepath)

    def test_audit_log_written(self, stego_safetensors, tmp_dir):
        """After disarmament, the audit log must contain a relevant entry."""
        import logging
        from detector.disarmament_engine import _AUDIT_LOG_PATH
        engine = DisarmamentEngine()
        scan_result = {"layer3a": {"verdict": True}, "layer1": {"verdict": False},
                       "layer2": {"verdict": False}, "layer3b": {"verdict": False}}
        report = engine.disarm(stego_safetensors, scan_result)
        assert _AUDIT_LOG_PATH.exists()
        log_text = _AUDIT_LOG_PATH.read_text(encoding="utf-8")
        assert "disarmament_started" in log_text
        assert "lsb_phase_complete" in log_text
        os.remove(report.cleaned_filepath)


# ===========================================================================
# 5. Integration Tests — POST /scan → POST /disarm
# ===========================================================================

class TestDisarmEndpointIntegration:

    def test_scan_malicious_returns_scan_id(self, malicious_pt_direct):
        with open(malicious_pt_direct, "rb") as f:
            resp = client.post("/scan", files={"file": ("malicious.pt", f, "application/octet-stream")})
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("disarmament") is not None
        assert "scan_id" in data["disarmament"]
        assert data["disarmament"]["status"] == "READY_FOR_DISARMAMENT"

    def test_scan_clean_may_not_have_scan_id(self, benign_pt):
        with open(benign_pt, "rb") as f:
            resp = client.post("/scan", files={"file": ("benign.pt", f, "application/octet-stream")})
        assert resp.status_code == 200
        data = resp.json()
        # Clean files should have no disarmament section or it should be None
        disarm = data.get("disarmament")
        if disarm is not None:
            # If it has one, it shouldn't be READY (no threat detected)
            assert disarm.get("status") != "READY_FOR_DISARMAMENT"

    def test_disarm_without_confirm_returns_400(self, malicious_pt_direct):
        # First scan to get a scan_id
        with open(malicious_pt_direct, "rb") as f:
            resp = client.post("/scan", files={"file": ("malicious.pt", f, "application/octet-stream")})
        scan_id = resp.json()["disarmament"]["scan_id"]

        resp_disarm = client.post("/disarm", data={"scan_id": scan_id, "confirm": "false"})
        assert resp_disarm.status_code == 400

    def test_disarm_with_invalid_scan_id_returns_404(self):
        resp = client.post("/disarm", data={"scan_id": str(uuid.uuid4()), "confirm": "true"})
        assert resp.status_code == 404

    def test_disarm_without_scan_id_returns_422(self):
        resp = client.post("/disarm", data={"confirm": "true"})
        assert resp.status_code == 422

    def test_full_disarm_flow_stego_safetensors(self, real_stego_safetensors):
        """Scan stego → get scan_id → disarm → verify clean download."""
        with open(real_stego_safetensors, "rb") as f:
            scan_resp = client.post("/scan", files={"file": ("stego.safetensors", f, "application/octet-stream")})
        assert scan_resp.status_code == 200

        disarm_info = scan_resp.json().get("disarmament")
        if disarm_info is None:
            pytest.skip("Stego file was not flagged — Layer 3a bundle may not be present")

        scan_id = disarm_info["scan_id"]

        disarm_resp = client.post("/disarm", data={"scan_id": scan_id, "confirm": "true"})
        assert disarm_resp.status_code == 200
        assert "attachment" in disarm_resp.headers.get("content-disposition", "")
        assert disarm_resp.headers.get("x-verification-passed") == "true"
        assert disarm_resp.headers.get("x-payload-sha256", "N/A") != ""

    def test_full_disarm_flow_malicious_pickle(self, malicious_pt_direct):
        """Scan malicious pickle → disarm → returned file loads safely."""
        with open(malicious_pt_direct, "rb") as f:
            scan_resp = client.post("/scan", files={"file": ("malicious.pt", f, "application/octet-stream")})
        assert scan_resp.status_code == 200

        scan_id = scan_resp.json()["disarmament"]["scan_id"]

        disarm_resp = client.post("/disarm", data={"scan_id": scan_id, "confirm": "true"})
        assert disarm_resp.status_code == 200

        # Write returned file and verify it loads safely
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp.write(disarm_resp.content)
            tmp_path = tmp.name

        try:
            loaded = torch.load(tmp_path, map_location="cpu", weights_only=True)
            assert isinstance(loaded, dict)
        finally:
            os.remove(tmp_path)

    def test_disarm_is_one_time_use(self, malicious_pt_direct):
        """Same scan_id cannot be used twice."""
        with open(malicious_pt_direct, "rb") as f:
            scan_resp = client.post("/scan", files={"file": ("malicious.pt", f, "application/octet-stream")})
        scan_id = scan_resp.json()["disarmament"]["scan_id"]

        client.post("/disarm", data={"scan_id": scan_id, "confirm": "true"})
        second = client.post("/disarm", data={"scan_id": scan_id, "confirm": "true"})
        assert second.status_code == 404

    def test_disarm_response_has_forensic_headers(self, malicious_pt_direct):
        with open(malicious_pt_direct, "rb") as f:
            scan_resp = client.post("/scan", files={"file": ("malicious.pt", f, "application/octet-stream")})
        scan_id = scan_resp.json()["disarmament"]["scan_id"]

        disarm_resp = client.post("/disarm", data={"scan_id": scan_id, "confirm": "true"})
        assert disarm_resp.status_code == 200
        for header in ["x-original-sha256", "x-cleaned-sha256", "x-phases-applied"]:
            assert header in disarm_resp.headers, f"Missing header: {header}"

    def test_disarm_streams_correct_file_extension(self, malicious_pt_direct):
        with open(malicious_pt_direct, "rb") as f:
            scan_resp = client.post("/scan", files={"file": ("malicious.pt", f, "application/octet-stream")})
        scan_id = scan_resp.json()["disarmament"]["scan_id"]

        disarm_resp = client.post("/disarm", data={"scan_id": scan_id, "confirm": "true"})
        assert disarm_resp.status_code == 200
        cd = disarm_resp.headers.get("content-disposition", "")
        assert ".pt" in cd


# ===========================================================================
# 6. Rate Limiting Tests
# ===========================================================================

class TestRateLimiting:

    def test_rate_limit_allows_up_to_10_requests(self, malicious_pt_direct):
        """First 10 disarm calls from same IP within window must not 429."""
        from backend.main import _rate_counters
        # Clear counters for test isolation
        _rate_counters.clear()

        # Generate 10 scan sessions
        scan_ids = []
        for _ in range(10):
            with open(malicious_pt_direct, "rb") as f:
                resp = client.post("/scan", files={"file": ("m.pt", f, "application/octet-stream")})
            if resp.json().get("disarmament"):
                scan_ids.append(resp.json()["disarmament"]["scan_id"])

        # Each disarm call should succeed (not 429)
        for scan_id in scan_ids:
            r = client.post("/disarm", data={"scan_id": scan_id, "confirm": "true"})
            assert r.status_code != 429, f"Rate limited too early on scan_id {scan_id}"

    def test_rate_limit_blocks_11th_request(self, malicious_pt_direct):
        """The 11th disarm call from the same IP must return 429."""
        from backend.main import _rate_counters
        _rate_counters.clear()

        # Fill up 10 slots manually in the rate counter
        import collections
        import time
        _rate_counters["testclient"] = collections.deque([time.time()] * 10)

        # Get a valid scan_id
        with open(malicious_pt_direct, "rb") as f:
            scan_resp = client.post("/scan", files={"file": ("m.pt", f, "application/octet-stream")})
        scan_id = scan_resp.json()["disarmament"]["scan_id"]

        r = client.post("/disarm", data={"scan_id": scan_id, "confirm": "true"})
        assert r.status_code == 429
        assert "Retry-After" in r.headers


# ===========================================================================
# 7. Edge Case Tests
# ===========================================================================

class TestEdgeCases:

    def test_zero_weight_model_does_not_crash(self, tmp_dir):
        """A model with all-zero weights must disarm without error."""
        path = os.path.join(tmp_dir, "zeros.pt")
        torch.save({"w": torch.zeros(10, 10)}, path)
        result = disarm_lsb(path, x_bits=8)
        assert os.path.exists(result.cleaned_filepath)
        os.remove(result.cleaned_filepath)

    def test_non_float32_tensors_unchanged(self, tmp_dir):
        """int8 / bool tensors must not be modified during LSB disarmament."""
        path = os.path.join(tmp_dir, "mixed.pt")
        int8_tensor = torch.randint(0, 127, (4, 4), dtype=torch.int8)
        fp32_tensor = torch.randn(4, 4, dtype=torch.float32)
        torch.save({"int8": int8_tensor, "fp32": fp32_tensor}, path)

        result = disarm_lsb(path, x_bits=8)
        cleaned = torch.load(result.cleaned_filepath, map_location="cpu", weights_only=True)
        assert torch.equal(cleaned["int8"], int8_tensor), "int8 tensor must be unchanged"
        os.remove(result.cleaned_filepath)

    def test_composite_threat_applies_both_phases(self, malicious_pt_direct):
        """A file flagged by both L1 and L3a should apply both opcode and lsb phases."""
        engine = DisarmamentEngine()
        scan_result = {
            "layer1":  {"verdict": True},
            "layer2":  {"verdict": True},
            "layer3a": {"verdict": True},
            "layer3b": {"verdict": False},
        }
        report = engine.disarm(malicious_pt_direct, scan_result)
        # Both phases should be present
        assert "opcode" in report.phases_applied
        assert "lsb" in report.phases_applied
        os.remove(report.cleaned_filepath)

    def test_disarm_large_file_no_oom(self, tmp_dir):
        """A moderately large model (>50MB) must disarm without OOM."""
        path = os.path.join(tmp_dir, "large.pt")
        # ~50MB: 12M float32 weights
        torch.save({"w": torch.randn(3000, 4000)}, path)
        result = disarm_lsb(path, x_bits=8)
        assert result.weights_modified > 0
        os.remove(result.cleaned_filepath)

    def test_session_file_deleted_after_disarm_api(self, malicious_pt_direct):
        """After a successful /disarm, the session should be gone."""
        with open(malicious_pt_direct, "rb") as f:
            scan_resp = client.post("/scan", files={"file": ("m.pt", f, "application/octet-stream")})
        scan_id = scan_resp.json()["disarmament"]["scan_id"]

        client.post("/disarm", data={"scan_id": scan_id, "confirm": "true"})

        # Second call must 404
        second = client.post("/disarm", data={"scan_id": scan_id, "confirm": "true"})
        assert second.status_code == 404

    def test_disarm_returns_different_content_than_original(self, malicious_pt_direct):
        """The disarmed file bytes must differ from the original."""
        original_bytes = Path(malicious_pt_direct).read_bytes()

        with open(malicious_pt_direct, "rb") as f:
            scan_resp = client.post("/scan", files={"file": ("m.pt", f, "application/octet-stream")})
        scan_id = scan_resp.json()["disarmament"]["scan_id"]

        disarm_resp = client.post("/disarm", data={"scan_id": scan_id, "confirm": "true"})
        assert disarm_resp.content != original_bytes
