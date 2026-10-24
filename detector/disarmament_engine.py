"""
detector/disarmament_engine.py
===============================
Disarmament orchestrator.

Determines which disarmers to apply based on the cached scan result,
runs them in the correct order (opcode first, then LSB), verifies the
cleaned file re-scans as CLEAN, and writes an entry to the audit log.

Disarmament order:
  1. L1/L2 flagged  → disarm_opcodes (removes dangerous callables)
  2. L3a flagged     → disarm_lsb     (zeros LSB payload)
  3. Verify          → full re-scan must return verdict != MALICIOUS
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

# Ensure project root is on the path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from detector.disarmer_lsb import disarm_lsb, LsbDisarmResult
from detector.disarmer_opcode import disarm_opcodes, OpcodeDisarmResult, DisarmamentNotPossibleError


# ---------------------------------------------------------------------------
# Audit logging
# ---------------------------------------------------------------------------

_LOGS_DIR = _PROJECT_ROOT / "logs"
_LOGS_DIR.mkdir(exist_ok=True)
_AUDIT_LOG_PATH = _LOGS_DIR / "disarmament_audit.log"

_audit_logger = logging.getLogger("disarmament_audit")
if not _audit_logger.handlers:
    _handler = logging.FileHandler(str(_AUDIT_LOG_PATH), encoding="utf-8")
    _handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ"))
    _audit_logger.addHandler(_handler)
    _audit_logger.setLevel(logging.INFO)
    _audit_logger.propagate = False


def _audit(event: str, data: dict) -> None:
    """Write a JSON audit log entry."""
    _audit_logger.info(json.dumps({"event": event, **data}))


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class DisarmamentVerificationError(Exception):
    """Raised when the cleaned file fails the post-disarmament re-scan."""


# Re-export so callers only need to import from this module
__all__ = [
    "DisarmamentEngine",
    "DisarmamentReport",
    "DisarmamentVerificationError",
    "DisarmamentNotPossibleError",
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class DisarmamentReport:
    cleaned_filepath: str
    original_sha256: str
    cleaned_sha256: str
    phases_applied: list[str]                  # e.g. ["opcode", "lsb"]

    # L3a forensics (None if L3a was not involved)
    payload_sha256: str | None = None
    payload_size_bytes: int | None = None

    # L1/L2 forensics (None if opcode disarmament was not involved)
    removed_opcode_count: int | None = None
    dangerous_callables: list[str] | None = None
    opcode_phase_used: int | None = None       # 1 or 2

    # Verification
    verification_result: dict = field(default_factory=dict)
    verification_passed: bool = False


# ---------------------------------------------------------------------------
# Helper: SHA-256 of a file
# ---------------------------------------------------------------------------

def _file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Helper: determine which layers flagged
# ---------------------------------------------------------------------------

def _layer_flagged(scan_result: dict, layer_key: str) -> bool:
    layer_data = scan_result.get(layer_key, {})
    if isinstance(layer_data, dict):
        return bool(layer_data.get("verdict", False))
    return False


# ---------------------------------------------------------------------------
# Disarmament Engine
# ---------------------------------------------------------------------------

class DisarmamentEngine:
    """
    Orchestrates multi-layer disarmament: opcode first, then LSB, then verify.

    Usage:
        engine = DisarmamentEngine()
        report = engine.disarm(filepath, scan_result)
    """

    def disarm(self, filepath: str, scan_result: dict) -> DisarmamentReport:
        """
        Run all required disarmament phases and verify the result.

        Args:
            filepath:    Path to the malicious model file (from session store).
            scan_result: The original scan response dict (from POST /scan).

        Returns:
            DisarmamentReport with cleaned file path and forensic evidence.

        Raises:
            DisarmamentNotPossibleError: If the file cannot be disarmed.
            DisarmamentVerificationError: If the cleaned file still scans as MALICIOUS.
        """
        original_sha256 = _file_sha256(filepath)
        ext = Path(filepath).suffix.lower()

        _audit("disarmament_started", {
            "filepath": os.path.basename(filepath),
            "original_sha256": original_sha256,
            "ext": ext,
        })

        # Determine which phases are needed
        needs_opcode = (
            _layer_flagged(scan_result, "layer1") or
            _layer_flagged(scan_result, "layer2")
        ) and ext in (".pt", ".pth", ".pkl")

        needs_lsb = (
            _layer_flagged(scan_result, "layer3a") or
            ext in (".safetensors",)  # always strip LSB from safetensors if flagged
        )

        # If nothing was specifically flagged, apply a conservative full disarm
        if not needs_opcode and not needs_lsb:
            if ext in (".pt", ".pth", ".pkl"):
                needs_opcode = True
            needs_lsb = True

        phases_applied: list[str] = []
        current_filepath = filepath

        # Track intermediate temp files for cleanup
        intermediate_files: list[str] = []

        opcode_result: OpcodeDisarmResult | None = None
        lsb_result: LsbDisarmResult | None = None

        try:
            # ------------------------------------------------------------------
            # Phase 1: Opcode disarmament (L1/L2)
            # ------------------------------------------------------------------
            if needs_opcode:
                opcode_result = disarm_opcodes(current_filepath)
                intermediate_files.append(opcode_result.cleaned_filepath)
                current_filepath = opcode_result.cleaned_filepath
                phases_applied.append("opcode")

                _audit("opcode_phase_complete", {
                    "original_sha256": original_sha256,
                    "phase_used": opcode_result.phase_used,
                    "removed_opcode_count": opcode_result.removed_opcode_count,
                    "dangerous_callables": opcode_result.dangerous_callables,
                })

            # ------------------------------------------------------------------
            # Phase 2: LSB disarmament (L3a)
            # ------------------------------------------------------------------
            if needs_lsb and ext in (".pt", ".pth", ".pkl", ".safetensors", ".h5", ".hdf5"):
                lsb_result = disarm_lsb(current_filepath, x_bits=8)
                intermediate_files.append(lsb_result.cleaned_filepath)
                current_filepath = lsb_result.cleaned_filepath
                phases_applied.append("lsb")

                _audit("lsb_phase_complete", {
                    "original_sha256": original_sha256,
                    "payload_sha256": lsb_result.payload_sha256,
                    "payload_size_bytes": lsb_result.payload_size_bytes,
                    "tensors_processed": lsb_result.tensors_processed,
                    "weights_modified": lsb_result.weights_modified,
                })

            # ------------------------------------------------------------------
            # Finalize: rename the last intermediate to a stable "cleaned" path
            # ------------------------------------------------------------------
            cleaned_path = filepath.replace(ext, f"_disarmed{ext}")
            # Avoid collision if cleaned_path == filepath
            if os.path.abspath(cleaned_path) == os.path.abspath(filepath):
                cleaned_path = filepath.replace(ext, f"_disarmed_{int(time.time())}{ext}")

            shutil.copy2(current_filepath, cleaned_path)
            cleaned_sha256 = _file_sha256(cleaned_path)

            # ------------------------------------------------------------------
            # Verification: re-scan the cleaned file
            # ------------------------------------------------------------------
            verification_result = self._verify(cleaned_path)
            verification_passed = (
                verification_result.get("fused", {}).get("verdict", "MALICIOUS") != "MALICIOUS"
            )

            _audit("verification_complete", {
                "original_sha256": original_sha256,
                "cleaned_sha256": cleaned_sha256,
                "phases_applied": phases_applied,
                "verification_passed": verification_passed,
                "fused_verdict": verification_result.get("fused", {}).get("verdict", "UNKNOWN"),
            })

            if not verification_passed:
                os.remove(cleaned_path)
                raise DisarmamentVerificationError(
                    f"Cleaned file still scans as "
                    f"{verification_result.get('fused', {}).get('verdict', 'MALICIOUS')}. "
                    f"Manual review required."
                )

            report = DisarmamentReport(
                cleaned_filepath=cleaned_path,
                original_sha256=original_sha256,
                cleaned_sha256=cleaned_sha256,
                phases_applied=phases_applied,
                payload_sha256=lsb_result.payload_sha256 if lsb_result else None,
                payload_size_bytes=lsb_result.payload_size_bytes if lsb_result else None,
                removed_opcode_count=opcode_result.removed_opcode_count if opcode_result else None,
                dangerous_callables=opcode_result.dangerous_callables if opcode_result else None,
                opcode_phase_used=opcode_result.phase_used if opcode_result else None,
                verification_result=verification_result,
                verification_passed=verification_passed,
            )
            return report

        finally:
            # Clean up intermediate temp files (not the final cleaned_path)
            for tmp in intermediate_files:
                if os.path.exists(tmp) and os.path.abspath(tmp) != os.path.abspath(
                    current_filepath if "cleaned_path" not in dir() else cleaned_path
                ):
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass

    def _verify(self, cleaned_filepath: str) -> dict:
        """Run the full detection pipeline on the cleaned file."""
        try:
            from detector.layer1_modelscan import Layer1ModelScan
            from detector.layer2_classifier import Layer2Classifier
            from backend.fusion import FusionEngine

            ext = Path(cleaned_filepath).suffix.lower()

            l1 = Layer1ModelScan().scan(cleaned_filepath)
        except Exception as e:
            l1 = {"layer": 1, "verdict": False, "issues": [], "coverage_note": str(e)}

        try:
            from detector.layer2_classifier import Layer2Classifier
            l2 = Layer2Classifier().predict(cleaned_filepath)
        except Exception as e:
            l2 = {"layer": 2, "verdict": False, "probability": 0.0, "coverage_note": str(e)}

        try:
            from detector.layer3a_gf import Layer3aGFDetector
            l3a = Layer3aGFDetector().predict(cleaned_filepath)
        except Exception as e:
            l3a = {"layer": "3a", "verdict": False, "score": 0.0, "coverage_note": str(e)}

        try:
            from detector.layer3b_weights import scan as l3b_scan
            l3b = l3b_scan(cleaned_filepath)
        except Exception as e:
            l3b = {"layer": "3b", "verdict": False, "score": 0.0, "coverage_note": str(e)}

        try:
            from backend.fusion import FusionEngine
            ext = Path(cleaned_filepath).suffix.lower()
            fused = FusionEngine().fuse(ext, l1, l2, l3a, l3b)
        except Exception as e:
            fused = {"verdict": "UNKNOWN", "error": str(e)}

        return {"layer1": l1, "layer2": l2, "layer3a": l3a, "layer3b": l3b, "fused": fused}


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
disarmament_engine = DisarmamentEngine()
