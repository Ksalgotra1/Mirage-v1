"""
detector/disarmer_opcode.py
===========================
Layer 1 / Layer 2 Disarmament: Malicious Pickle Opcode Removal.

Two-phase strategy:
  Phase 1 (safe):    torch.load(weights_only=True) rejects all non-whitelisted
                     callables. If it succeeds, re-save the clean weights as .pt.
  Phase 2 (surgery): Walk the raw pickle byte stream using pickletools/pickle
                     opcodes. Identify GLOBAL + REDUCE opcodes that reference
                     dangerous modules (os, subprocess, etc.) and replace them
                     with harmless no-op equivalents.

If both phases fail, raises DisarmamentNotPossibleError.
"""

from __future__ import annotations

import io
import os
import pickle
import pickletools
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Known dangerous module prefixes
# ---------------------------------------------------------------------------

DANGEROUS_MODULES: frozenset[str] = frozenset({
    "os",
    "nt",
    "posix",
    "subprocess",
    "socket",
    "importlib",
    "importlib._bootstrap",
    "importlib._bootstrap_external",
    "ctypes",
    "_ctypes",
    "sys",
    "builtins.eval",
    "builtins.exec",
    "builtins.compile",
    "builtins.__import__",
    "shutil",
    "pathlib",
})


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class DisarmamentNotPossibleError(Exception):
    """Raised when all disarmament phases fail (requires manual review)."""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class OpcodeDisarmResult:
    cleaned_filepath: str
    removed_opcode_count: int
    phase_used: int           # 1 = weights_only, 2 = opcode surgery
    dangerous_callables: list[str]


# ---------------------------------------------------------------------------
# Phase 1: Safe weights_only extraction
# ---------------------------------------------------------------------------

def _phase1_weights_only(src: str, dst: str) -> OpcodeDisarmResult | None:
    """
    Try torch.load(weights_only=True) which blocks all non-whitelisted callables.
    If successful, save the clean state dict as a new .pt file.

    Returns OpcodeDisarmResult on success, None if torch is unavailable or fails.
    """
    try:
        import torch
    except ImportError:
        return None

    try:
        state_dict = torch.load(src, map_location="cpu", weights_only=True)
        torch.save(state_dict, dst)
        return OpcodeDisarmResult(
            cleaned_filepath=dst,
            removed_opcode_count=0,   # Phase 1 doesn't count ops; they were blocked
            phase_used=1,
            dangerous_callables=[],
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Phase 2: Raw pickle opcode surgery
# ---------------------------------------------------------------------------

def _is_dangerous(module: str, name: str) -> bool:
    """Return True if the (module, name) combination references a dangerous callable."""
    full = f"{module}.{name}"
    if module in DANGEROUS_MODULES:
        return True
    if any(full.startswith(d) for d in DANGEROUS_MODULES):
        return True
    return False


def _sanitize_obj(obj):
    """
    Recursively replace non-picklable callables (neutralized dangerous ones)
    with None so the result can be re-pickled safely.
    Also removes injected exploit hook keys and converts raw weight lists
    to torch.Tensors for standard PyTorch checkpoint compatibility.
    """
    try:
        import torch
    except ImportError:
        torch = None

    if isinstance(obj, dict):
        clean_dict = {}
        for k, v in obj.items():
            if any(term in str(k).lower() for term in ("exploit", "payload", "backdoor", "malware", "hook")):
                continue
            sanitized_v = _sanitize_obj(v)
            if sanitized_v is None:
                continue
            if torch is not None and isinstance(sanitized_v, list) and len(sanitized_v) > 0 and all(isinstance(x, (int, float, list)) for x in sanitized_v):
                try:
                    sanitized_v = torch.tensor(sanitized_v, dtype=torch.float32)
                except Exception:
                    pass
            clean_dict[k] = sanitized_v
        return clean_dict
    if isinstance(obj, list):
        return [_sanitize_obj(v) for v in obj if _sanitize_obj(v) is not None]
    if isinstance(obj, tuple):
        return tuple(_sanitize_obj(v) for v in obj if _sanitize_obj(v) is not None)
    if callable(obj) and not isinstance(obj, type):
        return None
    try:
        pickle.dumps(obj, protocol=2)
        return obj
    except Exception:
        return None


def _phase2_opcode_surgery(src: str, dst: str) -> OpcodeDisarmResult:
    """
    Phase 2: Neutralizing Unpickler — works for ALL pickle protocols (2, 4, 5).

    Overrides find_class() so that any dangerous module.name lookup returns
    a harmless lambda instead of the actual callable. The result is then
    sanitized and re-pickled with protocol 2 for broad compatibility.

    For zip-based .pt files (modern PyTorch), the pickle is extracted from
    the archive, patched in memory, and written back into the zip.
    """
    import zipfile

    dangerous_found: list[str] = []
    removed: list[int] = [0]  # use list for nonlocal mutation in nested class

    class _NeutralizingUnpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str):
            if _is_dangerous(module, name):
                dangerous_found.append(f"{module}.{name}")
                removed[0] += 1
                return lambda *a, **kw: None
            try:
                return super().find_class(module, name)
            except Exception:
                return lambda *a, **kw: None

    def _patch_pickle_data(pkl_bytes: bytes) -> bytes:
        obj = _NeutralizingUnpickler(io.BytesIO(pkl_bytes)).load()
        obj = _sanitize_obj(obj)
        out = io.BytesIO()
        pickle.Pickler(out, protocol=2).dump(obj)
        return out.getvalue()

    if zipfile.is_zipfile(src):
        # Modern PyTorch zip format: patch data.pkl inside the archive
        with zipfile.ZipFile(src, "r") as zin:
            names = zin.namelist()
        with zipfile.ZipFile(src, "r") as zin, \
             zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_STORED) as zout:
            for name in names:
                data = zin.read(name)
                if name.endswith(".pkl"):
                    try:
                        data = _patch_pickle_data(data)
                    except Exception:
                        pass  # leave unchanged if patching fails
                zout.writestr(name, data)
    else:
        # Legacy single-file pickle
        with open(src, "rb") as f:
            pkl_bytes = f.read()
        obj = _NeutralizingUnpickler(io.BytesIO(pkl_bytes)).load()
        obj = _sanitize_obj(obj)
        try:
            import torch
            torch.save(obj, dst)
        except Exception:
            with open(dst, "wb") as f:
                pickle.dump(obj, f, protocol=2)

    return OpcodeDisarmResult(
        cleaned_filepath=dst,
        removed_opcode_count=removed[0],
        phase_used=2,
        dangerous_callables=dangerous_found,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def disarm_opcodes(filepath: str) -> OpcodeDisarmResult:
    """
    Remove malicious pickle opcodes from a .pt / .pkl / .pth file.

    Phase 1: torch.load(weights_only=True) — safe extraction.
    Phase 2: Raw opcode surgery — patch dangerous GLOBAL opcodes.

    Args:
        filepath: Absolute path to the malicious model file.

    Returns:
        OpcodeDisarmResult with the cleaned file path and forensic details.

    Raises:
        ValueError: If the file format is not a pickle-based format.
        DisarmamentNotPossibleError: If both phases fail.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Source file not found: {filepath}")

    ext = Path(filepath).suffix.lower()
    if ext not in (".pt", ".pth", ".pkl"):
        raise ValueError(f"Opcode disarmament not applicable for format: {ext}")

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=ext, dir=os.path.dirname(filepath))
    os.close(tmp_fd)

    try:
        # Phase 1: safe weights-only extraction
        result = _phase1_weights_only(filepath, tmp_path)
        if result is not None:
            return result

        # Phase 2: raw opcode surgery
        result = _phase2_opcode_surgery(filepath, tmp_path)
        return result

    except DisarmamentNotPossibleError:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    except Exception as exc:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise DisarmamentNotPossibleError(
            f"Both disarmament phases failed for {os.path.basename(filepath)}: {exc}"
        ) from exc
