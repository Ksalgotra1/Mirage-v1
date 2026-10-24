"""
backend/session_store.py
========================
In-memory scan session registry.

Sessions are created by POST /scan when a MALICIOUS verdict is returned.
They are consumed (and deleted) by POST /disarm.
Each session holds the path to the uploaded file and the full scan result.

TTL: 1 hour (configurable via SESSION_TTL_SECONDS).
Persistence: in-memory only (MVP — sessions are lost on server restart).
"""

from __future__ import annotations

import os
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

SESSION_TTL_SECONDS: int = 3600  # 1 hour

# Sessions directory: PROJECT_ROOT/sessions/
_SESSIONS_DIR = Path(__file__).resolve().parent.parent / "sessions"
_SESSIONS_DIR.mkdir(exist_ok=True)


@dataclass
class SessionEntry:
    scan_id: str
    filepath: str          # path inside _SESSIONS_DIR (NOT the original tmp path)
    scan_result: dict
    created_at: float = field(default_factory=time.time)
    ttl_seconds: int = SESSION_TTL_SECONDS

    @property
    def expires_at(self) -> float:
        return self.created_at + self.ttl_seconds

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at

    @property
    def seconds_remaining(self) -> int:
        remaining = self.expires_at - time.time()
        return max(0, int(remaining))


class ScanSessionStore:
    """Thread-safe in-memory session store for scan results awaiting disarmament."""

    def __init__(self) -> None:
        self._store: dict[str, SessionEntry] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, source_filepath: str, scan_result: dict) -> str:
        """
        Copy the source file into the sessions directory and register a session.

        Args:
            source_filepath: Path to the scanned temp file (still alive).
            scan_result:     The full scan response dict from POST /scan.

        Returns:
            scan_id (UUID4 string).
        """
        scan_id = str(uuid.uuid4())
        ext = os.path.splitext(source_filepath)[1]
        dest_path = str(_SESSIONS_DIR / f"{scan_id}{ext}")

        shutil.copy2(source_filepath, dest_path)

        entry = SessionEntry(
            scan_id=scan_id,
            filepath=dest_path,
            scan_result=scan_result,
        )
        with self._lock:
            self._store[scan_id] = entry

        return scan_id

    def get(self, scan_id: str) -> SessionEntry | None:
        """Return the session entry, or None if not found or expired."""
        with self._lock:
            entry = self._store.get(scan_id)
            if entry is None:
                return None
            if entry.is_expired:
                self._delete_locked(scan_id)
                return None
            return entry

    def delete(self, scan_id: str) -> bool:
        """Delete a session (and its stored file). Returns True if it existed."""
        with self._lock:
            return self._delete_locked(scan_id)

    def purge_expired(self) -> int:
        """Remove all expired sessions. Returns the number purged."""
        with self._lock:
            expired = [sid for sid, e in self._store.items() if e.is_expired]
            for sid in expired:
                self._delete_locked(sid)
            return len(expired)

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _delete_locked(self, scan_id: str) -> bool:
        """Must be called with self._lock held."""
        entry = self._store.pop(scan_id, None)
        if entry is None:
            return False
        try:
            if os.path.exists(entry.filepath):
                os.remove(entry.filepath)
        except OSError:
            pass  # best-effort cleanup
        return True


# ---------------------------------------------------------------------------
# Singleton — shared across the FastAPI application
# ---------------------------------------------------------------------------
session_store = ScanSessionStore()
