"""In-memory scan jobs and SSE-safe event buffering."""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScanJob:
    scan_id: str
    filename: str
    filepath: str
    created_at: float = field(default_factory=time.time)
    events: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=200))
    next_event_id: int = 1
    terminal: bool = False
    result: dict[str, Any] | None = None
    condition: threading.Condition = field(default_factory=threading.Condition)

    def emit(self, stage: str, status: str, message: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        with self.condition:
            event = {
                "id": self.next_event_id,
                "timestamp": round(time.time(), 3),
                "stage": stage,
                "status": status,
                "message": message,
                "metadata": metadata or {},
            }
            self.next_event_id += 1
            self.events.append(event)
            self.condition.notify_all()
            return event

    def finish(self, result: dict[str, Any]) -> None:
        with self.condition:
            self.result = result
            self.terminal = True
            self.condition.notify_all()


class ScanJobStore:
    """Small, thread-safe job registry. Jobs expire after one hour."""

    def __init__(self, ttl_seconds: int = 3600) -> None:
        self.ttl_seconds = ttl_seconds
        self._jobs: dict[str, ScanJob] = {}
        self._lock = threading.Lock()

    def create(self, filename: str, filepath: str) -> ScanJob:
        self.purge_expired()
        job = ScanJob(scan_id=str(uuid.uuid4()), filename=filename, filepath=filepath)
        with self._lock:
            self._jobs[job.scan_id] = job
        return job

    def get(self, scan_id: str) -> ScanJob | None:
        self.purge_expired()
        with self._lock:
            return self._jobs.get(scan_id)

    def purge_expired(self) -> None:
        cutoff = time.time() - self.ttl_seconds
        with self._lock:
            expired = [key for key, job in self._jobs.items() if job.created_at < cutoff]
            for key in expired:
                self._jobs.pop(key, None)


scan_job_store = ScanJobStore()
