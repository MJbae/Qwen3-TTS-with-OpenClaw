from __future__ import annotations

import fcntl
from pathlib import Path
from typing import TextIO

from .errors import ErrorCode, QTTSError


class SingleWorkerLock:
    def __init__(self, lock_path: Path):
        self.lock_path = lock_path
        self._fp: TextIO | None = None

    def acquire(self) -> None:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        fp = self.lock_path.open("a+")
        try:
            fcntl.flock(fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            fp.close()
            raise QTTSError(
                ErrorCode.QUEUE_LOCKED,
                f"another worker is already running for lock: {self.lock_path}",
                retryable=False,
                cause=exc,
            ) from exc
        self._fp = fp

    def release(self) -> None:
        if self._fp is None:
            return
        fcntl.flock(self._fp.fileno(), fcntl.LOCK_UN)
        self._fp.close()
        self._fp = None

    def __enter__(self) -> "SingleWorkerLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
