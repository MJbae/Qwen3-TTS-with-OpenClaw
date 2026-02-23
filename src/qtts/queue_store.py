from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import ErrorCode, QTTSError


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteJobQueue:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    voice_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    language TEXT NOT NULL DEFAULT 'Auto',
                    seed INTEGER,
                    max_new_tokens INTEGER,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL DEFAULT 2,
                    worker_id TEXT,
                    error_code TEXT,
                    error_message TEXT,
                    output_path TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_jobs_status_created
                ON jobs(status, created_at)
                """
            )

    def submit_job(
        self,
        *,
        voice_id: str,
        text: str,
        language: str,
        seed: int | None,
        max_new_tokens: int,
        max_attempts: int,
    ) -> str:
        text = text.strip()
        if not text:
            raise QTTSError(ErrorCode.INVALID_INPUT, "job text cannot be empty")

        job_id = uuid.uuid4().hex
        now = utc_now_iso()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs(
                    id, voice_id, text, language, seed, max_new_tokens,
                    status, attempts, max_attempts, created_at, updated_at
                ) VALUES(?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?, ?)
                """,
                (
                    job_id,
                    voice_id,
                    text,
                    language,
                    seed,
                    max_new_tokens,
                    max_attempts,
                    now,
                    now,
                ),
            )

        return job_id

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()

        if row is None:
            raise QTTSError(ErrorCode.JOB_NOT_FOUND, f"job-id not found: {job_id}")

        return dict(row)

    def claim_pending_job(self, worker_id: str) -> dict[str, Any] | None:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT id FROM jobs WHERE status = 'pending' ORDER BY created_at LIMIT 1"
            ).fetchone()
            if row is None:
                conn.execute("COMMIT")
                return None

            job_id = row["id"]
            now = utc_now_iso()
            updated = conn.execute(
                """
                UPDATE jobs
                SET status = 'running', worker_id = ?, started_at = ?, updated_at = ?
                WHERE id = ? AND status = 'pending'
                """,
                (worker_id, now, now, job_id),
            )
            if updated.rowcount != 1:
                conn.execute("ROLLBACK")
                return None

            claimed = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            conn.execute("COMMIT")
            return dict(claimed) if claimed is not None else None
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def mark_completed(self, job_id: str, output_path: str) -> None:
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = 'completed', output_path = ?, finished_at = ?,
                    updated_at = ?, error_code = NULL, error_message = NULL
                WHERE id = ?
                """,
                (output_path, now, now, job_id),
            )

    def mark_failure(
        self,
        job_id: str,
        *,
        error_code: str,
        error_message: str,
        retryable: bool,
    ) -> dict[str, Any]:
        job = self.get_job(job_id)

        attempts = int(job.get("attempts", 0)) + 1
        max_attempts = int(job.get("max_attempts", 2))
        can_retry = retryable and attempts < max_attempts

        now = utc_now_iso()
        status = "pending" if can_retry else "failed"

        with self._connect() as conn:
            if can_retry:
                conn.execute(
                    """
                    UPDATE jobs
                    SET status = 'pending', attempts = ?, error_code = ?,
                        error_message = ?, updated_at = ?, started_at = NULL,
                        worker_id = NULL
                    WHERE id = ?
                    """,
                    (attempts, error_code, error_message, now, job_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE jobs
                    SET status = 'failed', attempts = ?, error_code = ?,
                        error_message = ?, updated_at = ?, finished_at = ?,
                        worker_id = NULL
                    WHERE id = ?
                    """,
                    (attempts, error_code, error_message, now, now, job_id),
                )

        result = self.get_job(job_id)
        result["status"] = status
        return result
