from pathlib import Path

from qtts.queue_store import SQLiteJobQueue


def test_queue_submit_claim_complete(tmp_path: Path):
    queue = SQLiteJobQueue(tmp_path / "queue.sqlite3")

    job_id = queue.submit_job(
        voice_id="voice1",
        text="hello",
        language="Auto",
        seed=None,
        max_new_tokens=256,
        max_attempts=2,
    )

    claimed = queue.claim_pending_job("worker-1")
    assert claimed is not None
    assert claimed["id"] == job_id
    assert claimed["status"] == "running"

    queue.mark_completed(job_id, str(tmp_path / "out.wav"))
    job = queue.get_job(job_id)
    assert job["status"] == "completed"


def test_queue_retry_then_fail(tmp_path: Path):
    queue = SQLiteJobQueue(tmp_path / "queue.sqlite3")

    job_id = queue.submit_job(
        voice_id="voice1",
        text="hello",
        language="Auto",
        seed=None,
        max_new_tokens=256,
        max_attempts=2,
    )

    claimed = queue.claim_pending_job("worker-1")
    assert claimed is not None

    state = queue.mark_failure(
        job_id,
        error_code="SYNTHESIS_FAIL",
        error_message="temporary",
        retryable=True,
    )
    assert state["status"] == "pending"

    claimed = queue.claim_pending_job("worker-1")
    assert claimed is not None

    state = queue.mark_failure(
        job_id,
        error_code="SYNTHESIS_FAIL",
        error_message="final",
        retryable=True,
    )
    assert state["status"] == "failed"
