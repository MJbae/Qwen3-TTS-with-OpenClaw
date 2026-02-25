from __future__ import annotations

import argparse
import importlib
from importlib import metadata as importlib_metadata
import json
import os
import platform
import shutil
import socket
import sys
import time
from pathlib import Path
from typing import IO
from typing import Any

from .config import RuntimePaths, resolve_runtime_paths
from .constants import (
    DEFAULT_CPU_INTEROP_THREADS,
    DEFAULT_CPU_THREADS,
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    DEFAULT_LANGUAGE,
    DEFAULT_LOW_CPU_MEM_USAGE,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MODEL_ID,
    DEFAULT_SPLIT_MAX_CHARS,
    DEFAULT_TIMEOUT_SEC,
    QUEUE_DB_NAME,
)
from .engine import QwenTTSEngine
from .errors import ErrorCode, QTTSError, exit_code_for
from .logging_utils import setup_logging
from .queue_store import SQLiteJobQueue
from .service import SynthesisService
from .voice_store import VoiceStore
from .worker_lock import SingleWorkerLock


def print_json(payload: Any, stream: IO[str] | None = None) -> None:
    print(json.dumps(payload, ensure_ascii=True, indent=2), file=stream or sys.stdout)


def resolve_queue_db_path(queue_arg: str | None, runtime_paths: RuntimePaths) -> Path:
    if not queue_arg:
        return runtime_paths.queue_db

    raw = Path(queue_arg).expanduser()
    if raw.exists() and raw.is_dir():
        return (raw / QUEUE_DB_NAME).resolve()

    if raw.suffix == "" and str(queue_arg).endswith("/"):
        return (raw / QUEUE_DB_NAME).resolve()

    return raw.resolve()


def is_cpu_like_device(device: Any) -> bool:
    if isinstance(device, str):
        normalized = device.strip().lower()
        return normalized == "cpu" or normalized.startswith("cpu:")
    return False


def get_total_ram_gib() -> float | None:
    if os.name == "nt" or not hasattr(os, "sysconf"):
        return None
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
    except (OSError, ValueError, TypeError):
        return None

    if not isinstance(page_size, int) or not isinstance(phys_pages, int):
        return None
    if page_size <= 0 or phys_pages <= 0:
        return None
    return (page_size * phys_pages) / float(1024**3)


def build_service(args, voice_store: VoiceStore, logger) -> SynthesisService:
    engine = QwenTTSEngine(
        model_id=args.model_id,
        device=args.device,
        attn_implementation=args.attn_implementation,
        dtype=getattr(args, "dtype", DEFAULT_DTYPE),
        low_cpu_mem_usage=getattr(args, "low_cpu_mem_usage", DEFAULT_LOW_CPU_MEM_USAGE),
        cpu_threads=getattr(args, "cpu_threads", DEFAULT_CPU_THREADS),
        cpu_interop_threads=getattr(args, "cpu_interop_threads", DEFAULT_CPU_INTEROP_THREADS),
    )
    return SynthesisService(voice_store=voice_store, engine=engine, logger=logger)


def run_doctor(args, runtime_paths: RuntimePaths, logger) -> int:
    checks: list[dict[str, Any]] = []

    py_ok = sys.version_info >= (3, 10)
    checks.append(
        {
            "name": "python_version",
            "ok": py_ok,
            "detail": f"{sys.version.split()[0]} (need >= 3.10)",
        }
    )

    try:
        importlib.import_module("qwen_tts")
        try:
            qwen_tts_version = importlib_metadata.version("qwen-tts")
            checks.append(
                {
                    "name": "import_qwen_tts",
                    "ok": True,
                    "detail": f"ok (qwen-tts {qwen_tts_version})",
                }
            )
        except importlib_metadata.PackageNotFoundError:
            checks.append(
                {
                    "name": "import_qwen_tts",
                    "ok": False,
                    "detail": "qwen-tts package metadata not found",
                }
            )
        except Exception as exc:  # noqa: BLE001
            checks.append({"name": "import_qwen_tts", "ok": False, "detail": str(exc)})
    except Exception as exc:  # noqa: BLE001
        checks.append({"name": "import_qwen_tts", "ok": False, "detail": str(exc)})

    for module_name in ("torch", "torchaudio", "soundfile"):
        try:
            importlib.import_module(module_name)
            checks.append({"name": f"import_{module_name}", "ok": True, "detail": "ok"})
        except Exception as exc:  # noqa: BLE001
            checks.append({"name": f"import_{module_name}", "ok": False, "detail": str(exc)})

    sox_path = shutil.which("sox")
    checks.append(
        {
            "name": "command_sox",
            "ok": bool(sox_path),
            "detail": sox_path or "sox not found in PATH",
        }
    )

    total_ram_gib = get_total_ram_gib()
    if total_ram_gib is None:
        checks.append(
            {
                "name": "system_ram_gib",
                "ok": True,
                "detail": "unavailable",
            }
        )
    elif total_ram_gib < 12:
        checks.append(
            {
                "name": "system_ram_gib",
                "ok": True,
                "detail": f"warning: {total_ram_gib:.2f} GiB (<12 GiB recommended for 1.7B CPU)",
            }
        )
    else:
        checks.append(
            {
                "name": "system_ram_gib",
                "ok": True,
                "detail": f"{total_ram_gib:.2f} GiB",
            }
        )

    cpu_like_device = is_cpu_like_device(args.device)
    checks.append(
        {
            "name": "device_cpu_only",
            "ok": cpu_like_device,
            "detail": (
                f"{args.device!r} is cpu-like"
                if cpu_like_device
                else f"{args.device!r} is not cpu-like; this project targets CPU-only inference"
            ),
        }
    )

    runtime_paths.ensure()
    for label, path in (
        ("runtime/voices", runtime_paths.voices_dir),
        ("runtime/jobs", runtime_paths.jobs_dir),
        ("runtime/output", runtime_paths.output_dir),
        ("logs", runtime_paths.logs_dir),
    ):
        checks.append({"name": f"path_{label}", "ok": path.exists(), "detail": str(path)})

    queue_db = resolve_queue_db_path(args.queue, runtime_paths)
    try:
        SQLiteJobQueue(queue_db)
        checks.append({"name": "queue_db", "ok": True, "detail": str(queue_db)})
    except Exception as exc:  # noqa: BLE001
        checks.append({"name": "queue_db", "ok": False, "detail": str(exc)})

    smoke_result = None
    if args.run_smoke:
        if not args.voice_id:
            raise QTTSError(
                ErrorCode.INVALID_INPUT,
                "--run-smoke requires --voice-id",
            )

        voice_store = VoiceStore(runtime_paths.voices_dir)
        service = build_service(args, voice_store, logger)
        out_path = Path(args.out).expanduser() if args.out else runtime_paths.output_dir / "doctor_smoke.wav"
        smoke_result = service.speak_to_file(
            voice_id=args.voice_id,
            text=args.smoke_text,
            out_path=out_path,
            language=args.language,
            max_new_tokens=args.max_new_tokens,
            timeout_sec=args.timeout_sec,
            retries=args.retries,
            split_max_chars=args.split_max_chars,
        )
        checks.append({"name": "smoke_speak", "ok": True, "detail": str(out_path.resolve())})

    ok = all(item["ok"] for item in checks)
    payload = {
        "ok": ok,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
        },
        "checks": checks,
        "smoke": smoke_result,
    }
    print_json(payload)

    return 0 if ok else 1


def run_clone_create(args, runtime_paths: RuntimePaths, logger) -> int:
    voice_store = VoiceStore(runtime_paths.voices_dir)
    service = build_service(args, voice_store, logger)

    record, reused = service.clone_voice(
        voice_id=args.voice_id,
        ref_audio=Path(args.ref_audio).expanduser().resolve(),
        ref_text=args.ref_text,
        force=args.force,
        x_vector_only_mode=args.x_vector_only_mode,
    )

    print_json(
        {
            "voice_id": record.get("voice_id", args.voice_id),
            "prompt_path": record.get("_prompt_path"),
            "ref_audio_path": record.get("ref_audio_path"),
            "duration_sec": record.get("duration_sec"),
            "reused": reused,
        }
    )
    return 0


def run_speak(args, runtime_paths: RuntimePaths, logger) -> int:
    voice_store = VoiceStore(runtime_paths.voices_dir)
    service = build_service(args, voice_store, logger)

    result = service.speak_to_file(
        voice_id=args.voice_id,
        text=args.text,
        out_path=Path(args.out).expanduser().resolve(),
        language=args.language,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        timeout_sec=args.timeout_sec,
        retries=args.retries,
        split_max_chars=args.split_max_chars,
    )
    print_json(result)
    return 0


def run_batch(args, runtime_paths: RuntimePaths, logger) -> int:
    voice_store = VoiceStore(runtime_paths.voices_dir)
    service = build_service(args, voice_store, logger)

    summary = service.batch_to_dir(
        voice_id=args.voice_id,
        input_path=Path(args.input).expanduser().resolve(),
        out_dir=Path(args.out_dir).expanduser().resolve(),
        language=args.language,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        timeout_sec=args.timeout_sec,
        retries=args.retries,
        split_max_chars=args.split_max_chars,
        resume=not args.no_resume,
    )
    print_json(summary)
    if summary["failed"] > 0:
        return exit_code_for(QTTSError(ErrorCode.SYNTHESIS_FAIL, "batch had failed items"))
    return 0


def run_voices_list(_args, runtime_paths: RuntimePaths) -> int:
    voice_store = VoiceStore(runtime_paths.voices_dir)
    print_json({"voices": voice_store.list_voices()})
    return 0


def run_voices_delete(args, runtime_paths: RuntimePaths) -> int:
    voice_store = VoiceStore(runtime_paths.voices_dir)
    voice_store.delete_voice(args.voice_id)
    print_json({"deleted": args.voice_id})
    return 0


def run_job_submit(args, runtime_paths: RuntimePaths) -> int:
    queue_db = resolve_queue_db_path(args.queue, runtime_paths)
    queue = SQLiteJobQueue(queue_db)

    voice_store = VoiceStore(runtime_paths.voices_dir)
    voice_store.load_voice(args.voice_id)

    job_id = queue.submit_job(
        voice_id=args.voice_id,
        text=args.text,
        language=args.language,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        max_attempts=args.max_attempts,
    )
    print_json({"job_id": job_id, "queue": str(queue_db)})
    return 0


def run_job_status(args, runtime_paths: RuntimePaths) -> int:
    queue_db = resolve_queue_db_path(args.queue, runtime_paths)
    queue = SQLiteJobQueue(queue_db)
    job = queue.get_job(args.job_id)
    print_json(job)
    return 0


def run_job_fetch(args, runtime_paths: RuntimePaths) -> int:
    queue_db = resolve_queue_db_path(args.queue, runtime_paths)
    queue = SQLiteJobQueue(queue_db)
    job = queue.get_job(args.job_id)

    if job.get("status") != "completed":
        raise QTTSError(
            ErrorCode.JOB_NOT_READY,
            f"job is not completed: status={job.get('status')}",
        )

    output_path = job.get("output_path")
    if not output_path:
        raise QTTSError(ErrorCode.JOB_NOT_READY, "job has no output path")

    src = Path(output_path)
    if not src.exists():
        raise QTTSError(
            ErrorCode.JOB_NOT_READY,
            f"job output does not exist: {src}",
        )

    dest = Path(args.out).expanduser().resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)

    print_json({"job_id": args.job_id, "out": str(dest), "source": str(src.resolve())})
    return 0


def run_serve_start(args, runtime_paths: RuntimePaths, logger) -> int:
    queue_db = resolve_queue_db_path(args.queue, runtime_paths)
    queue = SQLiteJobQueue(queue_db)

    voice_store = VoiceStore(runtime_paths.voices_dir)
    service = build_service(args, voice_store, logger)

    lock_path = queue_db.with_suffix(queue_db.suffix + ".lock")
    worker_id = f"{socket.gethostname()}-{os.getpid()}"

    print_json(
        {
            "event": "worker_start",
            "worker_id": worker_id,
            "queue": str(queue_db),
            "lock": str(lock_path),
            "poll_interval_sec": args.poll_interval_sec,
            "runtime_output": str(runtime_paths.output_dir),
        }
    )

    with SingleWorkerLock(lock_path):
        try:
            while True:
                job = queue.claim_pending_job(worker_id)
                if job is None:
                    time.sleep(args.poll_interval_sec)
                    continue

                job_id = str(job["id"])
                out_path = (runtime_paths.output_dir / f"{job_id}.wav").resolve()

                try:
                    result = service.speak_to_file(
                        voice_id=str(job["voice_id"]),
                        text=str(job["text"]),
                        out_path=out_path,
                        language=str(job.get("language") or DEFAULT_LANGUAGE),
                        seed=job.get("seed"),
                        max_new_tokens=int(job.get("max_new_tokens") or DEFAULT_MAX_NEW_TOKENS),
                        timeout_sec=args.timeout_sec,
                        retries=args.retries,
                        split_max_chars=args.split_max_chars,
                    )
                    queue.mark_completed(job_id, result["out"])
                    logger.info(
                        "job completed",
                        extra={"event": "job_completed", "job_id": job_id, "voice_id": job["voice_id"]},
                    )
                except QTTSError as exc:
                    queue.mark_failure(
                        job_id,
                        error_code=exc.code.value,
                        error_message=exc.message,
                        retryable=exc.retryable,
                    )
                    logger.error(
                        "job failed",
                        extra={
                            "event": "job_failed",
                            "job_id": job_id,
                            "voice_id": job["voice_id"],
                            "error_code": exc.code.value,
                        },
                    )
                except Exception as exc:  # noqa: BLE001
                    queue.mark_failure(
                        job_id,
                        error_code=ErrorCode.INTERNAL_ERROR.value,
                        error_message=str(exc),
                        retryable=False,
                    )
                    logger.exception(
                        "job failed with unhandled exception",
                        extra={
                            "event": "job_failed_unhandled",
                            "job_id": job_id,
                            "voice_id": job["voice_id"],
                            "error_code": ErrorCode.INTERNAL_ERROR.value,
                        },
                    )
        except KeyboardInterrupt:
            print_json({"event": "worker_stop", "worker_id": worker_id})

    return 0


def add_common_synthesis_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--language", default=DEFAULT_LANGUAGE)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--timeout-sec", type=int, default=DEFAULT_TIMEOUT_SEC)
    parser.add_argument("--retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--split-max-chars", type=int, default=DEFAULT_SPLIT_MAX_CHARS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qtts", description="Local Qwen3-TTS CLI")
    parser.add_argument("--runtime-root", default="runtime", help="runtime root directory")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Qwen3-TTS model id/path")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="inference device map (default: cpu)")
    parser.add_argument(
        "--dtype",
        default=DEFAULT_DTYPE,
        choices=("float32", "bfloat16", "float16"),
        help="model dtype for load/generation",
    )
    parser.add_argument(
        "--low-cpu-mem-usage",
        dest="low_cpu_mem_usage",
        action="store_true",
        default=DEFAULT_LOW_CPU_MEM_USAGE,
        help="use low CPU memory loading mode",
    )
    parser.add_argument(
        "--no-low-cpu-mem-usage",
        dest="low_cpu_mem_usage",
        action="store_false",
        help="disable low CPU memory loading mode",
    )
    parser.add_argument("--cpu-threads", type=int, default=DEFAULT_CPU_THREADS, help="torch CPU threads")
    parser.add_argument(
        "--cpu-interop-threads",
        type=int,
        default=DEFAULT_CPU_INTEROP_THREADS,
        help="torch CPU interop threads",
    )
    parser.add_argument("--attn-implementation", default=None, help="optional attention implementation")
    parser.add_argument("--verbose", action="store_true")

    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="validate runtime and dependencies")
    doctor.add_argument("--queue", default=None, help="queue sqlite path or directory")
    doctor.add_argument("--run-smoke", action="store_true", help="run speech smoke test")
    doctor.add_argument("--voice-id", default=None, help="required when --run-smoke is set")
    doctor.add_argument("--smoke-text", default="Doctor smoke test sentence.")
    doctor.add_argument("--out", default=None, help="smoke output wav path")
    add_common_synthesis_args(doctor)

    clone = subparsers.add_parser("clone", help="voice clone operations")
    clone_sub = clone.add_subparsers(dest="clone_command", required=True)

    clone_create = clone_sub.add_parser("create", help="create/reuse voice clone")
    clone_create.add_argument("--voice-id", required=True)
    clone_create.add_argument("--ref-audio", required=True)
    clone_create.add_argument("--ref-text", required=True)
    clone_create.add_argument("--force", action="store_true")
    clone_create.add_argument("--x-vector-only-mode", action="store_true")

    speak = subparsers.add_parser("speak", help="single text synthesis")
    speak.add_argument("--voice-id", required=True)
    speak.add_argument("--text", required=True)
    speak.add_argument("--out", required=True)
    add_common_synthesis_args(speak)

    batch = subparsers.add_parser("batch", help="batch synthesis from txt file")
    batch.add_argument("--voice-id", required=True)
    batch.add_argument("--input", required=True, help="text file with one sentence per line")
    batch.add_argument("--out-dir", required=True)
    batch.add_argument("--no-resume", action="store_true", help="do not skip existing outputs")
    add_common_synthesis_args(batch)

    voices = subparsers.add_parser("voices", help="voice management")
    voices_sub = voices.add_subparsers(dest="voices_command", required=True)

    voices_sub.add_parser("list", help="list voices")
    voices_delete = voices_sub.add_parser("delete", help="delete voice")
    voices_delete.add_argument("--voice-id", required=True)

    serve = subparsers.add_parser("serve", help="local queue worker")
    serve_sub = serve.add_subparsers(dest="serve_command", required=True)

    serve_start = serve_sub.add_parser("start", help="start queue worker")
    serve_start.add_argument("--queue", default=None, help="queue sqlite path or directory")
    serve_start.add_argument("--poll-interval-sec", type=float, default=1.0)
    add_common_synthesis_args(serve_start)

    job = subparsers.add_parser("job", help="queue job operations")
    job_sub = job.add_subparsers(dest="job_command", required=True)

    job_submit = job_sub.add_parser("submit", help="submit synthesis job")
    job_submit.add_argument("--queue", default=None, help="queue sqlite path or directory")
    job_submit.add_argument("--voice-id", required=True)
    job_submit.add_argument("--text", required=True)
    job_submit.add_argument("--language", default=DEFAULT_LANGUAGE)
    job_submit.add_argument("--seed", type=int, default=None)
    job_submit.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    job_submit.add_argument("--max-attempts", type=int, default=3)

    job_status = job_sub.add_parser("status", help="check job status")
    job_status.add_argument("--queue", default=None, help="queue sqlite path or directory")
    job_status.add_argument("--job-id", required=True)

    job_fetch = job_sub.add_parser("fetch", help="fetch completed job output")
    job_fetch.add_argument("--queue", default=None, help="queue sqlite path or directory")
    job_fetch.add_argument("--job-id", required=True)
    job_fetch.add_argument("--out", required=True)

    return parser


def dispatch(args, runtime_paths: RuntimePaths, logger) -> int:
    if args.command == "doctor":
        return run_doctor(args, runtime_paths, logger)

    if args.command == "clone" and args.clone_command == "create":
        return run_clone_create(args, runtime_paths, logger)

    if args.command == "speak":
        return run_speak(args, runtime_paths, logger)

    if args.command == "batch":
        return run_batch(args, runtime_paths, logger)

    if args.command == "voices" and args.voices_command == "list":
        return run_voices_list(args, runtime_paths)

    if args.command == "voices" and args.voices_command == "delete":
        return run_voices_delete(args, runtime_paths)

    if args.command == "serve" and args.serve_command == "start":
        return run_serve_start(args, runtime_paths, logger)

    if args.command == "job" and args.job_command == "submit":
        return run_job_submit(args, runtime_paths)

    if args.command == "job" and args.job_command == "status":
        return run_job_status(args, runtime_paths)

    if args.command == "job" and args.job_command == "fetch":
        return run_job_fetch(args, runtime_paths)

    raise QTTSError(ErrorCode.INVALID_INPUT, "unsupported command")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    runtime_paths = resolve_runtime_paths(args.runtime_root)
    runtime_paths.ensure()

    logger = setup_logging(runtime_paths.logs_dir / "qtts.log", verbose=args.verbose)

    try:
        return dispatch(args, runtime_paths, logger)
    except QTTSError as err:
        logger.error(
            err.message,
            extra={"event": "error", "error_code": err.code.value},
        )
        print_json(err.to_dict(), stream=sys.stderr)
        return exit_code_for(err)
    except Exception as exc:  # noqa: BLE001
        logger.exception("unhandled error", extra={"event": "error_unhandled"})
        err = QTTSError(ErrorCode.INTERNAL_ERROR, "unhandled error", cause=exc)
        print_json(err.to_dict(), stream=sys.stderr)
        return exit_code_for(err)


if __name__ == "__main__":
    raise SystemExit(main())
