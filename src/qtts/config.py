from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .constants import (
    JOBS_DIRNAME,
    LOGS_ROOT,
    OUTPUT_DIRNAME,
    QUEUE_DB_NAME,
    RUNTIME_ROOT,
    VOICES_DIRNAME,
)


@dataclass(frozen=True)
class RuntimePaths:
    runtime_root: Path
    voices_dir: Path
    jobs_dir: Path
    output_dir: Path
    logs_dir: Path
    queue_db: Path

    def ensure(self) -> None:
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)


def resolve_runtime_paths(runtime_root: str | Path | None = None) -> RuntimePaths:
    root = Path(runtime_root) if runtime_root is not None else RUNTIME_ROOT
    root = root.expanduser().resolve()

    jobs_dir = root / JOBS_DIRNAME
    return RuntimePaths(
        runtime_root=root,
        voices_dir=root / VOICES_DIRNAME,
        jobs_dir=jobs_dir,
        output_dir=root / OUTPUT_DIRNAME,
        logs_dir=LOGS_ROOT.resolve(),
        queue_db=jobs_dir / QUEUE_DB_NAME,
    )
