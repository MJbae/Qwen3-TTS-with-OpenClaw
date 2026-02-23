from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .audio_utils import audio_duration_seconds
from .constants import MIN_REF_AUDIO_SEC
from .errors import ErrorCode, QTTSError

VOICE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{1,63}$")
PROMPT_FILE = "prompt.json"


class VoiceStore:
    def __init__(self, voices_dir: Path, min_ref_audio_sec: float = MIN_REF_AUDIO_SEC):
        self.voices_dir = voices_dir
        self.min_ref_audio_sec = min_ref_audio_sec
        self.voices_dir.mkdir(parents=True, exist_ok=True)

    def validate_voice_id(self, voice_id: str) -> None:
        if not VOICE_ID_RE.match(voice_id):
            raise QTTSError(
                ErrorCode.INVALID_INPUT,
                "voice-id must match ^[A-Za-z0-9][A-Za-z0-9_-]{1,63}$",
            )

    def voice_dir(self, voice_id: str) -> Path:
        return self.voices_dir / voice_id

    def prompt_path(self, voice_id: str) -> Path:
        return self.voice_dir(voice_id) / PROMPT_FILE

    def exists(self, voice_id: str) -> bool:
        return self.prompt_path(voice_id).exists()

    def create_voice(
        self,
        voice_id: str,
        ref_audio: Path,
        ref_text: str,
        *,
        prompt_cache: dict[str, Any] | None,
        force: bool = False,
        allow_reuse: bool = True,
        x_vector_only_mode: bool = False,
    ) -> tuple[dict[str, Any], bool]:
        self.validate_voice_id(voice_id)

        ref_text = ref_text.strip()
        if not ref_text:
            raise QTTSError(ErrorCode.INVALID_INPUT, "ref-text cannot be empty")

        if not ref_audio.exists() or not ref_audio.is_file():
            raise QTTSError(ErrorCode.INVALID_INPUT, f"ref-audio does not exist: {ref_audio}")

        prompt_path = self.prompt_path(voice_id)
        if prompt_path.exists() and not force:
            if allow_reuse:
                return self.load_voice(voice_id), True
            raise QTTSError(
                ErrorCode.VOICE_ALREADY_EXISTS,
                f"voice-id already exists: {voice_id}",
            )

        try:
            duration = audio_duration_seconds(ref_audio)
        except ModuleNotFoundError as exc:
            raise QTTSError(
                ErrorCode.IMPORT_FAIL,
                "soundfile is required for audio validation",
                cause=exc,
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise QTTSError(
                ErrorCode.AUDIO_VALIDATION_FAIL,
                f"failed to read reference audio: {ref_audio}",
                cause=exc,
            ) from exc
        if duration < self.min_ref_audio_sec:
            raise QTTSError(
                ErrorCode.AUDIO_VALIDATION_FAIL,
                f"reference audio is too short ({duration:.2f}s). minimum is {self.min_ref_audio_sec:.2f}s",
            )

        voice_dir = self.voice_dir(voice_id)
        voice_dir.mkdir(parents=True, exist_ok=True)

        ext = ref_audio.suffix.lower() or ".wav"
        ref_copy_name = f"reference{ext}"
        ref_copy_path = voice_dir / ref_copy_name
        shutil.copy2(ref_audio, ref_copy_path)

        record: dict[str, Any] = {
            "voice_id": voice_id,
            "ref_text": ref_text,
            "ref_audio_file": ref_copy_name,
            "duration_sec": round(duration, 3),
            "x_vector_only_mode": bool(x_vector_only_mode),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if prompt_cache:
            record["prompt_cache"] = prompt_cache

        with prompt_path.open("w", encoding="utf-8") as fp:
            json.dump(record, fp, ensure_ascii=True, indent=2)

        return self.load_voice(voice_id), False

    def update_prompt_cache(self, voice_id: str, prompt_cache: dict[str, Any]) -> dict[str, Any]:
        record = self.load_voice(voice_id)
        record["prompt_cache"] = prompt_cache
        record["updated_at"] = datetime.now(timezone.utc).isoformat()

        payload = {k: v for k, v in record.items() if not k.startswith("_") and k != "ref_audio_path"}
        with self.prompt_path(voice_id).open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=True, indent=2)

        return self.load_voice(voice_id)

    def load_voice(self, voice_id: str) -> dict[str, Any]:
        self.validate_voice_id(voice_id)

        prompt_path = self.prompt_path(voice_id)
        if not prompt_path.exists():
            raise QTTSError(ErrorCode.VOICE_NOT_FOUND, f"voice-id not found: {voice_id}")

        with prompt_path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)

        voice_dir = self.voice_dir(voice_id)
        ref_audio_file = data.get("ref_audio_file")
        if not isinstance(ref_audio_file, str):
            raise QTTSError(
                ErrorCode.INVALID_INPUT,
                f"voice metadata is malformed: {prompt_path}",
            )

        data["_voice_dir"] = str(voice_dir)
        data["_prompt_path"] = str(prompt_path)
        data["ref_audio_path"] = str((voice_dir / ref_audio_file).resolve())
        return data

    def list_voices(self) -> list[dict[str, Any]]:
        voices: list[dict[str, Any]] = []
        for path in sorted(self.voices_dir.iterdir() if self.voices_dir.exists() else []):
            if not path.is_dir():
                continue
            prompt_path = path / PROMPT_FILE
            if not prompt_path.exists():
                continue
            try:
                with prompt_path.open("r", encoding="utf-8") as fp:
                    data = json.load(fp)
                voices.append(
                    {
                        "voice_id": data.get("voice_id", path.name),
                        "created_at": data.get("created_at"),
                        "updated_at": data.get("updated_at"),
                        "duration_sec": data.get("duration_sec"),
                    }
                )
            except Exception:  # noqa: BLE001
                voices.append(
                    {
                        "voice_id": path.name,
                        "created_at": None,
                        "updated_at": None,
                        "duration_sec": None,
                    }
                )
        return voices

    def delete_voice(self, voice_id: str) -> None:
        self.validate_voice_id(voice_id)
        target = self.voice_dir(voice_id)
        if not target.exists():
            raise QTTSError(ErrorCode.VOICE_NOT_FOUND, f"voice-id not found: {voice_id}")
        shutil.rmtree(target)
