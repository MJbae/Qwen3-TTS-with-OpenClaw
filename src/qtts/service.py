from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any

from .audio_utils import concatenate_with_pause, write_wav
from .constants import (
    DEFAULT_LANGUAGE,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT_SEC,
)
from .engine import QwenTTSEngine
from .errors import ErrorCode, QTTSError
from .prompt_codec import decode_prompt, encode_prompt
from .text_utils import split_text
from .voice_store import VoiceStore


class SynthesisService:
    def __init__(self, voice_store: VoiceStore, engine: QwenTTSEngine, logger):
        self.voice_store = voice_store
        self.engine = engine
        self.logger = logger

    def clone_voice(
        self,
        *,
        voice_id: str,
        ref_audio: Path,
        ref_text: str,
        force: bool = False,
        x_vector_only_mode: bool = False,
    ) -> tuple[dict[str, Any], bool]:
        if self.voice_store.exists(voice_id) and not force:
            return self.voice_store.load_voice(voice_id), True

        prompt = self.engine.create_voice_clone_prompt(
            ref_audio_path=ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only_mode,
        )

        prompt_cache = encode_prompt(prompt)
        return self.voice_store.create_voice(
            voice_id,
            ref_audio,
            ref_text,
            prompt_cache=prompt_cache,
            force=force,
            allow_reuse=True,
            x_vector_only_mode=x_vector_only_mode,
        )

    def _resolve_prompt(self, voice_id: str, voice_record: dict[str, Any]) -> Any:
        prompt_cache = voice_record.get("prompt_cache")
        if isinstance(prompt_cache, dict):
            try:
                return decode_prompt(prompt_cache)
            except QTTSError:
                self.logger.warning(
                    "cached prompt is invalid. rebuilding prompt",
                    extra={"event": "prompt_cache_invalid", "voice_id": voice_id},
                )

        prompt = self.engine.create_voice_clone_prompt(
            ref_audio_path=Path(voice_record["ref_audio_path"]),
            ref_text=str(voice_record["ref_text"]),
            x_vector_only_mode=bool(voice_record.get("x_vector_only_mode", False)),
        )
        self.voice_store.update_prompt_cache(voice_id, encode_prompt(prompt))
        return prompt

    def _call_with_timeout(self, fn, timeout_sec: int):
        if timeout_sec <= 0:
            return fn()

        pool = ThreadPoolExecutor(max_workers=1)
        future = pool.submit(fn)
        try:
            return future.result(timeout=timeout_sec)
        except FutureTimeoutError:
            future.cancel()
            raise
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    def _synthesize_chunk_with_retry(
        self,
        *,
        chunk_text: str,
        prompt: Any,
        language: str,
        seed: int | None,
        max_new_tokens: int,
        timeout_sec: int,
        retries: int,
        chunk_index: int,
    ) -> tuple[Any, int]:
        last_error: QTTSError | None = None

        for attempt in range(retries + 1):
            try:
                return self._call_with_timeout(
                    lambda: self.engine.synthesize(
                        text=chunk_text,
                        voice_clone_prompt=prompt,
                        language=language,
                        max_new_tokens=max_new_tokens,
                        seed=seed,
                    ),
                    timeout_sec,
                )
            except FutureTimeoutError as exc:
                last_error = QTTSError(
                    ErrorCode.TIMEOUT,
                    f"synthesis timeout after {timeout_sec}s (chunk={chunk_index + 1})",
                    retryable=True,
                    cause=exc,
                )
            except QTTSError as exc:
                last_error = exc
            except Exception as exc:  # noqa: BLE001
                last_error = QTTSError(
                    ErrorCode.SYNTHESIS_FAIL,
                    f"synthesis failed at chunk={chunk_index + 1}",
                    retryable=False,
                    cause=exc,
                )

            if last_error is None:
                continue

            should_retry = attempt < retries and last_error.retryable
            if should_retry:
                self.logger.warning(
                    "retrying chunk synthesis",
                    extra={
                        "event": "synthesis_retry",
                        "voice_id": None,
                    },
                )
                continue

            raise last_error

        if last_error is None:
            raise QTTSError(ErrorCode.INTERNAL_ERROR, "unexpected synthesis state")
        raise last_error

    def _synthesize_text_to_file(
        self,
        *,
        text: str,
        out_path: Path,
        prompt: Any,
        language: str,
        seed: int | None,
        max_new_tokens: int,
        timeout_sec: int,
        retries: int,
        split_max_chars: int,
    ) -> dict[str, Any]:
        chunks = split_text(text, max_chars=split_max_chars)
        if not chunks:
            raise QTTSError(ErrorCode.INVALID_INPUT, "text cannot be empty")

        chunk_wavs = []
        sample_rate: int | None = None

        for idx, chunk in enumerate(chunks):
            wav, sr = self._synthesize_chunk_with_retry(
                chunk_text=chunk,
                prompt=prompt,
                language=language,
                seed=seed,
                max_new_tokens=max_new_tokens,
                timeout_sec=timeout_sec,
                retries=retries,
                chunk_index=idx,
            )
            if sample_rate is None:
                sample_rate = sr
            elif sample_rate != sr:
                raise QTTSError(
                    ErrorCode.SYNTHESIS_FAIL,
                    "chunk sample rates do not match",
                    retryable=False,
                )
            chunk_wavs.append(wav)

        if sample_rate is None:
            raise QTTSError(ErrorCode.SYNTHESIS_FAIL, "no audio generated")

        full_wav = concatenate_with_pause(chunk_wavs, sample_rate)
        write_wav(out_path, full_wav, sample_rate)

        duration = float(len(full_wav) / sample_rate) if sample_rate > 0 else 0.0
        return {
            "out": str(out_path.resolve()),
            "sample_rate": sample_rate,
            "duration_sec": round(duration, 3),
            "chunks": len(chunks),
        }

    def speak_to_file(
        self,
        *,
        voice_id: str,
        text: str,
        out_path: Path,
        language: str = DEFAULT_LANGUAGE,
        seed: int | None = None,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        timeout_sec: int = DEFAULT_TIMEOUT_SEC,
        retries: int = DEFAULT_MAX_RETRIES,
        split_max_chars: int = 240,
    ) -> dict[str, Any]:
        voice_record = self.voice_store.load_voice(voice_id)
        prompt = self._resolve_prompt(voice_id, voice_record)
        result = self._synthesize_text_to_file(
            text=text,
            out_path=out_path,
            prompt=prompt,
            language=language,
            seed=seed,
            max_new_tokens=max_new_tokens,
            timeout_sec=timeout_sec,
            retries=retries,
            split_max_chars=split_max_chars,
        )
        result["voice_id"] = voice_id
        return result

    def batch_to_dir(
        self,
        *,
        voice_id: str,
        input_path: Path,
        out_dir: Path,
        language: str = DEFAULT_LANGUAGE,
        seed: int | None = None,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        timeout_sec: int = DEFAULT_TIMEOUT_SEC,
        retries: int = DEFAULT_MAX_RETRIES,
        split_max_chars: int = 240,
        resume: bool = True,
    ) -> dict[str, Any]:
        if not input_path.exists() or not input_path.is_file():
            raise QTTSError(ErrorCode.INVALID_INPUT, f"input file not found: {input_path}")

        lines = [line.strip() for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            raise QTTSError(ErrorCode.INVALID_INPUT, "input file has no non-empty lines")

        voice_record = self.voice_store.load_voice(voice_id)
        prompt = self._resolve_prompt(voice_id, voice_record)

        out_dir.mkdir(parents=True, exist_ok=True)

        success = 0
        failures: list[dict[str, Any]] = []
        skipped = 0

        for idx, line in enumerate(lines, start=1):
            out_path = out_dir / f"{idx:04d}.wav"
            if resume and out_path.exists():
                skipped += 1
                continue

            try:
                self._synthesize_text_to_file(
                    text=line,
                    out_path=out_path,
                    prompt=prompt,
                    language=language,
                    seed=seed,
                    max_new_tokens=max_new_tokens,
                    timeout_sec=timeout_sec,
                    retries=retries,
                    split_max_chars=split_max_chars,
                )
                success += 1
            except QTTSError as exc:
                failures.append(
                    {
                        "line": idx,
                        "text": line,
                        "error_code": exc.code.value,
                        "message": exc.message,
                    }
                )

        return {
            "voice_id": voice_id,
            "input": str(input_path.resolve()),
            "out_dir": str(out_dir.resolve()),
            "total": len(lines),
            "success": success,
            "skipped": skipped,
            "failed": len(failures),
            "failures": failures,
        }
