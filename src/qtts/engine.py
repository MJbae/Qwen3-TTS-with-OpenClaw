from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from .constants import (
    DEFAULT_CPU_INTEROP_THREADS,
    DEFAULT_CPU_THREADS,
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    DEFAULT_LANGUAGE,
    DEFAULT_LOW_CPU_MEM_USAGE,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL_ID,
)
from .errors import ErrorCode, QTTSError


class QwenTTSEngine:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = DEFAULT_DEVICE,
        attn_implementation: str | None = None,
        dtype: str | Any = DEFAULT_DTYPE,
        low_cpu_mem_usage: bool = DEFAULT_LOW_CPU_MEM_USAGE,
        cpu_threads: int | None = DEFAULT_CPU_THREADS,
        cpu_interop_threads: int | None = DEFAULT_CPU_INTEROP_THREADS,
    ):
        self.model_id = model_id
        self.device = device
        self.attn_implementation = attn_implementation
        self.dtype = dtype
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.cpu_threads = cpu_threads
        self.cpu_interop_threads = cpu_interop_threads
        self._model: Any = None
        self._torch: Any = None

    @staticmethod
    def _is_cpu_like_device(device: Any) -> bool:
        if isinstance(device, str):
            normalized = device.strip().lower()
            return normalized == "cpu" or normalized.startswith("cpu:")
        if isinstance(device, dict):
            return all(QwenTTSEngine._is_cpu_like_device(value) for value in device.values())
        if isinstance(device, (list, tuple, set)):
            return all(QwenTTSEngine._is_cpu_like_device(value) for value in device)
        return False

    def _resolve_torch_dtype(self) -> Any:
        if self._torch is None:
            return None

        if self.dtype is None:
            return None

        if not isinstance(self.dtype, str):
            return self.dtype

        dtype_name = self.dtype.strip().lower()
        aliases = {
            "float32": "float32",
            "fp32": "float32",
            "float16": "float16",
            "fp16": "float16",
            "half": "float16",
            "bfloat16": "bfloat16",
            "bf16": "bfloat16",
        }
        torch_attr = aliases.get(dtype_name)
        if not torch_attr or not hasattr(self._torch, torch_attr):
            raise QTTSError(
                ErrorCode.INVALID_INPUT,
                f"unsupported dtype: {self.dtype!r}. Expected one of: float32, bfloat16, float16",
            )
        return getattr(self._torch, torch_attr)

    @staticmethod
    def _validate_thread_value(name: str, value: int | None) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise QTTSError(ErrorCode.INVALID_INPUT, f"{name} must be a positive integer or unset")
        return value

    def _configure_torch_cpu_threads(self) -> None:
        if self._torch is None:
            return

        cpu_threads = self._validate_thread_value("cpu_threads", self.cpu_threads)
        cpu_interop_threads = self._validate_thread_value("cpu_interop_threads", self.cpu_interop_threads)

        try:
            if cpu_threads is not None:
                self._torch.set_num_threads(cpu_threads)
            if cpu_interop_threads is not None and hasattr(self._torch, "set_num_interop_threads"):
                self._torch.set_num_interop_threads(cpu_interop_threads)
        except Exception as exc:  # noqa: BLE001
            raise QTTSError(
                ErrorCode.MODEL_LOAD_FAIL,
                "failed to configure torch CPU thread settings",
                cause=exc,
            ) from exc

    def _build_from_pretrained_attempts(self, torch_dtype: Any) -> list[dict[str, Any]]:
        base: dict[str, Any] = {"device_map": self.device}
        if self.low_cpu_mem_usage is not None:
            base["low_cpu_mem_usage"] = self.low_cpu_mem_usage
        if self.attn_implementation:
            base["attn_implementation"] = self.attn_implementation

        attempts: list[dict[str, Any]] = []
        for dtype_key in ("torch_dtype", "dtype", None):
            kwargs = dict(base)
            if dtype_key is not None and torch_dtype is not None:
                kwargs[dtype_key] = torch_dtype
            attempts.append(kwargs)

            for key in ("attn_implementation", "low_cpu_mem_usage", "device_map"):
                fallback = dict(kwargs)
                fallback.pop(key, None)
                attempts.append(fallback)

            minimal = dict(kwargs)
            minimal.pop("attn_implementation", None)
            minimal.pop("low_cpu_mem_usage", None)
            minimal.pop("device_map", None)
            attempts.append(minimal)

        unique_attempts: list[dict[str, Any]] = []
        seen: set[tuple[tuple[str, str], ...]] = set()
        for kwargs in attempts:
            key = tuple(sorted((k, repr(v)) for k, v in kwargs.items()))
            if key in seen:
                continue
            seen.add(key)
            unique_attempts.append(kwargs)
        return unique_attempts

    @staticmethod
    def _looks_like_memory_error(exc: Exception) -> bool:
        message = str(exc).lower()
        markers = (
            "out of memory",
            "cannot allocate memory",
            "insufficient memory",
            "std::bad_alloc",
            "defaultcpuallocator",
            "bad alloc",
            "failed to allocate memory",
            "memoryerror",
            "malloc",
        )
        return any(marker in message for marker in markers)

    def _load_model(self) -> None:
        if self._model is not None:
            return

        if not self._is_cpu_like_device(self.device):
            raise QTTSError(
                ErrorCode.INVALID_INPUT,
                (
                    f"unsupported device {self.device!r}. "
                    "This project is CPU-only; use --device cpu."
                ),
            )

        try:
            import torch  # type: ignore

            self._torch = torch
        except Exception as exc:  # noqa: BLE001
            raise QTTSError(
                ErrorCode.IMPORT_FAIL,
                "failed to import torch",
                cause=exc,
            ) from exc

        try:
            from qwen_tts import Qwen3TTSModel  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise QTTSError(
                ErrorCode.IMPORT_FAIL,
                "failed to import qwen_tts.Qwen3TTSModel",
                cause=exc,
            ) from exc

        self._configure_torch_cpu_threads()
        torch_dtype = self._resolve_torch_dtype()
        attempts = self._build_from_pretrained_attempts(torch_dtype)

        last_type_error: Exception | None = None
        for kwargs in attempts:
            try:
                self._model = Qwen3TTSModel.from_pretrained(self.model_id, **kwargs)
                return
            except TypeError as exc:
                last_type_error = exc
                continue
            except Exception as exc:  # noqa: BLE001
                if self._looks_like_memory_error(exc):
                    raise QTTSError(
                        ErrorCode.MODEL_LOAD_FAIL,
                        (
                            "insufficient RAM to load the 1.7B model on CPU. "
                            "Try --low-cpu-mem-usage, lower --cpu-threads, close other apps, "
                            "or use --model-id Qwen/Qwen3-TTS-12Hz-0.6B-Base."
                        ),
                        cause=exc,
                    ) from exc
                raise QTTSError(
                    ErrorCode.MODEL_LOAD_FAIL,
                    f"failed to load model: {self.model_id}",
                    cause=exc,
                ) from exc

        raise QTTSError(
            ErrorCode.MODEL_LOAD_FAIL,
            f"model init arguments are incompatible for model: {self.model_id}",
            cause=last_type_error,
        )

    def create_voice_clone_prompt(
        self,
        *,
        ref_audio_path: Path,
        ref_text: str,
        x_vector_only_mode: bool = False,
    ) -> Any:
        self._load_model()
        attempts = [
            {
                "ref_audio": str(ref_audio_path),
                "ref_text": ref_text,
                "x_vector_only_mode": x_vector_only_mode,
            },
            {
                "ref_audio": str(ref_audio_path),
                "ref_text": ref_text,
            },
            {
                "ref_audio_path": str(ref_audio_path),
                "ref_text": ref_text,
                "x_vector_only_mode": x_vector_only_mode,
            },
            {
                "ref_audio_path": str(ref_audio_path),
                "ref_text": ref_text,
            },
        ]

        last_error: Exception | None = None
        for kwargs in attempts:
            try:
                return self._model.create_voice_clone_prompt(**kwargs)
            except TypeError as exc:
                last_error = exc
                continue
            except Exception as exc:  # noqa: BLE001
                raise QTTSError(
                    ErrorCode.SYNTHESIS_FAIL,
                    "failed to build voice clone prompt",
                    cause=exc,
                ) from exc

        raise QTTSError(
            ErrorCode.SYNTHESIS_FAIL,
            "failed to build voice clone prompt",
            cause=last_error,
        )

    def _seed(self, seed: int | None) -> None:
        import numpy as np  # type: ignore

        if seed is None:
            return

        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        if self._torch is not None:
            self._torch.manual_seed(seed)

    def synthesize(
        self,
        *,
        text: str,
        voice_clone_prompt: Any,
        language: str = DEFAULT_LANGUAGE,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        seed: int | None = None,
    ) -> tuple[Any, int]:
        import numpy as np  # type: ignore

        self._load_model()
        self._seed(seed)

        kwargs = {
            "text": text,
            "language": language,
            "voice_clone_prompt": voice_clone_prompt,
            "max_new_tokens": max_new_tokens,
        }

        try:
            wavs, sr = self._model.generate_voice_clone(**kwargs)
        except TypeError:
            kwargs.pop("max_new_tokens", None)
            try:
                wavs, sr = self._model.generate_voice_clone(**kwargs)
            except Exception as exc:  # noqa: BLE001
                raise QTTSError(
                    ErrorCode.SYNTHESIS_FAIL,
                    "voice clone synthesis failed",
                    cause=exc,
                    retryable=True,
                ) from exc
        except Exception as exc:  # noqa: BLE001
            raise QTTSError(
                ErrorCode.SYNTHESIS_FAIL,
                "voice clone synthesis failed",
                cause=exc,
                retryable=True,
            ) from exc

        if wavs is None:
            raise QTTSError(ErrorCode.SYNTHESIS_FAIL, "model returned no audio", retryable=True)

        first = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
        if hasattr(first, "detach"):
            first = first.detach().cpu().numpy()

        audio = np.asarray(first, dtype=np.float32).flatten()
        if audio.size == 0:
            raise QTTSError(ErrorCode.SYNTHESIS_FAIL, "model returned empty audio", retryable=True)

        return audio, int(sr)
