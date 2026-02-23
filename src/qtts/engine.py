from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from .constants import DEFAULT_LANGUAGE, DEFAULT_MAX_NEW_TOKENS, DEFAULT_MODEL_ID
from .errors import ErrorCode, QTTSError


class QwenTTSEngine:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cpu",
        attn_implementation: str | None = None,
    ):
        self.model_id = model_id
        self.device = device
        self.attn_implementation = attn_implementation
        self._model: Any = None
        self._torch: Any = None

    def _load_model(self) -> None:
        if self._model is not None:
            return

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

        dtype = self._torch.float32 if str(self.device).startswith("cpu") else self._torch.bfloat16

        attempts: list[dict[str, Any]] = []
        if self.attn_implementation:
            attempts.append(
                {
                    "device_map": self.device,
                    "dtype": dtype,
                    "attn_implementation": self.attn_implementation,
                }
            )
        attempts.extend(
            [
                {"device_map": self.device, "dtype": dtype},
                {"device_map": self.device},
                {},
            ]
        )

        last_type_error: Exception | None = None
        for kwargs in attempts:
            try:
                self._model = Qwen3TTSModel.from_pretrained(self.model_id, **kwargs)
                return
            except TypeError as exc:
                last_type_error = exc
                continue
            except Exception as exc:  # noqa: BLE001
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

        try:
            return self._model.create_voice_clone_prompt(
                ref_audio_path=str(ref_audio_path),
                ref_text=ref_text,
                x_vector_only_mode=x_vector_only_mode,
            )
        except TypeError:
            try:
                return self._model.create_voice_clone_prompt(
                    ref_audio_path=str(ref_audio_path),
                    ref_text=ref_text,
                )
            except Exception as exc:  # noqa: BLE001
                raise QTTSError(
                    ErrorCode.SYNTHESIS_FAIL,
                    "failed to build voice clone prompt",
                    cause=exc,
                ) from exc
        except Exception as exc:  # noqa: BLE001
            raise QTTSError(
                ErrorCode.SYNTHESIS_FAIL,
                "failed to build voice clone prompt",
                cause=exc,
            ) from exc

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
