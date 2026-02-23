from __future__ import annotations

import base64
import pickle
from typing import Any

from .errors import ErrorCode, QTTSError


def encode_prompt(prompt_obj: Any) -> dict[str, Any]:
    payload = base64.b64encode(pickle.dumps(prompt_obj)).decode("ascii")
    return {
        "kind": "pickle-b64",
        "payload": payload,
    }


def decode_prompt(prompt_cache: dict[str, Any] | None) -> Any:
    if not prompt_cache:
        return None

    kind = prompt_cache.get("kind")
    payload = prompt_cache.get("payload")

    if kind != "pickle-b64" or not isinstance(payload, str):
        raise QTTSError(
            ErrorCode.INVALID_INPUT,
            "prompt cache format is invalid",
            retryable=False,
        )

    try:
        raw = base64.b64decode(payload.encode("ascii"))
        return pickle.loads(raw)
    except Exception as exc:  # noqa: BLE001
        raise QTTSError(
            ErrorCode.INVALID_INPUT,
            "prompt cache is corrupted",
            retryable=False,
            cause=exc,
        ) from exc
