from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ErrorCode(str, Enum):
    INVALID_INPUT = "INVALID_INPUT"
    AUDIO_VALIDATION_FAIL = "AUDIO_VALIDATION_FAIL"
    VOICE_NOT_FOUND = "VOICE_NOT_FOUND"
    VOICE_ALREADY_EXISTS = "VOICE_ALREADY_EXISTS"
    MODEL_LOAD_FAIL = "MODEL_LOAD_FAIL"
    SYNTHESIS_FAIL = "SYNTHESIS_FAIL"
    TIMEOUT = "TIMEOUT"
    IMPORT_FAIL = "IMPORT_FAIL"
    QUEUE_LOCKED = "QUEUE_LOCKED"
    JOB_NOT_FOUND = "JOB_NOT_FOUND"
    JOB_NOT_READY = "JOB_NOT_READY"
    INTERNAL_ERROR = "INTERNAL_ERROR"


EXIT_CODE_BY_ERROR = {
    ErrorCode.INVALID_INPUT: 2,
    ErrorCode.AUDIO_VALIDATION_FAIL: 3,
    ErrorCode.VOICE_NOT_FOUND: 4,
    ErrorCode.VOICE_ALREADY_EXISTS: 5,
    ErrorCode.MODEL_LOAD_FAIL: 6,
    ErrorCode.SYNTHESIS_FAIL: 7,
    ErrorCode.TIMEOUT: 8,
    ErrorCode.IMPORT_FAIL: 9,
    ErrorCode.QUEUE_LOCKED: 10,
    ErrorCode.JOB_NOT_FOUND: 11,
    ErrorCode.JOB_NOT_READY: 12,
    ErrorCode.INTERNAL_ERROR: 99,
}


@dataclass
class QTTSError(Exception):
    code: ErrorCode
    message: str
    retryable: bool = False
    cause: Exception | None = None

    def __str__(self) -> str:
        return f"{self.code.value}: {self.message}"

    def to_dict(self) -> dict[str, str]:
        data = {
            "error_code": self.code.value,
            "message": self.message,
        }
        if self.cause is not None:
            data["cause"] = str(self.cause)
        return data


def exit_code_for(err: QTTSError) -> int:
    return EXIT_CODE_BY_ERROR.get(err.code, EXIT_CODE_BY_ERROR[ErrorCode.INTERNAL_ERROR])
