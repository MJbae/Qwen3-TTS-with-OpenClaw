import sys
import types

import pytest

from qtts.engine import QwenTTSEngine
from qtts.errors import ErrorCode, QTTSError


def test_engine_rejects_non_cpu_device_before_model_load(monkeypatch):
    calls = {"from_pretrained": 0}

    fake_torch = types.ModuleType("torch")
    fake_torch.float32 = object()
    fake_torch.bfloat16 = object()

    class FakeQwen3TTSModel:
        @classmethod
        def from_pretrained(cls, _model_id, **_kwargs):
            calls["from_pretrained"] += 1
            return object()

    fake_qwen_tts = types.ModuleType("qwen_tts")
    fake_qwen_tts.Qwen3TTSModel = FakeQwen3TTSModel

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "qwen_tts", fake_qwen_tts)

    engine = QwenTTSEngine(model_id="Qwen/Qwen3-TTS-12Hz-1.7B-Base", device="cuda:0")

    with pytest.raises(QTTSError) as exc_info:
        engine._load_model()

    assert exc_info.value.code == ErrorCode.INVALID_INPUT
    assert "cpu" in exc_info.value.message.lower()
    assert calls["from_pretrained"] == 0
