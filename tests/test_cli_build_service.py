from types import SimpleNamespace

import qtts.cli as cli


MODEL_17B = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


def test_build_service_passes_cpu_runtime_options(monkeypatch):
    captured = {}

    class FakeEngine:
        def __init__(self, **kwargs):
            captured["engine_kwargs"] = kwargs

    class FakeService:
        def __init__(self, voice_store, engine, logger):
            self.voice_store = voice_store
            self.engine = engine
            self.logger = logger

    monkeypatch.setattr(cli, "QwenTTSEngine", FakeEngine)
    monkeypatch.setattr(cli, "SynthesisService", FakeService)

    args = SimpleNamespace(
        model_id=MODEL_17B,
        device="cpu",
        attn_implementation="eager",
        dtype="float32",
        cpu_threads=8,
        cpu_interop_threads=2,
        low_cpu_mem_usage=True,
    )
    voice_store = object()
    logger = object()

    service = cli.build_service(args, voice_store, logger)

    assert captured["engine_kwargs"] == {
        "model_id": MODEL_17B,
        "device": "cpu",
        "attn_implementation": "eager",
        "dtype": "float32",
        "cpu_threads": 8,
        "cpu_interop_threads": 2,
        "low_cpu_mem_usage": True,
    }
    assert service.voice_store is voice_store
    assert service.logger is logger
