from qtts.cli import build_parser


MODEL_17B = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


def _parse(argv: list[str]):
    parser = build_parser()
    return parser.parse_args(argv)


def test_parser_defaults_for_17b_cpu_speak():
    args = _parse(["speak", "--voice-id", "v1", "--text", "hello", "--out", "out.wav"])

    assert args.model_id == MODEL_17B
    assert args.device == "cpu"
    assert args.dtype == "float32"
    assert args.cpu_threads is None
    assert args.cpu_interop_threads == 1
    assert args.low_cpu_mem_usage is True
    assert args.split_max_chars == 180


def test_parser_low_cpu_mem_usage_boolean_toggle():
    disabled = _parse(
        [
            "--no-low-cpu-mem-usage",
            "speak",
            "--voice-id",
            "v1",
            "--text",
            "hello",
            "--out",
            "out.wav",
        ]
    )
    enabled = _parse(
        [
            "--low-cpu-mem-usage",
            "speak",
            "--voice-id",
            "v1",
            "--text",
            "hello",
            "--out",
            "out.wav",
        ]
    )

    assert disabled.low_cpu_mem_usage is False
    assert enabled.low_cpu_mem_usage is True
