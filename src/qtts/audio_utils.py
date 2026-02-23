from __future__ import annotations

from pathlib import Path

def audio_duration_seconds(audio_path: Path) -> float:
    import soundfile as sf  # type: ignore

    info = sf.info(str(audio_path))
    return float(info.duration)


def write_wav(audio_path: Path, waveform, sample_rate: int) -> None:
    import soundfile as sf  # type: ignore

    audio_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(audio_path), waveform, sample_rate)


def concatenate_with_pause(
    chunks: list, sample_rate: int, pause_seconds: float = 0.12
):
    import numpy as np  # type: ignore

    if not chunks:
        return np.array([], dtype=np.float32)

    pause_len = max(0, int(sample_rate * pause_seconds))
    pause = np.zeros(pause_len, dtype=np.float32)

    normalized = [np.asarray(chunk, dtype=np.float32).flatten() for chunk in chunks]

    if len(normalized) == 1:
        return normalized[0]

    pieces: list[np.ndarray] = []
    for index, chunk in enumerate(normalized):
        pieces.append(chunk)
        if index != len(normalized) - 1 and pause_len > 0:
            pieces.append(pause)

    return np.concatenate(pieces)
