from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def load_mono(path: Path, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """Load audio as mono float32 at target_sr."""
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1).astype(np.float32)
    if sr != target_sr:
        import resampy

        wav = resampy.resample(wav, sr, target_sr).astype(np.float32)
        sr = target_sr
    return wav, sr
