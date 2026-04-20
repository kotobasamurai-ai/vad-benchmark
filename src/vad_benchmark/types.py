from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SpeechSegment:
    start: float
    end: float


@dataclass(frozen=True)
class AudioItem:
    audio_path: Path
    speech_segments: tuple[SpeechSegment, ...]
    duration: float | None = None
    dataset: str = ""
    utt_id: str = ""
