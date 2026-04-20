from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from ..types import AudioItem


class DatasetLoader(ABC):
    """Yields AudioItems with speech_segments in seconds."""

    name: str = "base"

    @abstractmethod
    def items(self) -> Iterator[AudioItem]: ...


def take_up_to(items: Iterator[AudioItem], max_seconds: float | None) -> Iterator[AudioItem]:
    if max_seconds is None:
        yield from items
        return
    total = 0.0
    for it in items:
        dur = it.duration or 0.0
        if dur <= 0:
            import soundfile as sf

            info = sf.info(str(it.audio_path))
            dur = info.frames / info.samplerate
            it = AudioItem(
                audio_path=it.audio_path,
                speech_segments=it.speech_segments,
                duration=dur,
                dataset=it.dataset,
                utt_id=it.utt_id,
            )
        if total + dur > max_seconds and total > 0:
            break
        yield it
        total += dur
        if total >= max_seconds:
            break
