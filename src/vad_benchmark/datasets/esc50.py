"""ESC-50: 50 environmental sound classes, 2000 clips, 5s each, all NON-speech.

Download:
  git clone https://github.com/karolpiczak/ESC-50.git data/esc50

Layout expected:
  data/esc50/audio/*.wav
  data/esc50/meta/esc50.csv  (optional, not needed for labels)
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from ..types import AudioItem
from .base import DatasetLoader


class Esc50Loader(DatasetLoader):
    name = "esc50"

    def __init__(self, root: Path):
        self.root = Path(root)

    def items(self) -> Iterator[AudioItem]:
        audio_dir = self.root / "audio"
        for wav in sorted(audio_dir.glob("*.wav")):
            yield AudioItem(
                audio_path=wav,
                speech_segments=(),  # no speech
                duration=5.0,
                dataset=self.name,
                utt_id=wav.stem,
            )
