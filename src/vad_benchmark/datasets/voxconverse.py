"""VoxConverse: multi-speaker conversations with RTTM speaker-diarization labels.

Download:
  # Audio (test set)
  #   https://www.robots.ox.ac.uk/~vgg/data/voxconverse/
  # RTTM labels
  git clone https://github.com/joonson/voxconverse data/voxconverse-labels

Expected layout:
  data/voxconverse/test/*.wav
  data/voxconverse-labels/test/*.rttm
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from ..types import AudioItem, SpeechSegment
from .base import DatasetLoader
from .rttm import parse_rttm_as_speech_segments


class VoxConverseLoader(DatasetLoader):
    name = "voxconverse"

    def __init__(self, audio_root: Path, rttm_root: Path):
        self.audio_root = Path(audio_root)
        self.rttm_root = Path(rttm_root)

    def items(self) -> Iterator[AudioItem]:
        for wav in sorted(self.audio_root.glob("*.wav")):
            rttm = self.rttm_root / f"{wav.stem}.rttm"
            if not rttm.exists():
                continue
            segs: tuple[SpeechSegment, ...] = tuple(parse_rttm_as_speech_segments(rttm))
            yield AudioItem(
                audio_path=wav,
                speech_segments=segs,
                dataset=self.name,
                utt_id=wav.stem,
            )
