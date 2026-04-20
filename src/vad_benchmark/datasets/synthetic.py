"""Synthetic speech+noise dataset emitted by scripts/build_synthetic.py.

Layout:
    data/synthetic/audio/synth_000.wav
    data/synthetic/labels.json       # [{utt_id, duration, speech_segments:[{start,end},...]}]
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from ..types import AudioItem, SpeechSegment
from .base import DatasetLoader


class SyntheticLoader(DatasetLoader):
    name = "synthetic"

    def __init__(self, root: Path):
        self.root = Path(root)

    def items(self) -> Iterator[AudioItem]:
        labels = json.loads((self.root / "labels.json").read_text())
        for item in labels:
            wav = self.root / "audio" / f"{item['utt_id']}.wav"
            segs = tuple(
                SpeechSegment(float(s["start"]), float(s["end"]))
                for s in item["speech_segments"]
            )
            yield AudioItem(
                audio_path=wav,
                speech_segments=segs,
                duration=float(item["duration"]),
                dataset=self.name,
                utt_id=item["utt_id"],
            )
