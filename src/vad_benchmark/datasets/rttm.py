"""RTTM parser that flattens all speaker turns into merged speech segments."""

from __future__ import annotations

from pathlib import Path

from ..types import SpeechSegment


def parse_rttm_as_speech_segments(path: Path) -> list[SpeechSegment]:
    raw: list[tuple[float, float]] = []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) < 5 or parts[0] != "SPEAKER":
            continue
        start = float(parts[3])
        dur = float(parts[4])
        raw.append((start, start + dur))
    raw.sort()
    merged: list[tuple[float, float]] = []
    for s, e in raw:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return [SpeechSegment(s, e) for s, e in merged]
