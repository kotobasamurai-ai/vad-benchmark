"""Convert speech segments -> per-frame binary labels on the 31.25ms grid.

The Silero wiki evaluates on 31.25ms audio segments. At 16kHz that is exactly 500 samples
per frame. We compute the *fraction* of each frame covered by ground-truth speech, then
threshold at 0.5 to get a binary label.

Each VAD engine is also responsible for returning probabilities aligned to this same grid,
so labels and predictions compare 1:1.
"""

from __future__ import annotations

import numpy as np

from . import FRAME_MS
from .types import SpeechSegment


def num_frames(duration_s: float) -> int:
    return int(np.floor(duration_s * 1000.0 / FRAME_MS))


def segments_to_labels(
    segments: tuple[SpeechSegment, ...] | list[SpeechSegment],
    duration_s: float,
) -> np.ndarray:
    n = num_frames(duration_s)
    coverage = np.zeros(n, dtype=np.float32)
    frame_s = FRAME_MS / 1000.0
    for seg in segments:
        if seg.end <= seg.start:
            continue
        start_f = seg.start / frame_s
        end_f = seg.end / frame_s
        i0 = int(np.floor(start_f))
        i1 = int(np.ceil(end_f))
        i0 = max(0, i0)
        i1 = min(n, i1)
        for i in range(i0, i1):
            f_lo = i
            f_hi = i + 1
            overlap = max(0.0, min(end_f, f_hi) - max(start_f, f_lo))
            coverage[i] = min(1.0, coverage[i] + overlap)
    return (coverage >= 0.5).astype(np.uint8)


def resample_probs_to_grid(
    probs: np.ndarray,
    src_hop_ms: float,
    target_frames: int,
) -> np.ndarray:
    """Resample a per-chunk probability array to the 31.25ms grid by nearest-neighbor
    on frame midpoints. Engines with a native hop that matches FRAME_MS are a no-op.
    """
    if len(probs) == 0:
        return np.zeros(target_frames, dtype=np.float32)
    src_centers = (np.arange(len(probs), dtype=np.float64) + 0.5) * src_hop_ms
    tgt_centers = (np.arange(target_frames, dtype=np.float64) + 0.5) * FRAME_MS
    idx = np.clip(np.round(tgt_centers / src_hop_ms - 0.5).astype(int), 0, len(probs) - 1)
    return probs[idx].astype(np.float32)
