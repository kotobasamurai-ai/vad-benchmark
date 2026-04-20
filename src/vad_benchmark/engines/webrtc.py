from __future__ import annotations

import numpy as np

from ..labeling import num_frames, resample_probs_to_grid
from .base import VadEngine


class WebRtcEngine(VadEngine):
    """WebRTC VAD. Native hop is 10/20/30ms. We use 30ms and resample to the 31.25ms grid."""

    name = "webrtc"

    def __init__(self, aggressiveness: int = 2, frame_ms: int = 30):
        import webrtcvad

        self.vad = webrtcvad.Vad(aggressiveness)
        self.frame_ms = frame_ms

    def infer(self, wav: np.ndarray, sample_rate: int) -> np.ndarray:
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm = np.clip(wav * 32768.0, -32768, 32767).astype(np.int16).tobytes()
        hop = int(sample_rate * self.frame_ms / 1000) * 2  # bytes per frame (int16)
        probs: list[float] = []
        for i in range(0, len(pcm) - hop + 1, hop):
            frame = pcm[i : i + hop]
            is_speech = self.vad.is_speech(frame, sample_rate)
            probs.append(1.0 if is_speech else 0.0)
        probs_arr = np.asarray(probs, dtype=np.float32)
        duration = len(wav) / sample_rate
        target = num_frames(duration)
        return resample_probs_to_grid(probs_arr, self.frame_ms, target)
