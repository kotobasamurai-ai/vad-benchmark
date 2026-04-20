from __future__ import annotations

import numpy as np

from ..labeling import num_frames, resample_probs_to_grid
from .base import VadEngine


class SileroEngine(VadEngine):
    """Silero VAD. Native chunk is 512 samples at 16kHz (32ms) or 256 at 8kHz (32ms).
    Produces one probability per chunk. We resample to the 31.25ms grid.
    """

    name = "silero"

    def __init__(self, onnx: bool = False):
        import torch
        from silero_vad import load_silero_vad

        self.torch = torch
        self.model = load_silero_vad(onnx=onnx)

    def infer(self, wav: np.ndarray, sample_rate: int) -> np.ndarray:
        assert sample_rate in (8000, 16000)
        window = 512 if sample_rate == 16000 else 256
        hop_ms = window * 1000.0 / sample_rate  # 32ms at 16kHz
        t = self.torch.from_numpy(np.ascontiguousarray(wav))
        self.model.reset_states()
        probs: list[float] = []
        for i in range(0, len(t), window):
            chunk = t[i : i + window]
            if len(chunk) < window:
                chunk = self.torch.nn.functional.pad(chunk, (0, window - len(chunk)))
            p = self.model(chunk, sample_rate).item()
            probs.append(float(p))
        probs_arr = np.asarray(probs, dtype=np.float32)
        duration = len(wav) / sample_rate
        target = num_frames(duration)
        return resample_probs_to_grid(probs_arr, hop_ms, target)
