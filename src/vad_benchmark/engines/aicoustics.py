"""AI-coustics Quail VAD engine.

API (Python SDK >=2.0):

    import aic_sdk as aic
    model_path = aic.Model.download(model_id, "./models")
    model = aic.Model.from_file(model_path)
    config = aic.ProcessorConfig.optimal(model, num_channels=1)
    processor = aic.Processor(model, license_key, config)
    vad_ctx = processor.get_vad_context()
    vad_ctx.set_parameter(aic.VadParameter.Sensitivity, 6.0)
    # feed float32 (channels, frames) through processor.process(buf)
    # after each process() call, vad_ctx.is_speech_detected() -> bool

The SDK does not expose a continuous probability, only a boolean per processed buffer.
We therefore return 0.0 / 1.0 on the processor's native buffer hop and let
resample_probs_to_grid align to the 31.25ms grid.

The caller must set AICOUSTICS_LICENSE_KEY in the environment (or a .env file at the
project root).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ..labeling import num_frames, resample_probs_to_grid
from .base import VadEngine

DEFAULT_MODEL = "quail-l-16khz"


class AiCousticsEngine(VadEngine):
    name = "aicoustics"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        models_dir: str | Path = "models/aicoustics",
        sensitivity: float = 6.0,
        speech_hold_duration: float = 0.03,
        minimum_speech_duration: float = 0.0,
    ):
        import aic_sdk as aic  # type: ignore

        license_key = os.environ.get("AICOUSTICS_LICENSE_KEY") or os.environ.get(
            "AIC_SDK_LICENSE"
        )
        if not license_key:
            raise RuntimeError(
                "AICOUSTICS_LICENSE_KEY environment variable is required "
                "(set it in .env at the project root)"
            )

        self.aic = aic
        Path(models_dir).mkdir(parents=True, exist_ok=True)
        model_path = aic.Model.download(model_id, str(models_dir))
        self.model = aic.Model.from_file(model_path)
        self.config = aic.ProcessorConfig.optimal(self.model, num_channels=1)
        self.processor = aic.Processor(self.model, license_key, self.config)

        self.vad = self.processor.get_vad_context()
        self.vad.set_parameter(aic.VadParameter.Sensitivity, sensitivity)
        self.vad.set_parameter(aic.VadParameter.SpeechHoldDuration, speech_hold_duration)
        self.vad.set_parameter(aic.VadParameter.MinimumSpeechDuration, minimum_speech_duration)

        self.num_frames_per_buffer = int(self.config.num_frames)
        self.processor_sr = int(self.config.sample_rate)

    def _ensure_sr(self, wav: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
        if sample_rate == self.processor_sr:
            return wav, sample_rate
        import resampy

        resampled = resampy.resample(wav, sample_rate, self.processor_sr).astype(np.float32)
        return resampled, self.processor_sr

    def infer(self, wav: np.ndarray, sample_rate: int) -> np.ndarray:
        wav, sr = self._ensure_sr(wav, sample_rate)
        n = self.num_frames_per_buffer
        hop_ms = n * 1000.0 / sr
        probs: list[float] = []
        for i in range(0, len(wav), n):
            chunk = wav[i : i + n]
            if len(chunk) < n:
                chunk = np.pad(chunk, (0, n - len(chunk)))
            buf = np.ascontiguousarray(chunk[None, :], dtype=np.float32)  # (1, n)
            self.processor.process(buf)
            probs.append(1.0 if self.vad.is_speech_detected() else 0.0)
        probs_arr = np.asarray(probs, dtype=np.float32)
        duration_original = len(wav) / sr
        target = num_frames(duration_original)
        return resample_probs_to_grid(probs_arr, hop_ms, target)
