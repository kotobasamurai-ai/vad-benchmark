from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class VadEngine(ABC):
    """Return per-frame speech probabilities on the 31.25ms grid.

    Implementations receive mono float32 audio at `sample_rate` and must return a 1-D
    float32 array of length labeling.num_frames(duration). Each value in [0, 1].
    """

    name: str = "base"

    @abstractmethod
    def infer(self, wav: np.ndarray, sample_rate: int) -> np.ndarray: ...

    def close(self) -> None:
        pass
