from __future__ import annotations

from .base import VadEngine


def build_engine(name: str, **kwargs) -> VadEngine:
    if name == "webrtc":
        from .webrtc import WebRtcEngine

        return WebRtcEngine(**kwargs)
    if name == "silero":
        from .silero import SileroEngine

        return SileroEngine(**kwargs)
    if name == "aicoustics":
        from .aicoustics import AiCousticsEngine

        return AiCousticsEngine(**kwargs)
    raise ValueError(f"unknown engine: {name}")


# engine -> sweep parameter name
SWEEP_PARAM = {
    "webrtc": "aggressiveness",
    "aicoustics": "sensitivity",
}

# suggested sweep points for engines that only emit binary output
DEFAULT_SWEEP = {
    "webrtc": [0, 1, 2, 3],
    "aicoustics": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
}


KNOWN_ENGINES = ("webrtc", "silero", "aicoustics")
