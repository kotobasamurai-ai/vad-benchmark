from __future__ import annotations

from pathlib import Path

from .base import DatasetLoader


def build_loader(name: str, cfg: dict) -> DatasetLoader:
    if name == "esc50":
        from .esc50 import Esc50Loader

        return Esc50Loader(root=Path(cfg["root"]))
    if name == "voxconverse":
        from .voxconverse import VoxConverseLoader

        return VoxConverseLoader(
            audio_root=Path(cfg["audio_root"]),
            rttm_root=Path(cfg["rttm_root"]),
        )
    if name == "synthetic":
        from .synthetic import SyntheticLoader

        return SyntheticLoader(root=Path(cfg["root"]))
    raise ValueError(f"unknown dataset: {name}")


KNOWN_DATASETS = ("esc50", "voxconverse", "synthetic")
