from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .audio import load_mono
from .datasets.base import DatasetLoader, take_up_to
from .engines.base import VadEngine
from .labeling import num_frames, segments_to_labels
from .metrics import precision_recall_f1, roc_auc, single_class_accuracy
from .types import AudioItem

TARGET_SR = 16000


@dataclass
class ItemResult:
    dataset: str
    utt_id: str
    duration: float
    n_frames: int
    n_speech_frames: int
    probs: np.ndarray
    labels: np.ndarray


def run_engine_on_loader(
    engine: VadEngine,
    loader: DatasetLoader,
    max_seconds: float | None,
    progress: bool = True,
) -> Iterator[ItemResult]:
    items = take_up_to(loader.items(), max_seconds)
    if progress:
        items = tqdm(items, desc=f"{engine.name}/{loader.name}", unit="clip")
    for it in items:
        yield _run_one(engine, it)


def _run_one(engine: VadEngine, it: AudioItem) -> ItemResult:
    wav, sr = load_mono(it.audio_path, TARGET_SR)
    duration = len(wav) / sr
    labels = segments_to_labels(it.speech_segments, duration)
    n = num_frames(duration)
    if len(labels) != n:
        labels = labels[:n] if len(labels) > n else np.pad(labels, (0, n - len(labels)))
    probs = engine.infer(wav, sr)
    if len(probs) != n:
        probs = probs[:n] if len(probs) > n else np.pad(probs, (0, n - len(probs)))
    return ItemResult(
        dataset=it.dataset,
        utt_id=it.utt_id,
        duration=duration,
        n_frames=n,
        n_speech_frames=int(labels.sum()),
        probs=probs.astype(np.float32),
        labels=labels.astype(np.uint8),
    )


@dataclass
class DatasetMetrics:
    dataset: str
    engine: str
    n_clips: int
    total_seconds: float
    n_frames: int
    n_speech_frames: int
    roc_auc: float | None
    accuracy: float
    precision: float | None
    recall: float | None
    f1: float | None
    false_positive_rate: float


def aggregate(results: list[ItemResult], engine_name: str) -> DatasetMetrics:
    if not results:
        return DatasetMetrics(
            dataset="", engine=engine_name, n_clips=0, total_seconds=0,
            n_frames=0, n_speech_frames=0, roc_auc=None, accuracy=0.0,
            precision=None, recall=None, f1=None, false_positive_rate=0.0,
        )
    y_true = np.concatenate([r.labels for r in results])
    y_score = np.concatenate([r.probs for r in results])
    is_single_class = len(np.unique(y_true)) < 2
    acc = (
        single_class_accuracy(y_true, y_score)
        if is_single_class
        else float(np.mean((y_score >= 0.5).astype(np.uint8) == y_true))
    )
    cm = precision_recall_f1(y_true, y_score)
    return DatasetMetrics(
        dataset=results[0].dataset,
        engine=engine_name,
        n_clips=len(results),
        total_seconds=sum(r.duration for r in results),
        n_frames=int(y_true.size),
        n_speech_frames=int(y_true.sum()),
        roc_auc=roc_auc(y_true, y_score),
        accuracy=acc,
        precision=cm.precision,
        recall=cm.recall,
        f1=cm.f1,
        false_positive_rate=cm.false_positive_rate,
    )


def dump_json(metrics: list[DatasetMetrics], out: Path) -> None:
    import json

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([asdict(m) for m in metrics], indent=2))
