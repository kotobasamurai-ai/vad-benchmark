"""Orchestrates multi-point parameter sweeps across (engine, dataset) pairs.

For each sweep-enabled engine we instantiate one engine per parameter value, run it on
each dataset, and collect per-parameter frame-level metrics. A trapezoidal AUC over
the resulting (FPR, TPR) points is then reported per dataset.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .datasets import build_loader
from .engines import SWEEP_PARAM, build_engine
from .runner import run_engine_on_loader
from .sweep import SweepPoint, make_sweep_point, trapezoidal_auc


@dataclass
class SweepDatasetResult:
    engine: str
    dataset: str
    param_name: str
    points: list[SweepPoint] = field(default_factory=list)
    auc_trap: float | None = None


def _collect_probs_labels(results) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.concatenate([r.labels for r in results]) if results else np.zeros(0, dtype=np.uint8)
    y_score = (
        np.concatenate([r.probs for r in results]) if results else np.zeros(0, dtype=np.float32)
    )
    return y_true, y_score


def run_sweep(
    engine_name: str,
    values: list[float],
    datasets: list[str],
    config: dict,
    max_seconds: float | None,
) -> list[SweepDatasetResult]:
    if engine_name not in SWEEP_PARAM:
        raise ValueError(f"engine {engine_name} does not support sweep")
    param_name = SWEEP_PARAM[engine_name]

    # dataset -> list[SweepPoint]
    by_dataset: dict[str, list[SweepPoint]] = {ds: [] for ds in datasets}

    for v in values:
        kw = {param_name: v}
        engine = build_engine(engine_name, **kw)
        try:
            for ds_name in datasets:
                if ds_name not in config:
                    continue
                loader = build_loader(ds_name, config[ds_name])
                results = list(run_engine_on_loader(engine, loader, max_seconds))
                y_true, y_score = _collect_probs_labels(results)
                if len(np.unique(y_true)) < 2:
                    # Skip single-class datasets for sweep AUC; keep the point for FPR only.
                    pt = make_sweep_point(param_name, v, y_true, y_score)
                    by_dataset[ds_name].append(pt)
                    continue
                by_dataset[ds_name].append(make_sweep_point(param_name, v, y_true, y_score))
        finally:
            engine.close()

    out = []
    for ds_name, pts in by_dataset.items():
        dual_class = any(pt.precision is not None for pt in pts)
        auc = trapezoidal_auc(pts) if dual_class else None
        out.append(
            SweepDatasetResult(
                engine=engine_name,
                dataset=ds_name,
                param_name=param_name,
                points=pts,
                auc_trap=auc,
            )
        )
    return out


def dump_sweep(results: list[SweepDatasetResult], out: Path) -> None:
    import json

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([asdict(r) for r in results], indent=2, default=float))
