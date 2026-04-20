"""Parameter sweeps for engines that return binary speech/non-speech decisions.

Silero returns continuous probabilities, so roc_auc_score gives a true AUC from one run.
WebRTC and AI-coustics only return 0/1, so a single run gives only one (FPR, TPR) point.
To get an AUC-like number we run the engine at multiple operating points (aggressiveness,
sensitivity) and integrate the resulting (FPR, TPR) curve with the trapezoidal rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import precision_recall_f1


@dataclass
class SweepPoint:
    param_name: str
    param_value: float
    false_positive_rate: float
    true_positive_rate: float  # = recall
    precision: float | None
    f1: float | None
    accuracy: float


def make_sweep_point(
    param_name: str,
    param_value: float,
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> SweepPoint:
    cm = precision_recall_f1(y_true, y_score)
    y_pred = (y_score >= 0.5).astype(np.uint8)
    acc = float(np.mean(y_pred == y_true))
    tpr = cm.recall if cm.recall is not None else 0.0
    return SweepPoint(
        param_name=param_name,
        param_value=float(param_value),
        false_positive_rate=cm.false_positive_rate,
        true_positive_rate=tpr,
        precision=cm.precision,
        f1=cm.f1,
        accuracy=acc,
    )


def trapezoidal_auc(points: list[SweepPoint]) -> float | None:
    """Trapezoidal AUC from a set of (FPR, TPR) points.

    Sort by FPR ascending, anchor the curve at (0,0) and (1,1), integrate with np.trapz.
    Requires at least 2 distinct FPR values.
    """
    if not points:
        return None
    xs = [p.false_positive_rate for p in points]
    ys = [p.true_positive_rate for p in points]
    xs.append(0.0)
    ys.append(0.0)
    xs.append(1.0)
    ys.append(1.0)
    pairs = sorted(zip(xs, ys))
    xs_sorted = np.array([p[0] for p in pairs])
    ys_sorted = np.array([p[1] for p in pairs])
    if len(np.unique(xs_sorted)) < 2:
        return None
    return float(np.trapezoid(ys_sorted, xs_sorted))
