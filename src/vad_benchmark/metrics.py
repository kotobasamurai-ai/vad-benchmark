from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def accuracy_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> float:
    y_pred = (y_score >= threshold).astype(np.uint8)
    return float(np.mean(y_pred == y_true))


def single_class_accuracy(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> float:
    """Silero wiki uses 'accuracy on the entire audio' for single-class datasets.
    For speech-only files: fraction of frames predicted speech. For noise-only: fraction
    predicted non-speech. We detect the single class from y_true.
    """
    label = int(y_true[0])
    y_pred = (y_score >= threshold).astype(np.uint8)
    if label == 1:
        return float(np.mean(y_pred == 1))
    return float(np.mean(y_pred == 0))


@dataclass
class ClassMetrics:
    precision: float | None
    recall: float | None
    f1: float | None
    false_positive_rate: float  # noise frames incorrectly flagged as speech
    true_negative_rate: float  # specificity
    n_positive: int
    n_negative: int


def precision_recall_f1(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5
) -> ClassMetrics:
    """Frame-level precision/recall/F1 for the speech class.

    - precision = TP / (TP + FP). Undefined if the model predicts no positives.
    - recall    = TP / (TP + FN). Undefined if the ground truth has no positives.
    - FPR       = FP / (FP + TN). Degenerate if the ground truth has no negatives.
    """
    y_true = y_true.astype(np.uint8)
    y_pred = (y_score >= threshold).astype(np.uint8)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    n_pos = tp + fn
    n_neg = fp + tn

    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / n_pos if n_pos > 0 else None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = None
    fpr = fp / n_neg if n_neg > 0 else 0.0
    tnr = tn / n_neg if n_neg > 0 else 0.0
    return ClassMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        false_positive_rate=fpr,
        true_negative_rate=tnr,
        n_positive=n_pos,
        n_negative=n_neg,
    )
