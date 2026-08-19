from __future__ import annotations

import numpy as np


def binary_metrics(labels: list[int], scores: list[float], threshold: float) -> dict:
    y_true = np.asarray(labels, dtype=np.int32)
    y_score = np.asarray(scores, dtype=np.float32)
    y_pred = (y_score >= threshold).astype(np.int32)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    accuracy = (tp + tn) / max(len(y_true), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    return {
        "count": int(len(y_true)),
        "threshold": round(float(threshold), 6),
        "accuracy": round(float(accuracy), 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "specificity": round(float(specificity), 6),
        "f1": round(float(f1), 6),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def auroc(labels: list[int], scores: list[float]) -> float | None:
    y_true = np.asarray(labels, dtype=np.int32)
    y_score = np.asarray(scores, dtype=np.float32)
    positives = y_score[y_true == 1]
    negatives = y_score[y_true == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return None

    wins = 0.0
    total = float(len(positives) * len(negatives))
    for score in positives:
        wins += float((score > negatives).sum())
        wins += 0.5 * float((score == negatives).sum())
    return round(wins / total, 6)

