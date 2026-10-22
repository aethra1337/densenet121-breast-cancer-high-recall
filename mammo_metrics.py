"""Binary classification metrics shared by training and evaluation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

SCALAR_METRICS = (
    "roc_auc",
    "pr_auc",
    "malignant_recall",
    "benign_precision",
    "specificity",
    "malignant_precision",
    "accuracy",
    "balanced_accuracy",
    "f1_malignant",
)


def safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    """Compute class-1 (malignant) metrics at a fixed probability threshold."""
    y_pred = (y_prob >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)

    malignant_recall = safe_ratio(tp, tp + fn)
    benign_precision = safe_ratio(tn, tn + fn)
    specificity = safe_ratio(tn, tn + fp)
    malignant_precision = safe_ratio(tp, tp + fp)
    accuracy = safe_ratio(tp + tn, len(y_true))
    f1_malignant = safe_ratio(2 * tp, 2 * tp + fp + fn)
    balanced_accuracy = (
        (malignant_recall + specificity) / 2.0
        if math.isfinite(malignant_recall) and math.isfinite(specificity)
        else float("nan")
    )

    if len(np.unique(y_true)) == 2:
        roc_auc = float(roc_auc_score(y_true, y_prob))
        pr_auc = float(average_precision_score(y_true, y_prob))
    else:
        roc_auc = float("nan")
        pr_auc = float("nan")

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "malignant_recall": malignant_recall,
        "benign_precision": benign_precision,
        "specificity": specificity,
        "malignant_precision": malignant_precision,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "f1_malignant": f1_malignant,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def format_ratio(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(numeric):
        return "n/a"
    return f"{numeric * 100:.1f}%"


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value
