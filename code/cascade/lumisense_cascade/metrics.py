from __future__ import annotations

import json
from collections import Counter
from typing import Any

from .data_io import CLASS_IDS, LABELS, LETTERS


def class_from_response(text: str) -> str | None:
    try:
        value = json.loads(text.strip())
    except (json.JSONDecodeError, AttributeError):
        return None
    if not isinstance(value, dict):
        return None
    class_id = value.get("class_id")
    return class_id if class_id in LETTERS else None


def macro_f1(truth: list[str], predicted: list[str | None]) -> float:
    values = []
    for label in LABELS:
        letter = CLASS_IDS[label]
        tp = sum(t == letter and p == letter for t, p in zip(truth, predicted, strict=True))
        fp = sum(t != letter and p == letter for t, p in zip(truth, predicted, strict=True))
        fn = sum(t == letter and p != letter for t, p in zip(truth, predicted, strict=True))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(values) / len(values)


def minimum_class_recall(truth: list[str], predicted: list[str | None]) -> float:
    recalls = []
    for label in LABELS:
        letter = CLASS_IDS[label]
        tp = sum(t == letter and p == letter for t, p in zip(truth, predicted, strict=True))
        fn = sum(t == letter and p != letter for t, p in zip(truth, predicted, strict=True))
        recalls.append(tp / (tp + fn) if tp + fn else 0.0)
    return min(recalls)


def classification_metrics(truth: list[str], predicted: list[str | None]) -> dict[str, Any]:
    if len(truth) != len(predicted):
        raise ValueError("truth and predicted lengths differ")
    n = len(truth)
    if n == 0:
        raise ValueError("Cannot score an empty set")
    per_class = {}
    for label in LABELS:
        letter = CLASS_IDS[label]
        tp = sum(t == letter and p == letter for t, p in zip(truth, predicted, strict=True))
        fp = sum(t != letter and p == letter for t, p in zip(truth, predicted, strict=True))
        fn = sum(t == letter and p != letter for t, p in zip(truth, predicted, strict=True))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}
    return {
        "accuracy": sum(t == p for t, p in zip(truth, predicted, strict=True)) / n,
        "macro_f1": macro_f1(truth, predicted),
        "minimum_class_recall": minimum_class_recall(truth, predicted),
        "per_class": per_class,
        "prediction_distribution": dict(Counter(predicted)),
    }
