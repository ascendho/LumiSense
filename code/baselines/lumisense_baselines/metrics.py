from __future__ import annotations

import json
import math
import re
import statistics
from collections import Counter
from typing import Any

from .data_io import ID_TO_LABEL, LABELS


def clean_generation(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def parse_response(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(clean_generation(text))
    except (json.JSONDecodeError, AttributeError):
        return None
    if not isinstance(value, dict) or set(value) != {"class_id", "evidence"}:
        return None
    if value["class_id"] not in ID_TO_LABEL:
        return None
    evidence = value["evidence"]
    if not isinstance(evidence, list) or not 1 <= len(evidence) <= 3:
        return None
    for fact in evidence:
        if (
            not isinstance(fact, dict)
            or set(fact) != {"path", "value"}
            or not isinstance(fact["path"], str)
            or not fact["path"]
        ):
            return None
    return value


def resolve_path(value: Any, path: str) -> Any:
    current = value
    for key, index in re.findall(r"(?:^|\.)([^.\[\]]+)|\[(\d+)\]", path):
        current = current[int(index)] if index else current[key]
    return current


def exact_value(expected: Any, supplied: Any) -> bool:
    if isinstance(expected, bool) or isinstance(supplied, bool):
        return type(expected) is bool and type(supplied) is bool and expected == supplied
    if isinstance(expected, (int, float)) and isinstance(supplied, (int, float)):
        return math.isclose(float(expected), float(supplied), rel_tol=1e-9, abs_tol=0.0)
    return type(expected) is type(supplied) and expected == supplied


def classification_metrics(truth: list[str], predictions: list[str | None]) -> dict[str, Any]:
    per_class = {}
    f1_values = []
    recalls = []
    for label in LABELS:
        tp = sum(t == label and p == label for t, p in zip(truth, predictions, strict=True))
        fp = sum(t != label and p == label for t, p in zip(truth, predictions, strict=True))
        fn = sum(t == label and p != label for t, p in zip(truth, predictions, strict=True))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}
        f1_values.append(f1)
        recalls.append(recall)
    return {
        "accuracy": sum(t == p for t, p in zip(truth, predictions, strict=True)) / len(truth),
        "macro_f1": statistics.fmean(f1_values),
        "minimum_class_recall": min(recalls),
        "per_class": per_class,
        "prediction_distribution": dict(sorted(Counter(p for p in predictions if p).items())),
    }


def score_predictions(
    records: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    label_by_id = {item["sample_id"]: item for item in labels}
    record_by_id = {item["sample_id"]: item for item in records}
    prediction_by_id = {item["sample_id"]: item for item in predictions}
    expected_ids = set(label_by_id)
    if expected_ids != set(record_by_id) or expected_ids != set(prediction_by_id):
        raise ValueError("Record, label, and prediction IDs must match")

    truth: list[str] = []
    predicted: list[str | None] = []
    valid_count = 0
    relevant_cases = 0
    fact_count = 0
    grounded_facts = 0
    relevant_facts = 0
    for sample_id, label in label_by_id.items():
        truth.append(label["root_cause"])
        parsed = parse_response(prediction_by_id[sample_id]["response"])
        if parsed is None:
            predicted.append(None)
            continue
        valid_count += 1
        predicted.append(ID_TO_LABEL[parsed["class_id"]])
        checks = []
        for fact in parsed["evidence"]:
            exists = True
            try:
                actual = resolve_path(record_by_id[sample_id]["diagnostic_snapshot"], fact["path"])
            except (KeyError, IndexError, TypeError):
                exists = False
                actual = None
            grounded = exists and exact_value(actual, fact["value"])
            relevant = fact["path"] in set(label["acceptable_evidence_paths"])
            checks.append((grounded, relevant))
        fact_count += len(checks)
        grounded_facts += sum(grounded for grounded, _ in checks)
        relevant_facts += sum(grounded and relevant for grounded, relevant in checks)
        relevant_cases += any(grounded and relevant for grounded, relevant in checks)

    metrics = classification_metrics(truth, predicted)
    metrics.update(
        strict_output_rate=valid_count / len(truth),
        grounded_evidence_precision=grounded_facts / fact_count if fact_count else 0.0,
        relevant_evidence_precision=relevant_facts / fact_count if fact_count else 0.0,
        relevant_evidence_coverage=relevant_cases / len(truth),
    )
    return metrics
