from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable

LABELS = (
    "PROCESS_ANOMALY",
    "SENSOR_FAULT",
    "COMMUNICATION_FAULT",
    "POWER_FAULT",
)
CLASS_IDS = dict(zip(LABELS, "ABCD", strict=True))
ID_TO_LABEL = {value: key for key, value in CLASS_IDS.items()}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def load_pairs(data_dir: Path, split: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = load_jsonl(data_dir / f"{split}.jsonl")
    labels = load_jsonl(data_dir / "labels" / f"{split}.jsonl")
    labels_by_id = {item["sample_id"]: item for item in labels}
    record_ids = [item["sample_id"] for item in records]
    if len(labels_by_id) != len(labels) or set(labels_by_id) != set(record_ids):
        raise ValueError(f"Mismatched or duplicate record/label IDs in {split}")
    return records, [labels_by_id[sample_id] for sample_id in record_ids]


def select_balanced_pairs(
    records: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    *,
    per_class: int,
    seed: int,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if per_class <= 0:
        raise ValueError("per_class must be positive")
    rng = random.Random(seed)
    by_class: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {
        label: [] for label in LABELS
    }
    for record, label in zip(records, labels, strict=True):
        by_class[label["root_cause"]].append((record, label))
    selected = []
    for root_cause, items in by_class.items():
        if len(items) < per_class:
            raise ValueError(f"Requested {per_class} {root_cause} demos, found {len(items)}")
        rng.shuffle(items)
        selected.extend(items[:per_class])
    selected.sort(key=lambda item: CLASS_IDS[item[1]["root_cause"]])
    return selected
