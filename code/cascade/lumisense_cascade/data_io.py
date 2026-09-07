from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

LABELS = (
    "PROCESS_ANOMALY",
    "SENSOR_FAULT",
    "COMMUNICATION_FAULT",
    "POWER_FAULT",
)
CLASS_IDS = dict(zip(LABELS, "ABCD", strict=True))
LETTERS = tuple(CLASS_IDS.values())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def load_pairs(data_dir: Path, split: str = "validation") -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = load_jsonl(data_dir / f"{split}.jsonl")
    labels = load_jsonl(data_dir / "labels" / f"{split}.jsonl")
    labels_by_id = {item["sample_id"]: item for item in labels}
    record_ids = [item["sample_id"] for item in records]
    if len(labels_by_id) != len(labels) or set(labels_by_id) != set(record_ids):
        raise ValueError(f"Mismatched or duplicate record/label IDs in {split}")
    return records, [labels_by_id[sample_id] for sample_id in record_ids]
