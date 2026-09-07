from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .data_io import load_pairs, select_balanced_pairs

SYSTEM_PROMPT = """\
An upstream detector has already marked one temperature-sensor node as
anomalous. Diagnose the dominant root cause from the incomplete cross-layer
snapshot and event logs.

Classes:
- A = PROCESS_ANOMALY: the physical process changed while sensing, power, and
  communication remain plausible.
- B = SENSOR_FAULT: acquisition, self-test, configuration, freezing, or
  calibration evidence identifies the sensor as the source.
- C = COMMUNICATION_FAULT: transport loss, signal quality, delay, gaps,
  duplicates, retries, or outages dominate.
- D = POWER_FAULT: voltage, supply variation, brownouts, resets, load dips, or
  short uptime explain the anomaly and any secondary symptoms.

Return only compact JSON with exactly two keys. Cite one to three paths whose
values are copied exactly from the supplied snapshot:
{"class_id":"A|B|C|D","evidence":[{"path":"exact.path","value":"exact value"}]}
Do not propose an action and do not output chain-of-thought."""


def snapshot_text(record: dict[str, Any]) -> str:
    return json.dumps(
        record["diagnostic_snapshot"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def target_text(target: dict[str, Any]) -> str:
    return json.dumps(target, ensure_ascii=False, separators=(",", ":"))


def inference_messages(
    record: dict[str, Any],
    *,
    demos: list[tuple[dict[str, Any], dict[str, Any]]] | None = None,
) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if demos:
        for demo_record, demo_label in demos:
            messages.append({"role": "user", "content": snapshot_text(demo_record)})
            messages.append({"role": "assistant", "content": target_text(demo_label["target"])})
    messages.append({"role": "user", "content": snapshot_text(record)})
    return messages


def export_fewshot_demos(
    *,
    data_dir: Path,
    output: Path,
    shots_per_class: int,
    seed: int,
) -> dict[str, Any]:
    records, labels = load_pairs(data_dir, "train")
    demos = select_balanced_pairs(records, labels, per_class=shots_per_class, seed=seed)
    payload = {
        "seed": seed,
        "shots_per_class": shots_per_class,
        "source_split": "train",
        "sample_ids": [record["sample_id"] for record, _ in demos],
        "demos": [
            {"sample_id": record["sample_id"], "record": record, "label": label}
            for record, label in demos
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def load_fewshot_demos(path: Path) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [(item["record"], item["label"]) for item in payload["demos"]]
