from __future__ import annotations

import json
from typing import Any

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


def inference_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": snapshot_text(record)},
    ]
