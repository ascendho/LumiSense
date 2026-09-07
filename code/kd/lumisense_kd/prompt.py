from __future__ import annotations

import json
from typing import Any

from .data_io import CLASS_IDS

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

TEACHER_SYSTEM_PROMPT = SYSTEM_PROMPT + """\

Apply this private diagnostic rubric when producing training supervision:
prefer power when minimum voltage is at most 2.2 V, a brownout occurred, or
large voltage variation coincides with repeated load dips. Otherwise prefer
sensor when self-test/configuration failed, conversion errors are repeated, or
no fresh samples were produced. Otherwise prefer communication when outages,
large tail delay, heavy loss with retries, or joint duplicate/gap evidence is
present. Choose process anomaly only when the device layers remain plausible
and peer or reference context supports a physical change. Power-correlated
transport symptoms remain power faults.

For teacher supervision, cite exactly one evidence item: the strongest direct
positive indicator of the selected root cause. Do not cite healthy/normal
status, absence of another fault, or a secondary symptom. Preserve JSON value
types exactly: booleans and numbers must remain unquoted JSON booleans and
numbers, never strings. For unstable supply, cite voltage variation, load dips,
or their cross-layer alignment rather than a normal minimum voltage. A peer
direction ratio by itself is not sufficient process-anomaly evidence; cite the
actual target/peer change or recovery behavior.

Always inspect the power-monitor event even when structured power fields are
missing. A reported window minimum at or below 2.2 V or remaining battery at or
below 15 percent establishes POWER_FAULT and dominates secondary sensing or
transport symptoms. Zero or one conversion error without a failed self-test,
invalid configuration, frozen output, or sample stall is normal. A target or
peer temperature change with magnitude below 2.5 C is also normal and cannot
establish PROCESS_ANOMALY by itself."""


def snapshot_text(record: dict[str, Any]) -> str:
    return json.dumps(
        record["diagnostic_snapshot"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def target_text(target: dict[str, Any]) -> str:
    return json.dumps(target, ensure_ascii=False, separators=(",", ":"))


def inference_messages(record: dict[str, Any], *, teacher: bool = False) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": TEACHER_SYSTEM_PROMPT if teacher else SYSTEM_PROMPT},
        {"role": "user", "content": snapshot_text(record)},
    ]


def training_record(
    record: dict[str, Any],
    label: dict[str, Any],
    *,
    target: dict[str, Any] | None = None,
    teacher_logits: dict[str, float] | None = None,
    teacher_accepted: bool = False,
) -> dict[str, Any]:
    selected_target = target or label["target"]
    if selected_target["class_id"] != CLASS_IDS[label["root_cause"]]:
        raise ValueError(f"Training target and root cause disagree in {record['sample_id']}")
    output: dict[str, Any] = {
        "sample_id": record["sample_id"],
        "label": label["root_cause"],
        "class_id": label["class_id"],
        "messages": [
            *inference_messages(record),
            {"role": "assistant", "content": target_text(selected_target)},
        ],
        "teacher_accepted": bool(teacher_accepted),
    }
    if teacher_logits is not None:
        if set(teacher_logits) != set(CLASS_IDS.values()):
            raise ValueError("Teacher logits must be keyed by A/B/C/D")
        output["teacher_logits"] = {
            class_id: float(teacher_logits[class_id]) for class_id in CLASS_IDS.values()
        }
    return output
