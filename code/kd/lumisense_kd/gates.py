from __future__ import annotations

from typing import Any


def _check(name: str, value: float, minimum: float) -> dict[str, Any]:
    return {
        "name": name,
        "value": value,
        "minimum": minimum,
        "passed": value >= minimum,
    }


def teacher_gate(
    teacher_train: dict[str, Any],
    teacher_validation: dict[str, Any],
) -> dict[str, Any]:
    train_metrics = teacher_train["raw_metrics"]
    validation_metrics = teacher_validation["raw_metrics"]
    checks = [
        _check("validation_macro_f1", validation_metrics["macro_f1"], 0.90),
        _check("train_accepted_rate", teacher_train["accepted_rate"], 0.90),
        _check(
            "minimum_class_accepted_rate",
            teacher_train["minimum_class_accepted_rate"],
            0.85,
        ),
        _check("train_strict_output_rate", train_metrics["strict_output_rate"], 0.98),
        _check(
            "train_grounded_evidence_precision",
            train_metrics["grounded_evidence_precision"],
            0.98,
        ),
        _check(
            "train_relevant_evidence_precision",
            train_metrics["relevant_evidence_precision"],
            0.90,
        ),
    ]
    return {"gate": "teacher", "passed": all(item["passed"] for item in checks), "checks": checks}
