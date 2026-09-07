from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .data_io import load_jsonl
from .metrics import class_from_response, classification_metrics, macro_f1, minimum_class_recall


def load_cloud_classes(path: Path) -> dict[str, str]:
    cloud_by_id: dict[str, str] = {}
    for item in load_jsonl(path):
        parsed = class_from_response(item["response"])
        if parsed is not None:
            cloud_by_id[item["sample_id"]] = parsed
    return cloud_by_id


def risk_curve(confidence_rows: list[dict[str, Any]], cloud_by_id: dict[str, str]) -> list[dict[str, Any]]:
    thresholds = sorted({round(row["confidence"], 4) for row in confidence_rows if row["confidence"] is not None})
    rows = []
    for tau in thresholds:
        accepted = [
            row for row in confidence_rows
            if row["confidence"] is not None and row["confidence"] >= tau
        ]
        if not accepted:
            continue
        truth = [row["gold_class_id"] for row in accepted]
        edge_pred = [row["class_id"] for row in accepted]
        cascade_truth = []
        cascade_pred = []
        missing_cloud = 0
        for row in confidence_rows:
            if row["confidence"] is not None and row["confidence"] >= tau:
                cascade_pred.append(row["class_id"])
            else:
                cloud = cloud_by_id.get(row["sample_id"])
                if cloud is None:
                    missing_cloud += 1
                    cascade_pred.append(row["class_id"])
                else:
                    cascade_pred.append(cloud)
            cascade_truth.append(row["gold_class_id"])
        rows.append(
            {
                "tau": tau,
                "coverage": len(accepted) / len(confidence_rows),
                "accepted": len(accepted),
                "selective_accuracy": sum(row["correct"] for row in accepted) / len(accepted),
                "selective_macro_f1": macro_f1(truth, edge_pred),
                "cascade_accuracy": sum(t == p for t, p in zip(cascade_truth, cascade_pred, strict=True)) / len(confidence_rows),
                "cascade_macro_f1": macro_f1(cascade_truth, cascade_pred),
                "cascade_min_class_recall": minimum_class_recall(cascade_truth, cascade_pred),
                "cascade_missing_cloud": missing_cloud,
            }
        )
    return rows


def select_threshold(curve: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [row for row in curve if "cascade_macro_f1" in row]
    if not candidates:
        raise ValueError("No cascade candidates available")
    return max(candidates, key=lambda row: (row["cascade_macro_f1"], row["coverage"]))


def write_threshold_report(
    *,
    confidence_path: Path,
    cloud_predictions_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    rows = load_jsonl(confidence_path)
    cloud_by_id = load_cloud_classes(cloud_predictions_path)
    curve = risk_curve(rows, cloud_by_id)
    selected = select_threshold(curve)
    truth = [row["gold_class_id"] for row in rows]
    edge_pred = [row["class_id"] for row in rows]
    report = {
        "samples": len(rows),
        "cloud_samples": len(cloud_by_id),
        "no_class_token": sum(1 for row in rows if row["confidence"] is None),
        "edge_baseline": classification_metrics(truth, edge_pred),
        "curve": curve,
        "selected": selected,
        "selection_rule": "maximize validation cascade_macro_f1; break ties by higher edge coverage",
        "source_files": {
            "edge_confidence": str(confidence_path),
            "cloud_predictions": str(cloud_predictions_path),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def fixed_cascade_report(
    *,
    confidence_path: Path,
    cloud_predictions_path: Path,
    threshold: float,
    output_path: Path,
) -> dict[str, Any]:
    confidence_rows = load_jsonl(confidence_path)
    cloud_rows = load_cloud_classes(cloud_predictions_path)
    truth = []
    edge_pred = []
    cloud_pred = []
    cascade_pred = []
    missing_cloud = []
    accepted = 0
    for row in confidence_rows:
        sample_id = row["sample_id"]
        truth.append(row["gold_class_id"])
        edge_pred.append(row["class_id"])
        cloud = cloud_rows.get(sample_id)
        if cloud is None:
            missing_cloud.append(sample_id)
            cloud = row["class_id"]
        cloud_pred.append(cloud)
        if row["confidence"] is not None and row["confidence"] >= threshold:
            cascade_pred.append(row["class_id"])
            accepted += 1
        else:
            cascade_pred.append(cloud)
    cascade_metrics = classification_metrics(truth, cascade_pred)
    cascade_metrics.update(
        {
            "threshold": threshold,
            "accepted_edge_samples": accepted,
            "cloud_samples": len(confidence_rows) - accepted,
            "edge_coverage": accepted / len(confidence_rows),
            "cloud_fraction": 1.0 - accepted / len(confidence_rows),
        }
    )
    report = {
        "split": "validation",
        "num_samples": len(confidence_rows),
        "selection_rule": "fixed threshold selected on validation",
        "edge_baseline": classification_metrics(truth, edge_pred),
        "cloud_baseline": classification_metrics(truth, cloud_pred),
        "cascade": cascade_metrics,
        "missing_cloud_predictions": missing_cloud,
        "source_files": {
            "edge_confidence": str(confidence_path),
            "cloud_predictions": str(cloud_predictions_path),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
