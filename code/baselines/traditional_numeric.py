#!/usr/bin/env python3
"""Train and evaluate a numeric-feature random-forest baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from lumisense_baselines.data_io import load_pairs
from lumisense_baselines.features import extract_numeric_features

ROOT = Path(__file__).resolve().parents[2]


def vectorize(rows: list[dict[str, float]], feature_order: list[str]) -> list[list[float]]:
    return [[row.get(key, 0.0) for key in feature_order] for row in rows]


def metric_block(truth: list[str], pred: list[str], labels: list[str]) -> dict[str, Any]:
    from sklearn.metrics import f1_score, precision_recall_fscore_support

    recalls = dict(
        zip(
            labels,
            precision_recall_fscore_support(
                truth,
                pred,
                labels=labels,
                average=None,
                zero_division=0,
            )[2],
            strict=True,
        )
    )
    return {
        "accuracy": round(sum(t == p for t, p in zip(truth, pred, strict=True)) / len(truth), 4),
        "macro_f1": round(float(f1_score(truth, pred, average="macro")), 4),
        "minimum_class_recall": round(float(min(recalls.values())), 4),
        "per_class_recall": {key: round(float(value), 4) for key, value in recalls.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train numeric-feature traditional baselines on train and evaluate validation."
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "tests" / "fixtures" / "mini_cares")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "baselines" / "traditional_numeric.json")
    args = parser.parse_args()

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    train_records, train_labels = load_pairs(args.data_dir, "train")
    val_records, val_labels = load_pairs(args.data_dir, "validation")
    train_features = [extract_numeric_features(item["diagnostic_snapshot"]) for item in train_records]
    val_features = [extract_numeric_features(item["diagnostic_snapshot"]) for item in val_records]
    feature_order = sorted({key for row in train_features for key in row})
    train_x = vectorize(train_features, feature_order)
    val_x = vectorize(val_features, feature_order)
    train_y = [item["root_cause"] for item in train_labels]
    val_y = [item["root_cause"] for item in val_labels]
    labels = sorted(set(train_y))

    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    val_x = scaler.transform(val_x)
    models = {
        "logistic_regression": LogisticRegression(max_iter=5000),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=17),
    }
    results = {}
    for name, model in models.items():
        model.fit(train_x, train_y)
        results[name] = metric_block(val_y, list(model.predict(val_x)), labels)

    summary = {
        "split": "validation",
        "n_train": len(train_records),
        "n_validation": len(val_records),
        "n_features": len(feature_order),
        "task_note": "Traditional numeric classifiers predict labels only and cannot produce evidence-grounded JSON.",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
