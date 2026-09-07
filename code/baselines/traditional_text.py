#!/usr/bin/env python3
"""Train and evaluate a TF-IDF plus SVM event-text baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from lumisense_baselines.data_io import load_pairs
from lumisense_baselines.features import event_text

ROOT = Path(__file__).resolve().parents[2]


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
        description="Train a TF-IDF + linear SVM text baseline on event-log messages."
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "tests" / "fixtures" / "mini_cares")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "baselines" / "traditional_text.json")
    args = parser.parse_args()

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import LinearSVC

    train_records, train_labels = load_pairs(args.data_dir, "train")
    val_records, val_labels = load_pairs(args.data_dir, "validation")
    train_x = [event_text(record["diagnostic_snapshot"]) for record in train_records]
    val_x = [event_text(record["diagnostic_snapshot"]) for record in val_records]
    train_y = [item["root_cause"] for item in train_labels]
    val_y = [item["root_cause"] for item in val_labels]
    labels = sorted(set(train_y))

    model = make_pipeline(
        TfidfVectorizer(lowercase=True, ngram_range=(1, 2), max_features=20000),
        LinearSVC(C=1.0, class_weight="balanced", max_iter=5000),
    )
    model.fit(train_x, train_y)
    pred = list(model.predict(val_x))
    summary = {
        "split": "validation",
        "n_train": len(train_records),
        "n_validation": len(val_records),
        "task_note": "TF-IDF + SVM reads event text but still predicts labels only.",
        "results": {
            "tfidf_svm": metric_block(val_y, pred, labels),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
