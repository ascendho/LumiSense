#!/usr/bin/env python3
"""Evaluate a fixed-threshold edge-cloud cascade from saved predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumisense_cascade.cascade import fixed_cascade_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate cascade with a fixed validation threshold.")
    parser.add_argument("--confidence", type=Path, required=True)
    parser.add_argument("--cloud-predictions", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = fixed_cascade_report(
        confidence_path=args.confidence.resolve(),
        cloud_predictions_path=args.cloud_predictions.resolve(),
        threshold=args.threshold,
        output_path=args.output.resolve(),
    )
    print(
        json.dumps(
            {
                "threshold": report["cascade"]["threshold"],
                "edge_coverage": report["cascade"]["edge_coverage"],
                "cloud_fraction": report["cascade"]["cloud_fraction"],
                "macro_f1": report["cascade"]["macro_f1"],
                "minimum_class_recall": report["cascade"]["minimum_class_recall"],
                "accuracy": report["cascade"]["accuracy"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
