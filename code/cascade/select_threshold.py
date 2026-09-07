#!/usr/bin/env python3
"""Select a validation-only confidence threshold for edge-cloud cascade."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumisense_cascade.cascade import write_threshold_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Select the cascade threshold on validation.")
    parser.add_argument("--confidence", type=Path, required=True)
    parser.add_argument("--cloud-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = write_threshold_report(
        confidence_path=args.confidence.resolve(),
        cloud_predictions_path=args.cloud_predictions.resolve(),
        output_path=args.output.resolve(),
    )
    print(json.dumps(report["selected"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
