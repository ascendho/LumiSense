#!/usr/bin/env python3
"""Generate teacher responses and class-token logits for KD supervision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumisense_kd.teacher import annotate_dataset

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate train/validation data with a teacher model.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "tests" / "fixtures" / "mini_cares")
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--minimum-probability", type=float, default=0.45)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--samples-per-class", type=int)
    parser.add_argument("--selection-seed", type=int, default=17)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_samples is not None and args.samples_per_class is not None:
        parser.error("--max-samples and --samples-per-class are mutually exclusive")

    output = args.output or ROOT / "results" / "kd" / "teacher" / f"{args.split}.jsonl"
    summary_path = args.summary or ROOT / "results" / "kd" / "teacher" / f"{args.split}_summary.json"
    summary = annotate_dataset(
        model_path=args.model.resolve(),
        records_path=args.data_dir.resolve() / f"{args.split}.jsonl",
        labels_path=args.data_dir.resolve() / "labels" / f"{args.split}.jsonl",
        output_path=output.resolve(),
        summary_path=summary_path.resolve(),
        min_probability=args.minimum_probability,
        max_samples=args.max_samples,
        samples_per_class=args.samples_per_class,
        selection_seed=args.selection_seed,
        split=args.split,
        resume=args.resume,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
