#!/usr/bin/env python3
"""Evaluate prompted LLM baselines with the same CARES scoring contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumisense_baselines.evaluate import evaluate_mlx_baseline

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a base MLX language model baseline on validation.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "tests" / "fixtures" / "mini_cares")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--demos", type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--samples-per-class", type=int)
    parser.add_argument("--selection-seed", type=int, default=17)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()
    summary = evaluate_mlx_baseline(
        model_path=args.model.resolve(),
        data_dir=args.data_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        run_name=args.run_name,
        demos_path=args.demos.resolve() if args.demos else None,
        max_samples=args.max_samples,
        samples_per_class=args.samples_per_class,
        selection_seed=args.selection_seed,
        max_tokens=args.max_tokens,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
