#!/usr/bin/env python3
"""Export fixed few-shot demonstrations for public LLM baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumisense_baselines.prompt import export_fewshot_demos

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export fixed train examples for LLM few-shot baselines.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "tests" / "fixtures" / "mini_cares")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shots-per-class", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    payload = export_fewshot_demos(
        data_dir=args.data_dir,
        output=args.output,
        shots_per_class=args.shots_per_class,
        seed=args.seed,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
