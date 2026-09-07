#!/usr/bin/env python3
"""Train the KD LoRA adapter from an SFT warm start."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumisense_kd.distill import train_distilled


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a KD LoRA adapter from an SFT adapter.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--init-adapter", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--iterations", type=int, default=250)
    parser.add_argument("--teacher-weight", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    args = parser.parse_args()
    metadata = train_distilled(
        model_path=args.model.resolve(),
        data_dir=args.data_dir.resolve(),
        adapter_dir=args.adapter_dir.resolve(),
        init_adapter=args.init_adapter.resolve(),
        seed=args.seed,
        iterations=args.iterations,
        teacher_weight=args.teacher_weight,
        temperature=args.temperature,
        learning_rate=args.learning_rate,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
