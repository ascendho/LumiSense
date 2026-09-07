#!/usr/bin/env python3
"""Benchmark GGUF inference under CPU and memory-constrained proxy profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumisense_quant.benchmark import PROFILES, docker_benchmark_command, run_constrained_benchmark

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a constrained CPU-only gateway proxy benchmark.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--profile", choices=tuple(PROFILES), required=True)
    parser.add_argument("--image", default="ghcr.io/ggml-org/llama.cpp:full")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "quantization" / "resources" / "benchmark.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        command = docker_benchmark_command(
            model_path=args.model,
            profile=args.profile,
            image=args.image,
            container_name="lumisense-benchmark-dry-run",
        )
        print(json.dumps(command, ensure_ascii=False, indent=2))
        return
    summary = run_constrained_benchmark(
        model_path=args.model.resolve(),
        profile=args.profile,
        output_path=args.output.resolve(),
        image=args.image,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
