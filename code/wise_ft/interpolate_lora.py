#!/usr/bin/env python3
"""Interpolate SFT and KD LoRA deltas into a deployment adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumisense_wise.adapter import interpolate_lora_adapters


def main() -> None:
    parser = argparse.ArgumentParser(description="Interpolate SFT and KD LoRA adapters with WiSE-FT.")
    parser.add_argument("--sft-adapter", type=Path, required=True)
    parser.add_argument("--kd-adapter", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.5, help="KD endpoint weight in [0, 1].")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = interpolate_lora_adapters(
        sft_adapter=args.sft_adapter.resolve(),
        kd_adapter=args.kd_adapter.resolve(),
        output_dir=args.output.resolve(),
        alpha=args.alpha,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
