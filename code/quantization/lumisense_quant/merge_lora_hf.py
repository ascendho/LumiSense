from __future__ import annotations

import argparse
import json
from pathlib import Path

from .export import merge_mlx_lora_into_hf


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge an MLX LoRA adapter into an official HF base.")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = merge_mlx_lora_into_hf(
        base_dir=args.base.resolve(),
        adapter_dir=args.adapter.resolve(),
        output_dir=args.output.resolve(),
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
