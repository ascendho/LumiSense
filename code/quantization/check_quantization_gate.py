#!/usr/bin/env python3
"""Check whether GGUF quantization preserves enough validation quality."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumisense_quant.gates import quantization_gate


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Check GGUF fidelity against the MLX WiSE-FT baseline.")
    parser.add_argument("--mlx-summary", type=Path, required=True)
    parser.add_argument("--gguf-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = quantization_gate(read_json(args.mlx_summary), read_json(args.gguf_summary))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 2)


if __name__ == "__main__":
    main()
