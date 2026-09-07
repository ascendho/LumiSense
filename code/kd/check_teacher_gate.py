#!/usr/bin/env python3
"""Check whether teacher annotations are strong enough to enter KD training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumisense_kd.gates import teacher_gate


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether teacher annotations are safe for KD.")
    parser.add_argument("--teacher-train", type=Path, required=True)
    parser.add_argument("--teacher-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = teacher_gate(read_json(args.teacher_train), read_json(args.teacher_validation))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 2)


if __name__ == "__main__":
    main()
