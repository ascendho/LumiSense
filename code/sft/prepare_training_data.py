#!/usr/bin/env python3
"""Build MLX chat-format training JSONL files for the SFT stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from lumisense_sft.data_io import load_pairs, select_balanced_pairs, write_jsonl
from lumisense_sft.prompt import training_record

ROOT = Path(__file__).resolve().parents[2]


def prepare_training_data(
    data_dir: Path,
    output_dir: Path,
    *,
    samples_per_class: int | None,
    selection_seed: int,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "method": "sft",
        "samples_per_class": samples_per_class,
        "selection_seed": selection_seed,
        "splits": {},
        "test_opened": False,
    }
    for split, output_name in (("train", "train.jsonl"), ("validation", "valid.jsonl")):
        records, labels = load_pairs(data_dir, split)
        records, labels = select_balanced_pairs(
            records,
            labels,
            per_class=samples_per_class,
            seed=selection_seed,
        )
        converted = [
            training_record(record, label)
            for record, label in zip(records, labels, strict=True)
        ]
        write_jsonl(output_dir / output_name, converted)
        manifest["splits"][split] = {"file": output_name, "samples": len(converted)}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LumiSense SFT chat data.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "tests" / "fixtures" / "mini_cares")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-class", type=int)
    parser.add_argument("--selection-seed", type=int, default=17)
    args = parser.parse_args()
    manifest = prepare_training_data(
        args.data_dir.resolve(),
        args.output_dir.resolve(),
        samples_per_class=args.samples_per_class,
        selection_seed=args.selection_seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
