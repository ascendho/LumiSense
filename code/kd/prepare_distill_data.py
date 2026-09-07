#!/usr/bin/env python3
"""Combine SFT targets with accepted teacher logits for KD training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from lumisense_kd.data_io import load_jsonl, load_pairs, select_balanced_pairs, write_jsonl
from lumisense_kd.prompt import training_record

ROOT = Path(__file__).resolve().parents[2]


def prepare_distill_data(
    *,
    data_dir: Path,
    teacher_file: Path,
    output_dir: Path,
    samples_per_class: int | None,
    selection_seed: int,
) -> dict[str, Any]:
    teacher_records = load_jsonl(teacher_file)
    teacher_by_id = {item["sample_id"]: item for item in teacher_records}
    if len(teacher_by_id) != len(teacher_records):
        raise ValueError("Duplicate teacher sample IDs")

    manifest: dict[str, Any] = {
        "method": "distill",
        "teacher_file": str(teacher_file),
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
        converted = []
        for record, label in zip(records, labels, strict=True):
            if split == "train":
                if record["sample_id"] not in teacher_by_id:
                    raise ValueError(f"Missing teacher annotation: {record['sample_id']}")
                converted.append(teacher_by_id[record["sample_id"]])
            else:
                converted.append(training_record(record, label))
        output_path = output_dir / output_name
        write_jsonl(output_path, converted)
        manifest["splits"][split] = {"file": output_name, "samples": len(converted)}

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build KD training data from teacher annotations.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "tests" / "fixtures" / "mini_cares")
    parser.add_argument("--teacher-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-class", type=int)
    parser.add_argument("--selection-seed", type=int, default=17)
    args = parser.parse_args()
    manifest = prepare_distill_data(
        data_dir=args.data_dir.resolve(),
        teacher_file=args.teacher_file.resolve(),
        output_dir=args.output_dir.resolve(),
        samples_per_class=args.samples_per_class,
        selection_seed=args.selection_seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
