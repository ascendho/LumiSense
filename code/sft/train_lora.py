#!/usr/bin/env python3
"""Train the LumiSense supervised LoRA adapter with mlx-lm."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lumisense_sft.data_io import load_jsonl


def ensure_fresh(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def train_lora(
    *,
    model_path: Path,
    data_dir: Path,
    adapter_dir: Path,
    seed: int,
    iterations: int,
    dry_run: bool,
) -> dict[str, Any]:
    ensure_fresh(adapter_dir)
    config = {
        "model": str(model_path),
        "train": True,
        "fine_tune_type": "lora",
        "data": str(data_dir),
        "seed": seed,
        "num_layers": -1,
        "batch_size": 4,
        "iters": iterations,
        "learning_rate": 5e-5,
        "grad_accumulation_steps": 2,
        "adapter_path": str(adapter_dir),
        "max_seq_length": 2048,
        "mask_prompt": True,
        "steps_per_report": 10,
        "steps_per_eval": 35,
        "save_every": 35,
        "lora_parameters": {"rank": 8, "scale": 16.0, "dropout": 0.05},
    }
    config_path = adapter_dir / "training_config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    entrypoint = shutil.which("mlx_lm.lora") or str(Path(sys.executable).with_name("mlx_lm.lora"))
    command = [entrypoint, "--config", str(config_path)]
    metadata = {
        "method": "sft",
        "model": str(model_path),
        "seed": seed,
        "iterations": iterations,
        "train_samples": len(load_jsonl(data_dir / "train.jsonl")),
        "validation_samples": len(load_jsonl(data_dir / "valid.jsonl")),
        "command": command,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "dry_run": dry_run,
    }
    if dry_run:
        print("[dry-run] " + " ".join(command))
        metadata["completed"] = False
    else:
        started = time.perf_counter()
        if shutil.which(command[0]) is None and not Path(command[0]).exists():
            raise FileNotFoundError(command[0])
        subprocess.run(command, check=True)
        metadata["elapsed_s"] = time.perf_counter() - started
        metadata["completed"] = True
    (adapter_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the LumiSense SFT LoRA adapter.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--iterations", type=int, default=350)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    metadata = train_lora(
        model_path=args.model.resolve(),
        data_dir=args.data_dir.resolve(),
        adapter_dir=args.adapter_dir.resolve(),
        seed=args.seed,
        iterations=args.iterations,
        dry_run=args.dry_run,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
