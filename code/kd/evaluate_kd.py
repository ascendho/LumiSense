#!/usr/bin/env python3
"""Evaluate KD checkpoints or final adapters on CARES splits."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from lumisense_kd.data_io import load_pairs, select_balanced_pairs, write_jsonl
from lumisense_kd.metrics import score_predictions
from lumisense_kd.prompt import inference_messages

ROOT = Path(__file__).resolve().parents[2]


def render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    options = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    try:
        return tokenizer.apply_chat_template(messages, **options)
    except TypeError:
        options.pop("enable_thinking")
        return tokenizer.apply_chat_template(messages, **options)


def evaluate_mlx_adapter(
    *,
    model_path: Path,
    adapter_path: Path,
    data_dir: Path,
    split: str,
    output_dir: Path,
    run_name: str,
    max_samples: int | None,
    samples_per_class: int | None,
    selection_seed: int,
    max_tokens: int,
) -> dict[str, Any]:
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    records, labels = load_pairs(data_dir, split)
    if max_samples is not None and samples_per_class is not None:
        raise ValueError("max_samples and samples_per_class are mutually exclusive")
    records, labels = select_balanced_pairs(
        records,
        labels,
        per_class=samples_per_class,
        seed=selection_seed,
    )
    if max_samples is not None:
        records = records[:max_samples]
        selected_ids = {item["sample_id"] for item in records}
        labels = [item for item in labels if item["sample_id"] in selected_ids]

    model, tokenizer = load(
        str(model_path),
        adapter_path=str(adapter_path),
        tokenizer_config={"trust_remote_code": True},
    )
    sampler = make_sampler(temp=0.0)
    predictions = []
    for index, record in enumerate(records, start=1):
        start = time.perf_counter()
        response = generate(
            model,
            tokenizer,
            prompt=render_prompt(tokenizer, inference_messages(record)),
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False,
        )
        predictions.append(
            {
                "sample_id": record["sample_id"],
                "response": response.strip(),
                "latency_s": time.perf_counter() - start,
            }
        )
        print(f"[{index:04d}/{len(records):04d}] {record['sample_id']}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / f"{run_name}_{split}_predictions.jsonl"
    summary_path = output_dir / f"{run_name}_{split}_summary.json"
    write_jsonl(prediction_path, predictions)
    summary = {
        "run_name": run_name,
        "model": str(model_path),
        "adapter": str(adapter_path),
        "split": split,
        "samples": len(records),
        "metrics": score_predictions(records, labels, predictions),
        "prediction_path": str(prediction_path),
        "test_opened": False,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a KD adapter on train or validation.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "tests" / "fixtures" / "mini_cares")
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--samples-per-class", type=int)
    parser.add_argument("--selection-seed", type=int, default=17)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()
    summary = evaluate_mlx_adapter(
        model_path=args.model.resolve(),
        adapter_path=args.adapter.resolve(),
        data_dir=args.data_dir.resolve(),
        split=args.split,
        output_dir=args.output_dir.resolve(),
        run_name=args.run_name,
        max_samples=args.max_samples,
        samples_per_class=args.samples_per_class,
        selection_seed=args.selection_seed,
        max_tokens=args.max_tokens,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
