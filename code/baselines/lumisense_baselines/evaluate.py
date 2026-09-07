from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .data_io import load_pairs, select_balanced_pairs
from .metrics import score_predictions
from .prompt import inference_messages, load_fewshot_demos


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


def evaluate_mlx_baseline(
    *,
    model_path: Path,
    data_dir: Path,
    output_dir: Path,
    run_name: str,
    demos_path: Path | None,
    max_samples: int | None,
    samples_per_class: int | None,
    selection_seed: int,
    max_tokens: int,
) -> dict[str, Any]:
    if max_samples is not None and samples_per_class is not None:
        raise ValueError("--max-samples and --samples-per-class are mutually exclusive")

    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    records, labels = load_pairs(data_dir, "validation")
    if samples_per_class is not None:
        pairs = select_balanced_pairs(records, labels, per_class=samples_per_class, seed=selection_seed)
        records = [record for record, _ in pairs]
        labels = [label for _, label in pairs]
    elif max_samples is not None:
        records = records[:max_samples]
        keep = {record["sample_id"] for record in records}
        labels = [label for label in labels if label["sample_id"] in keep]

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / f"{run_name}_validation_predictions.jsonl"
    summary_path = output_dir / f"{run_name}_validation_summary.json"
    if predictions_path.exists():
        raise FileExistsError(f"Refusing to overwrite {predictions_path}")

    demos = load_fewshot_demos(demos_path) if demos_path else None
    model, tokenizer = load(str(model_path))
    sampler = make_sampler(temp=0.0)

    predictions = []
    with predictions_path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, record in enumerate(records, start=1):
            prompt = render_prompt(tokenizer, inference_messages(record, demos=demos))
            started = time.perf_counter()
            response = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                sampler=sampler,
                verbose=False,
            ).strip()
            prediction = {
                "sample_id": record["sample_id"],
                "response": response,
                "latency_s": time.perf_counter() - started,
            }
            predictions.append(prediction)
            handle.write(json.dumps(prediction, ensure_ascii=False, sort_keys=True) + "\n")
            print(f"[{index:04d}/{len(records):04d}] {record['sample_id']}", flush=True)

    summary = {
        "run_name": run_name,
        "model": str(model_path),
        "split": "validation",
        "samples": len(records),
        "demos": str(demos_path) if demos_path else None,
        "samples_per_class": samples_per_class,
        "selection_seed": selection_seed,
        "metrics": score_predictions(records, labels, predictions),
        "test_opened": False,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
