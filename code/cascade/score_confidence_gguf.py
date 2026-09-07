#!/usr/bin/env python3
"""Compute class-token routing confidence for the GGUF edge model."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path

from lumisense_cascade.data_io import load_pairs, write_jsonl
from lumisense_cascade.gguf import (
    next_token_probs,
    render_prompt,
    restricted_confidence,
    server_command,
    wait_for_health,
)
from lumisense_cascade.prompt import CLASS_LOGIT_PREFILL, inference_messages

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Score routing confidence for a GGUF edge model.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "tests" / "fixtures" / "mini_cares")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18086)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--ctx-size", type=int, default=2048)
    parser.add_argument("--n-probs", type=int, default=50)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--extra-server-arg", action="append", default=[])
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_dir))
    records, labels = load_pairs(args.data_dir.resolve(), "validation")
    if args.max_samples is not None:
        records = records[: args.max_samples]
        label_by_id = {item["sample_id"]: item for item in labels}
        labels = [label_by_id[item["sample_id"]] for item in records]
    label_by_id = {item["sample_id"]: item for item in labels}

    command = server_command(
        server=args.server.resolve(),
        model=args.model.resolve(),
        port=args.port,
        threads=args.threads,
        context_size=args.ctx_size,
        extra_args=args.extra_server_arg,
    )
    print(f"[server] {' '.join(command)}", flush=True)
    server = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
    )
    rows = []
    try:
        wait_for_health(args.port, server)
        for index, record in enumerate(records, start=1):
            # Prefix the assistant JSON response up to the class value and read
            # only the next-token probabilities for A/B/C/D routing.
            prompt = render_prompt(tokenizer, inference_messages(record)) + CLASS_LOGIT_PREFILL
            raw = next_token_probs(args.port, prompt, args.n_probs)
            predicted, confidence = restricted_confidence(raw)
            gold = label_by_id[record["sample_id"]]["class_id"]
            row = {
                "sample_id": record["sample_id"],
                "gold_class_id": gold,
                "class_id": predicted,
                "confidence": confidence,
                "correct": predicted == gold,
            }
            rows.append(row)
            if index % 20 == 0 or index == len(records):
                print(
                    f"[{index:04d}/{len(records):04d}] {record['sample_id']} "
                    f"pred={predicted} conf={confidence}",
                    flush=True,
                )
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()

    write_jsonl(args.output, rows)
    found = [row for row in rows if row["confidence"] is not None]
    correct = [row for row in rows if row["correct"]]
    wrong = [row for row in rows if not row["correct"]]
    summary = {
        "samples": len(rows),
        "class_token_found": len(found),
        "accuracy": len(correct) / len(rows) if rows else 0.0,
        "mean_confidence": statistics.fmean(row["confidence"] for row in found) if found else None,
        "mean_confidence_correct": statistics.fmean(
            row["confidence"] for row in correct if row["confidence"] is not None
        ) if correct else None,
        "mean_confidence_wrong": statistics.fmean(
            row["confidence"] for row in wrong if row["confidence"] is not None
        ) if wrong else None,
        "model": str(args.model),
        "split": "validation",
    }
    summary_path = args.output.with_name(args.output.stem + "_summary" + args.output.suffix)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
