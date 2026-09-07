#!/usr/bin/env python3
"""Evaluate a GGUF model through llama-server with the public CARES prompt."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from lumisense_quant.data_io import load_pairs, select_balanced_pairs, write_jsonl
from lumisense_quant.export import llama_server_command
from lumisense_quant.metrics import score_predictions
from lumisense_quant.prompt import inference_messages

ROOT = Path(__file__).resolve().parents[2]


def clean_generation(text: str) -> str:
    import re

    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def request_json(url: str, payload: dict | None = None, timeout: float = 30) -> dict:
    body = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def wait_for_server(port: int, process: subprocess.Popen, timeout: float = 180) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("llama-server exited during startup")
        try:
            request_json(f"http://127.0.0.1:{port}/health", timeout=2)
            return
        except (OSError, urllib.error.HTTPError, json.JSONDecodeError):
            time.sleep(1)
    raise TimeoutError("llama-server did not become healthy")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a quantized GGUF model on validation.")
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "tests" / "fixtures" / "mini_cares")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-class", type=int)
    parser.add_argument("--selection-seed", type=int, default=17)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--n-gpu-layers", type=int, default=0)
    parser.add_argument("--ctx-size", type=int, default=2048)
    parser.add_argument("--port", type=int, default=18080)
    args = parser.parse_args()

    split = "validation"
    records, labels = load_pairs(args.data_dir.resolve(), split)
    records, labels = select_balanced_pairs(
        records,
        labels,
        per_class=args.samples_per_class,
        seed=args.selection_seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / f"{args.run_name}_{split}_predictions.jsonl"
    summary_path = args.output_dir / f"{args.run_name}_{split}_summary.json"
    if prediction_path.exists():
        raise FileExistsError(prediction_path)

    command = llama_server_command(
        server=args.server.resolve(),
        model=args.model.resolve(),
        port=args.port,
        threads=args.threads,
        context_size=args.ctx_size,
        n_gpu_layers=args.n_gpu_layers,
    )
    server_log_path = args.output_dir / f"{args.run_name}_server.log"
    predictions = []
    process = None
    server_log = None
    try:
        server_log = server_log_path.open("w", encoding="utf-8", newline="\n")
        process = subprocess.Popen(command, stdout=server_log, stderr=subprocess.STDOUT, text=True)
        wait_for_server(args.port, process)
        for index, record in enumerate(records, start=1):
            payload = {
                "model": args.model.name,
                "messages": inference_messages(record),
                "temperature": 0,
                "max_tokens": args.max_tokens,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            started = time.perf_counter()
            response = request_json(
                f"http://127.0.0.1:{args.port}/v1/chat/completions",
                payload,
                timeout=300,
            )
            content = clean_generation(response["choices"][0]["message"]["content"])
            predictions.append(
                {
                    "sample_id": record["sample_id"],
                    "response": content,
                    "latency_s": time.perf_counter() - started,
                }
            )
            print(f"[{index:04d}/{len(records):04d}] {record['sample_id']}", flush=True)
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        if server_log is not None:
            server_log.close()

    write_jsonl(prediction_path, predictions)
    summary = {
        "run_name": args.run_name,
        "runtime": "llama.cpp",
        "model": str(args.model.resolve()),
        "split": split,
        "samples": len(records),
        "ctx_size": args.ctx_size,
        "metrics": score_predictions(records, labels, predictions),
        "server_command": command,
        "prediction_path": str(prediction_path),
        "test_opened": False,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
