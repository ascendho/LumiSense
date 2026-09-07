#!/usr/bin/env python3
"""Export a merged LumiSense adapter to BF16 and quantized GGUF files."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from lumisense_quant.export import gguf_export_commands


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a WiSE-FT adapter as a quantized GGUF model.")
    parser.add_argument("--model", type=Path, required=True, help="Official HF BF16 base model directory.")
    parser.add_argument("--adapter", type=Path, required=True, help="WiSE-FT LoRA adapter directory.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--llama-cpp", type=Path, required=True)
    parser.add_argument("--quantization", choices=("Q4_K_M", "Q5_K_M"), default="Q4_K_M")
    parser.add_argument("--convert-python", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    paths = gguf_export_commands(
        model_path=args.model.resolve(),
        adapter_path=args.adapter.resolve(),
        output_dir=args.output_dir.resolve(),
        llama_cpp=args.llama_cpp.resolve(),
        quantization=args.quantization,
        convert_python=args.convert_python.resolve() if args.convert_python else None,
    )
    serializable = {
        key: [[str(part) for part in command] for command in value]
        if key == "commands"
        else str(value)
        for key, value in paths.items()
    }
    if args.dry_run:
        print(json.dumps(serializable, ensure_ascii=False, indent=2))
        return
    if not args.model.is_dir() or not args.adapter.is_dir():
        raise FileNotFoundError("Model and adapter directories must exist")
    converter = paths["commands"][1][1]
    if not Path(converter).exists():
        raise FileNotFoundError(converter)
    quantizer = paths["commands"][2][0]
    if not Path(quantizer).exists():
        raise FileNotFoundError(quantizer)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    package_root = str(Path(__file__).resolve().parent)
    env["PYTHONPATH"] = package_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    started_at = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    for index, command in enumerate(paths["commands"], start=1):
        log_path = output_dir / f"step-{index}.log"
        with log_path.open("w", encoding="utf-8", newline="\n") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True, env=env)
        if process.returncode:
            raise RuntimeError(f"GGUF export step {index} failed; see {log_path}")
    quantized = paths["quantized_gguf"]
    metadata = {
        **serializable,
        "started_at_utc": started_at,
        "elapsed_s": time.perf_counter() - started,
        "model_bytes": quantized.stat().st_size,
        "completed": True,
    }
    (output_dir / "export_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
