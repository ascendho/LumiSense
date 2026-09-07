from __future__ import annotations

import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_safetensors_dir(path: Path) -> dict:
    from safetensors import safe_open

    tensors: dict = {}
    for file in path.glob("*.safetensors"):
        with safe_open(file, framework="pt") as handle:
            for key in handle.keys():
                tensors[key] = handle.get_tensor(key).clone()
    if not tensors:
        raise FileNotFoundError(f"No safetensors found under {path}")
    return tensors


def merge_mlx_lora_into_hf(
    *,
    base_dir: Path,
    adapter_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    from safetensors import safe_open
    from safetensors.torch import save_file

    config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    scale = float(config["lora_parameters"]["scale"])
    weights = load_safetensors_dir(base_dir)
    adapters: dict = {}
    with safe_open(adapter_dir / "adapters.safetensors", framework="pt") as handle:
        for key in handle.keys():
            adapters[key] = handle.get_tensor(key)

    pairs: dict[str, dict[str, object]] = defaultdict(dict)
    for key, value in adapters.items():
        if key.endswith(".lora_a"):
            pairs[key[: -len(".lora_a")]]["a"] = value
        elif key.endswith(".lora_b"):
            pairs[key[: -len(".lora_b")]]["b"] = value
        else:
            raise ValueError(f"Unexpected adapter key: {key}")

    applied = 0
    for prefix, pair in pairs.items():
        if not prefix.startswith("language_model.model."):
            raise ValueError(f"Unsupported adapter prefix: {prefix}")
        hf_key = "model.language_model." + prefix[len("language_model.model.") :] + ".weight"
        if hf_key not in weights:
            raise KeyError(hf_key)
        lora_a = pair["a"].float()
        lora_b = pair["b"].float()
        delta = (scale * lora_b.T) @ lora_a.T
        base_weight = weights[hf_key]
        if tuple(delta.shape) != tuple(base_weight.shape):
            raise ValueError(
                f"LoRA delta shape {tuple(delta.shape)} != weight {tuple(base_weight.shape)} for {hf_key}"
            )
        # Merge the MLX LoRA delta into the official HF base before GGUF
        # conversion. This avoids exporting an already-fused MLX checkpoint whose
        # tensor layout can produce invalid llama.cpp generations.
        weights[hf_key] = (base_weight.float() + delta).to(base_weight.dtype)
        applied += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(weights, str(output_dir / "model.safetensors"))
    for name in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "merges.txt",
        "chat_template.jinja",
        "preprocessor_config.json",
        "generation_config.json",
    ):
        source = base_dir / name
        if source.exists():
            shutil.copy2(source, output_dir / name)
    index = {
        "metadata": {"total_size": int(sum(tensor.nbytes for tensor in weights.values()))},
        "weight_map": {key: "model.safetensors" for key in weights},
    }
    (output_dir / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "applied": applied,
        "scale": scale,
        "formula": "(scale * B.T) @ A.T",
        "base_dir": str(base_dir.resolve()),
        "adapter_dir": str(adapter_dir.resolve()),
    }
    (output_dir / "merge_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def resolve_quantizer(llama_cpp: Path) -> Path:
    candidates = (
        llama_cpp / "build" / "bin" / "llama-quantize",
        llama_cpp / "llama-quantize",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_converter(llama_cpp: Path) -> Path:
    candidates = (
        llama_cpp / "convert_hf_to_gguf.py",
        llama_cpp / "convert-hf-to-gguf.py",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def gguf_export_commands(
    *,
    model_path: Path,
    adapter_path: Path,
    output_dir: Path,
    llama_cpp: Path,
    quantization: str,
    convert_python: Path | None = None,
) -> dict[str, Any]:
    if quantization not in {"Q4_K_M", "Q5_K_M"}:
        raise ValueError("quantization must be Q4_K_M or Q5_K_M")
    python = str(convert_python or Path(sys.executable))
    fused = output_dir / "fused-hf-lora-merge"
    intermediate = output_dir / "model-bf16.gguf"
    quantized = output_dir / f"model-{quantization.lower()}.gguf"
    converter = resolve_converter(llama_cpp)
    quantizer = resolve_quantizer(llama_cpp)
    return {
        "fused_model": fused,
        "intermediate_gguf": intermediate,
        "quantized_gguf": quantized,
        "merge_mode": "official_hf_base_lora_torch",
        "no_mtp": True,
        "commands": [
            # Export path: HF base + LoRA merge -> BF16 GGUF -> quantized GGUF.
            [
                python,
                "-m",
                "lumisense_quant.merge_lora_hf",
                "--base",
                str(model_path),
                "--adapter",
                str(adapter_path),
                "--output",
                str(fused),
            ],
            [
                python,
                str(converter),
                str(fused),
                "--outfile",
                str(intermediate),
                "--outtype",
                "bf16",
                "--no-mtp",
            ],
            [str(quantizer), str(intermediate), str(quantized), quantization],
        ],
    }


def llama_server_command(
    *,
    server: Path,
    model: Path,
    port: int,
    threads: int,
    context_size: int = 2048,
    n_gpu_layers: int = 0,
) -> list[str]:
    return [
        str(server),
        "--model",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        str(context_size),
        "--threads",
        str(threads),
        "--n-gpu-layers",
        str(n_gpu_layers),
    ]
