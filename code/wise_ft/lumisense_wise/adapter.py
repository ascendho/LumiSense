from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def alpha_label(alpha: float) -> str:
    text = f"{alpha:.3f}".rstrip("0").rstrip(".")
    return "a" + text.replace(".", "")


def read_config(adapter_dir: Path) -> dict[str, Any]:
    path = adapter_dir / "adapter_config.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def adapter_weights_path(adapter_dir: Path) -> Path:
    path = adapter_dir / "adapters.safetensors"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def interpolate_lora_adapters(
    *,
    sft_adapter: Path,
    kd_adapter: Path,
    output_dir: Path,
    alpha: float,
) -> dict[str, Any]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")

    import mlx.core as mx

    configs = [read_config(sft_adapter), read_config(kd_adapter)]
    lora_params = [config["lora_parameters"] for config in configs]
    ranks = {int(params["rank"]) for params in lora_params}
    scales = {float(params["scale"]) for params in lora_params}
    if len(ranks) != 1 or len(scales) != 1:
        raise ValueError(f"Mixed LoRA configs: ranks={ranks}, scales={scales}")
    rank = ranks.pop()
    scale = scales.pop()

    loaded = [
        mx.load(str(adapter_weights_path(sft_adapter))),
        mx.load(str(adapter_weights_path(kd_adapter))),
    ]
    keys = set(loaded[0])
    if set(loaded[1]) != keys:
        raise ValueError("SFT and KD adapters have different tensor keys")

    sft_weight = 1.0 - alpha
    kd_weight = alpha
    combined = {}
    for key in sorted(keys):
        sft_tensor = loaded[0][key]
        kd_tensor = loaded[1][key]
        if sft_tensor.shape != kd_tensor.shape:
            raise ValueError(f"Tensor shape mismatch for {key}: {sft_tensor.shape} vs {kd_tensor.shape}")
        # Concatenating LoRA factors doubles the rank and represents
        # (1-alpha) * Delta_SFT + alpha * Delta_KD exactly at inference time.
        if key.endswith("lora_a"):
            combined[key] = mx.concatenate([sft_tensor, kd_tensor], axis=1)
        elif key.endswith("lora_b"):
            combined[key] = mx.concatenate(
                [sft_weight * sft_tensor, kd_weight * kd_tensor],
                axis=0,
            )
        else:
            raise ValueError(f"Unexpected tensor key: {key}")

    output_dir.mkdir(parents=True, exist_ok=False)
    mx.save_safetensors(str(output_dir / "adapters.safetensors"), combined)
    config = dict(configs[0])
    config["lora_parameters"] = {
        "rank": rank * 2,
        "scale": scale,
        "dropout": 0.0,
    }
    (output_dir / "adapter_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "method": "wise_ft_lora_concat",
        "alpha": alpha,
        "weights": {"sft": sft_weight, "kd": kd_weight},
        "ingredients": {
            "sft_adapter": str(sft_adapter),
            "kd_adapter": str(kd_adapter),
        },
        "rank": config["lora_parameters"]["rank"],
        "scale": config["lora_parameters"]["scale"],
    }
    (output_dir / "wise_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata
