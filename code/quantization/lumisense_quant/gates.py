from __future__ import annotations

from typing import Any


def quantization_gate(
    mlx_summary: dict[str, Any],
    gguf_summary: dict[str, Any],
    *,
    maximum_macro_f1_drop: float = 0.02,
) -> dict[str, Any]:
    mlx_f1 = float(mlx_summary["metrics"]["macro_f1"])
    gguf_f1 = float(gguf_summary["metrics"]["macro_f1"])
    drop = mlx_f1 - gguf_f1
    strict = float(gguf_summary["metrics"]["strict_output_rate"])
    checks = [
        {
            "name": "macro_f1_drop",
            "value": drop,
            "maximum": maximum_macro_f1_drop,
            "passed": drop <= maximum_macro_f1_drop,
        },
        {
            "name": "gguf_strict_output_rate",
            "value": strict,
            "minimum": 0.98,
            "passed": strict >= 0.98,
        },
    ]
    return {
        "gate": "quantization",
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
    }
