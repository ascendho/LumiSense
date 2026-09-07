from __future__ import annotations

import json
import re
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

PROFILES = {
    "min_gateway": {"cpus": 1, "memory": "1g"},
    "tiny_gateway": {"cpus": 2, "memory": "2g"},
    "standard_gateway": {"cpus": 4, "memory": "4g"},
}


def parse_memory_bytes(value: str) -> int:
    used = value.split("/", 1)[0].strip()
    match = re.fullmatch(r"([0-9.]+)\s*([KMGTP]?i?B)", used, re.IGNORECASE)
    if not match:
        raise ValueError(f"Unsupported Docker memory value: {value}")
    amount = float(match.group(1))
    unit = match.group(2).lower()
    multipliers = {
        "b": 1,
        "kb": 1000,
        "kib": 1024,
        "mb": 1000**2,
        "mib": 1024**2,
        "gb": 1000**3,
        "gib": 1024**3,
        "tb": 1000**4,
        "tib": 1024**4,
    }
    return int(amount * multipliers[unit])


def docker_benchmark_command(
    *,
    model_path: Path,
    profile: str,
    image: str,
    container_name: str,
    repetitions: int = 3,
) -> list[str]:
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile: {profile}")
    limits = PROFILES[profile]
    return [
        "docker",
        "run",
        "--name",
        container_name,
        "--platform",
        "linux/arm64",
        "--network",
        "none",
        "--cpus",
        str(limits["cpus"]),
        "--memory",
        limits["memory"],
        "--memory-swap",
        limits["memory"],
        "--volume",
        f"{model_path.parent.resolve()}:/models:ro",
        "--entrypoint",
        "/usr/local/bin/llama-bench",
        image,
        "-m",
        f"/models/{model_path.name}",
        "-p",
        "512",
        "-n",
        "128",
        "-t",
        str(limits["cpus"]),
        "-r",
        str(repetitions),
        "-o",
        "json",
    ]


def run_constrained_benchmark(
    *,
    model_path: Path,
    profile: str,
    output_path: Path,
    image: str,
    repetitions: int = 3,
) -> dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    container_name = f"lumisense-bench-{uuid.uuid4().hex[:10]}"
    command = docker_benchmark_command(
        model_path=model_path,
        profile=profile,
        image=image,
        container_name=container_name,
        repetitions=repetitions,
    )
    peak_memory = 0
    stop = threading.Event()

    def monitor() -> None:
        nonlocal peak_memory
        while not stop.wait(0.2):
            result = subprocess.run(
                ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", container_name],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0 and result.stdout.strip():
                try:
                    peak_memory = max(peak_memory, parse_memory_bytes(result.stdout.strip()))
                except ValueError:
                    pass

    started = time.perf_counter()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    stdout, stderr = process.communicate()
    stop.set()
    thread.join(timeout=2)
    elapsed = time.perf_counter() - started
    inspect = subprocess.run(
        ["docker", "inspect", container_name, "--format", "{{.State.OOMKilled}}"],
        capture_output=True,
        text=True,
    )
    oom_killed = inspect.stdout.strip().lower() == "true"
    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
    try:
        parsed_output: Any = json.loads(stdout)
    except json.JSONDecodeError:
        parsed_output = stdout
    summary = {
        "profile": profile,
        "limits": PROFILES[profile],
        "model": str(model_path.resolve()),
        "model_bytes": model_path.stat().st_size,
        "image": image,
        "command": command,
        "return_code": process.returncode,
        "oom_killed": oom_killed,
        "peak_memory_bytes": peak_memory,
        "wall_time_s": elapsed,
        "benchmark": parsed_output,
        "stderr": stderr,
        "claim_scope": "resource-constrained Apple-Silicon proxy; not Raspberry Pi",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if process.returncode and not oom_killed:
        raise RuntimeError(f"Docker benchmark failed: {stderr.strip()}")
    return summary
