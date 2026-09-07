from __future__ import annotations

import json
import math
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from .data_io import LETTERS


def render_prompt(tokenizer, messages: list[dict[str, str]]) -> str:
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


def server_command(
    *,
    server: Path,
    model: Path,
    port: int,
    threads: int,
    context_size: int,
    extra_args: list[str] | None = None,
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
        *(extra_args or []),
    ]


def wait_for_health(port: int, process: subprocess.Popen, timeout_s: float = 180.0) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("llama-server exited during startup")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except Exception as error:  # noqa: BLE001
            last_error = error
        time.sleep(1.0)
    raise RuntimeError(f"llama-server did not become healthy: {last_error}")


def next_token_probs(port: int, prompt: str, n_probs: int, timeout_s: float = 120.0) -> dict[str, float]:
    payload = json.dumps(
        {
            "prompt": prompt,
            "n_predict": 1,
            "n_probs": n_probs,
            "temperature": 1.0,
            "cache_prompt": True,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        body = json.loads(response.read().decode("utf-8"))
    steps = body.get("completion_probabilities")
    if not steps:
        raise RuntimeError(f"Unexpected llama-server response keys: {sorted(body)}")
    step = steps[0]
    if "top_logprobs" in step:
        return {
            str(item["token"]): math.exp(float(item["logprob"]))
            for item in step["top_logprobs"]
        }
    return {str(item["tok_str"]): float(item["prob"]) for item in step["probs"]}


def restricted_confidence(raw: dict[str, float]) -> tuple[str | None, float | None]:
    letter_probs = {letter: raw[letter] for letter in LETTERS if letter in raw}
    if not letter_probs:
        return None, None
    total = sum(letter_probs.values())
    if total <= 0.0:
        return None, None
    predicted = max(letter_probs, key=letter_probs.__getitem__)
    return predicted, letter_probs[predicted] / total
