from __future__ import annotations

import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from .data_io import CLASS_IDS, load_jsonl
from .teacher import class_token_ids


def ensure_fresh(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def apply_chat_template(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    options = {
        "tokenize": True,
        "return_dict": False,
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": False,
    }
    try:
        return list(tokenizer.apply_chat_template(messages, **options))
    except TypeError:
        options.pop("enable_thinking")
        return list(tokenizer.apply_chat_template(messages, **options))


def common_prefix_length(left: list[int], right: list[int]) -> int:
    count = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        count += 1
    return count


def tokenize_training_records(
    records: list[dict[str, Any]], tokenizer: Any
) -> list[tuple[list[int], int, int, list[float], float]]:
    """Tokenize supervised targets and align teacher logits to the class token."""
    ids = class_token_ids(tokenizer)
    id_set = set(ids)
    class_order = list(CLASS_IDS.values())
    tokenized = []
    for record in records:
        messages = record["messages"]
        prompt_tokens = apply_chat_template(
            tokenizer, messages[:-1], add_generation_prompt=True
        )
        full_tokens = apply_chat_template(
            tokenizer, messages, add_generation_prompt=False
        )
        response_start = common_prefix_length(prompt_tokens, full_tokens)
        class_positions = [
            index
            for index, token in enumerate(full_tokens)
            if index >= response_start and token in id_set
        ]
        if not class_positions:
            raise ValueError(f"Cannot locate class token in {record['sample_id']}")
        class_position = class_positions[0]
        expected_token = ids[class_order.index(record["class_id"])]
        if full_tokens[class_position] != expected_token:
            raise ValueError(f"Unexpected class token in {record['sample_id']}")
        teacher = record.get("teacher_logits")
        if teacher is None:
            teacher_values = [0.0] * len(class_order)
            teacher_mask = 0.0
        else:
            teacher_values = [float(teacher[class_id]) for class_id in class_order]
            if not all(math.isfinite(value) for value in teacher_values):
                raise ValueError("Teacher logits must be finite")
            teacher_mask = float(bool(record.get("teacher_accepted")))
        tokenized.append(
            (full_tokens, response_start, class_position, teacher_values, teacher_mask)
        )
    return tokenized


def distillation_batches(
    dataset: list[tuple[list[int], int, int, list[float], float]],
    batch_size: int,
    max_seq_length: int,
    loop: bool = False,
    seed: int | None = None,
    comm_group: Any = None,
) -> Iterator[tuple[Any, Any, Any, Any, Any, Any]]:
    import mlx.core as mx
    import numpy as np

    if comm_group is not None and comm_group.size() != 1:
        raise ValueError("Distributed distillation is not supported")
    if len(dataset) < batch_size:
        raise ValueError("Dataset is smaller than the batch size")
    rng = np.random.default_rng(seed)
    while True:
        order = rng.permutation(len(dataset))
        for offset in range(0, len(order) - batch_size + 1, batch_size):
            items = [dataset[index] for index in order[offset : offset + batch_size]]
            lengths = [len(item[0]) for item in items]
            if max(lengths) > max_seq_length:
                raise ValueError(f"Sequence length {max(lengths)} exceeds {max_seq_length}")
            width = max(lengths) - 1
            inputs = np.zeros((batch_size, width), dtype=np.int32)
            targets = np.zeros((batch_size, width), dtype=np.int32)
            masks = np.zeros((batch_size, width), dtype=np.float32)
            for row, item in enumerate(items):
                tokens, response_start, _, _, _ = item
                length = len(tokens) - 1
                inputs[row, :length] = tokens[:-1]
                targets[row, :length] = tokens[1:]
                # Only assistant-response tokens contribute to the generation
                # loss. Prompt tokens remain context, not supervised targets.
                masks[row, max(0, response_start - 1) : length] = 1.0
            yield (
                mx.array(inputs),
                mx.array(targets),
                mx.array(masks),
                mx.array([item[2] - 1 for item in items]),
                mx.array([item[3] for item in items]),
                mx.array([item[4] for item in items]),
            )
        if not loop:
            break


def make_distillation_loss(token_ids: list[int], teacher_weight: float, temperature: float):
    import mlx.core as mx
    import mlx.nn as nn

    token_index = mx.array(token_ids)

    def loss(model, inputs, targets, masks, class_positions, teacher_logits, teacher_mask):
        output = model(inputs)
        logits = output.logits if hasattr(output, "logits") else output
        token_loss = nn.losses.cross_entropy(logits, targets)
        denominator = mx.maximum(masks.sum(), mx.array(1.0))
        generation_loss = (token_loss * masks).sum() / denominator
        if teacher_weight <= 0:
            return generation_loss, masks.sum()

        rows = mx.arange(inputs.shape[0])
        # KD is intentionally restricted to the A/B/C/D decision token. The JSON
        # contract is still learned from SFT targets, while teacher logits refine
        # root-cause discrimination only when the teacher gate accepts a sample.
        student_logits = logits[rows, class_positions, :][:, token_index]
        student_scaled = student_logits / temperature
        teacher_scaled = teacher_logits / temperature
        student_log_probs = student_scaled - mx.logsumexp(
            student_scaled, axis=-1, keepdims=True
        )
        teacher_log_probs = teacher_scaled - mx.logsumexp(
            teacher_scaled, axis=-1, keepdims=True
        )
        teacher_probs = mx.exp(teacher_log_probs)
        kl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(axis=-1)
        accepted = teacher_mask.sum()
        distillation_loss = mx.where(
            accepted > 0,
            (kl * teacher_mask).sum() / mx.maximum(accepted, 1),
            mx.array(0.0),
        )
        total = generation_loss + teacher_weight * temperature * temperature * distillation_loss
        return total, masks.sum()

    return loss


def evaluate_distillation(
    model: Any,
    dataset: list[tuple[list[int], int, int, list[float], float]],
    *,
    loss_fn: Any,
    batch_size: int,
    num_batches: int,
    max_seq_length: int,
    seed: int,
) -> float:
    import mlx.core as mx

    model.eval()
    total = mx.array(0.0)
    tokens = mx.array(0)
    batches = distillation_batches(
        dataset,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        loop=False,
        seed=seed,
    )
    for index, batch in enumerate(batches):
        if index >= num_batches:
            break
        value, count = loss_fn(model, *batch)
        total = total + value * count
        tokens = tokens + count
        mx.eval(total, tokens)
        mx.clear_cache()
    return float((total / mx.maximum(tokens, mx.array(1.0))).item())


def train_distillation_loop(
    *,
    model: Any,
    optimizer: Any,
    train_data: list[tuple[list[int], int, int, list[float], float]],
    valid_data: list[tuple[list[int], int, int, list[float], float]],
    loss_fn: Any,
    adapter_file: Path,
    iterations: int,
    batch_size: int,
    grad_accumulation_steps: int,
    max_seq_length: int,
    seed: int,
    steps_per_report: int = 10,
    steps_per_eval: int = 35,
    steps_per_save: int = 35,
    val_batches: int = 32,
) -> None:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_map

    loss_value_and_grad = nn.value_and_grad(model, loss_fn)
    model.train()
    losses = 0.0
    n_tokens = 0.0
    steps = 0
    trained_tokens = 0.0
    train_time = 0.0
    grad_accum = None
    batches = distillation_batches(
        train_data,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        loop=True,
        seed=seed,
    )
    for iteration, batch in zip(range(1, iterations + 1), batches):
        tic = time.perf_counter()
        if iteration == 1 or iteration % steps_per_eval == 0 or iteration == iterations:
            val_loss = evaluate_distillation(
                model,
                valid_data,
                loss_fn=loss_fn,
                batch_size=batch_size,
                num_batches=val_batches,
                max_seq_length=max_seq_length,
                seed=seed + iteration,
            )
            model.train()
            print(
                f"Iter {iteration}: Val loss {val_loss:.3f}, "
                f"Val took {time.perf_counter() - tic:.3f}s",
                flush=True,
            )
            mx.clear_cache()
            tic = time.perf_counter()

        (lvalue, toks), grad = loss_value_and_grad(model, *batch)
        if grad_accum is None:
            grad_accum = grad
        else:
            grad_accum = tree_map(lambda left, right: left + right, grad_accum, grad)
        do_update = iteration % grad_accumulation_steps == 0
        if do_update:
            grad_accum = tree_map(
                lambda value: value / grad_accumulation_steps, grad_accum
            )
            optimizer.update(model, grad_accum)
            grad_accum = None
            mx.eval(model.parameters(), optimizer.state)
        else:
            mx.eval(lvalue, toks, grad_accum)
        mx.clear_cache()

        losses += float(lvalue.item())
        n_tokens += float(toks.item())
        steps += 1
        train_time += time.perf_counter() - tic

        if iteration % steps_per_report == 0 or iteration == iterations:
            train_loss = losses / max(steps, 1)
            it_sec = steps_per_report / max(train_time, 1e-6)
            tokens_sec = n_tokens / max(train_time, 1e-6)
            trained_tokens += n_tokens
            peak_mem = mx.get_peak_memory() / 1e9
            print(
                f"Iter {iteration}: Train loss {train_loss:.3f}, "
                f"Learning Rate {float(optimizer.learning_rate.item()):.3e}, "
                f"It/sec {it_sec:.3f}, "
                f"Tokens/sec {tokens_sec:.3f}, "
                f"Trained Tokens {trained_tokens}, "
                f"Peak mem {peak_mem:.3f} GB",
                flush=True,
            )
            losses = 0.0
            n_tokens = 0.0
            steps = 0
            train_time = 0.0

        if iteration % steps_per_save == 0 or iteration == iterations:
            from mlx.utils import tree_flatten

            adapter_file.parent.mkdir(parents=True, exist_ok=True)
            adapter_weights = dict(tree_flatten(model.trainable_parameters()))
            mx.save_safetensors(str(adapter_file), adapter_weights)
            checkpoint = adapter_file.parent / f"{iteration:07d}_adapters.safetensors"
            mx.save_safetensors(str(checkpoint), adapter_weights)
            print(
                f"Iter {iteration}: Saved adapter weights to {adapter_file} and {checkpoint}.",
                flush=True,
            )
            mx.clear_cache()


def train_distilled(
    *,
    model_path: Path,
    data_dir: Path,
    adapter_dir: Path,
    init_adapter: Path,
    seed: int,
    iterations: int,
    teacher_weight: float = 0.5,
    temperature: float = 2.0,
    learning_rate: float = 1e-5,
    max_seq_length: int = 2048,
) -> dict[str, Any]:
    import mlx.core as mx
    import mlx.optimizers as optim
    import numpy as np
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers, print_trainable_parameters
    from mlx_lm.utils import save_config

    ensure_fresh(adapter_dir)
    started_at = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    np.random.seed(seed)
    mx.random.seed(seed)
    mx.set_cache_limit(256 * 1024**2)
    mx.clear_cache()

    model, tokenizer = load(str(model_path), tokenizer_config={"trust_remote_code": True})
    train_data = tokenize_training_records(load_jsonl(data_dir / "train.jsonl"), tokenizer)
    valid_data = tokenize_training_records(load_jsonl(data_dir / "valid.jsonl"), tokenizer)
    longest = max(len(item[0]) for item in (*train_data, *valid_data))
    if longest > max_seq_length:
        raise ValueError(f"Longest sequence {longest} exceeds {max_seq_length}")

    model.freeze()
    lora = {"rank": 8, "scale": 16.0, "dropout": 0.05}
    linear_to_lora_layers(model, -1, lora)
    model.load_weights(str(init_adapter), strict=False)
    print(f"Warm-started LoRA weights from {init_adapter}", flush=True)
    print_trainable_parameters(model)

    token_ids = class_token_ids(tokenizer)
    batch_size = 1
    grad_accumulation_steps = 8
    config = {
        "model": str(model_path),
        "method": "verified_evidence_logit_distillation",
        "fine_tune_type": "lora",
        "num_layers": -1,
        "seed": seed,
        "iterations": iterations,
        "teacher_weight": teacher_weight,
        "temperature": temperature,
        "learning_rate": learning_rate,
        "max_seq_length": max_seq_length,
        "batch_size": batch_size,
        "grad_accumulation_steps": grad_accumulation_steps,
        "lora_parameters": lora,
        "class_token_ids": dict(zip(CLASS_IDS.values(), token_ids, strict=True)),
        "trainer": "clear_cache_distillation_loop",
        "init_adapter": str(init_adapter),
    }
    save_config(config, adapter_dir / "adapter_config.json")
    optimizer = optim.AdamW(learning_rate=learning_rate)
    adapter_file = adapter_dir / "adapters.safetensors"
    print(f"Starting distillation..., iters: {iterations}", flush=True)
    train_distillation_loop(
        model=model,
        optimizer=optimizer,
        train_data=train_data,
        valid_data=valid_data,
        loss_fn=make_distillation_loss(token_ids, teacher_weight, temperature),
        adapter_file=adapter_file,
        iterations=iterations,
        batch_size=batch_size,
        grad_accumulation_steps=grad_accumulation_steps,
        max_seq_length=max_seq_length,
        seed=seed,
    )
    mx.clear_cache()
    metadata = {
        **config,
        "train_samples": len(train_data),
        "validation_samples": len(valid_data),
        "started_at_utc": started_at,
        "elapsed_s": time.perf_counter() - started,
        "completed": True,
    }
    (adapter_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata
