from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .data_io import CLASS_IDS, ID_TO_LABEL, LABELS, load_jsonl, select_balanced_pairs
from .metrics import evidence_checks, parse_response, score_predictions
from .prompt import inference_messages, training_record

CLASS_LOGIT_PREFILL = '{"class_id":"'


def softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    exps = [math.exp(value - maximum) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def clean_generation(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def class_token_ids(tokenizer: Any) -> list[int]:
    token_ids = []
    for class_id in CLASS_IDS.values():
        encoded = tokenizer.encode(class_id, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"Class ID {class_id} is not one token: {encoded}")
        token_ids.append(encoded[0])
    if len(set(token_ids)) != len(token_ids):
        raise ValueError("Class IDs do not have unique token IDs")
    return token_ids


def apply_generation_template(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    options = {
        "tokenize": True,
        "return_dict": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    try:
        return list(tokenizer.apply_chat_template(messages, **options))
    except TypeError:
        options.pop("enable_thinking")
        return list(tokenizer.apply_chat_template(messages, **options))


def rendered_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
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


def restricted_teacher_logits(
    model: Any, batch: Any, length: int, token_ids: list[int]
) -> Any:
    import mlx.core as mx

    text_model = model.language_model
    hidden = text_model.model(batch)
    final_state = hidden[0, length - 1, :]
    if text_model.args.tie_word_embeddings:
        vocabulary_logits = text_model.model.embed_tokens.as_linear(final_state)
    else:
        vocabulary_logits = text_model.lm_head(final_state)
    return vocabulary_logits[mx.array(token_ids)]


def teacher_response_accepted(
    response: dict[str, Any] | None,
    record: dict[str, Any],
    label: dict[str, Any],
) -> bool:
    if response is None or response["class_id"] != label["class_id"]:
        return False
    checks = evidence_checks(
        response,
        record["diagnostic_snapshot"],
        set(label["acceptable_evidence_paths"]),
    )
    return bool(checks) and all(item["grounded"] and item["relevant"] for item in checks)


def teacher_quality_summary(
    *,
    records: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    model_path: Path,
    split: str,
    min_probability: float,
    samples_per_class: int | None,
    selection_seed: int,
) -> dict[str, Any]:
    annotation_by_id = {item["sample_id"]: item for item in annotations}
    record_ids = {item["sample_id"] for item in records}
    if len(annotation_by_id) != len(annotations) or set(annotation_by_id) != record_ids:
        raise ValueError("Teacher annotations must cover the selected records exactly")

    processed_by_class = Counter(item["root_cause"] for item in labels)
    accepted_by_class = Counter(
        item["label"] for item in annotations if item.get("teacher_accepted")
    )
    class_quality = {}
    for root_cause in LABELS:
        processed = processed_by_class[root_cause]
        accepted = accepted_by_class[root_cause]
        class_quality[root_cause] = {
            "samples": processed,
            "accepted": accepted,
            "accepted_rate": accepted / processed if processed else 0.0,
        }

    predictions = []
    for record in records:
        annotation = annotation_by_id[record["sample_id"]]
        predictions.append(
            {
                "sample_id": record["sample_id"],
                "response": clean_generation(annotation.get("teacher_raw_response", "")),
                "latency_s": float(annotation.get("teacher_latency_s", 0.0)),
            }
        )
    raw_metrics = score_predictions(records, labels, predictions)
    accepted_count = sum(bool(item.get("teacher_accepted")) for item in annotations)
    accepted_rates = [class_quality[label]["accepted_rate"] for label in LABELS]
    return {
        "model": str(model_path),
        "split": split,
        "samples": len(annotations),
        "accepted": accepted_count,
        "accepted_rate": accepted_count / len(annotations) if annotations else 0.0,
        "minimum_class_accepted_rate": min(accepted_rates, default=0.0),
        "per_class": class_quality,
        "raw_metrics": raw_metrics,
        "minimum_probability": min_probability,
        "samples_per_class": samples_per_class,
        "selection_seed": selection_seed,
        "test_opened": False,
    }


def annotate_dataset(
    *,
    model_path: Path,
    records_path: Path,
    labels_path: Path,
    output_path: Path,
    summary_path: Path,
    min_probability: float = 0.45,
    max_tokens: int = 128,
    max_samples: int | None = None,
    samples_per_class: int | None = None,
    selection_seed: int = 17,
    split: str = "train",
    resume: bool = False,
) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    records = load_jsonl(records_path)
    labels = load_jsonl(labels_path)
    if max_samples is not None and samples_per_class is not None:
        raise ValueError("max_samples and samples_per_class are mutually exclusive")
    records, labels = select_balanced_pairs(
        records,
        labels,
        per_class=samples_per_class,
        seed=selection_seed,
    )
    if max_samples is not None:
        records = records[:max_samples]
        selected_ids = {item["sample_id"] for item in records}
        labels = [item for item in labels if item["sample_id"] in selected_ids]
    label_by_id = {item["sample_id"]: item for item in labels}

    existing = load_jsonl(output_path) if output_path.exists() and resume else []
    if output_path.exists() and not resume:
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    completed = {item["sample_id"] for item in existing}
    if len(completed) != len(existing):
        raise ValueError("Duplicate IDs in resumed teacher annotations")
    selected_ids = {item["sample_id"] for item in records}
    if not completed <= selected_ids:
        raise ValueError("Resumed annotations contain IDs outside the selected subset")

    model, tokenizer = load(str(model_path))
    class_ids = list(CLASS_IDS.values())
    token_ids = class_token_ids(tokenizer)
    sampler = make_sampler(temp=0.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if existing else "w"
    with output_path.open(mode, encoding="utf-8", newline="\n") as handle:
        for index, record in enumerate(records, start=1):
            if record["sample_id"] in completed:
                continue
            label = label_by_id[record["sample_id"]]
            messages = inference_messages(record, teacher=True)
            tokens = apply_generation_template(tokenizer, messages)
            tokens.extend(tokenizer.encode(CLASS_LOGIT_PREFILL, add_special_tokens=False))
            batch = mx.array([tokens])

            start = time.perf_counter()
            logits_array = restricted_teacher_logits(model, batch, len(tokens), token_ids)
            mx.eval(logits_array)
            logits = [float(value) for value in logits_array.tolist()]
            probabilities = softmax(logits)
            predicted_index = max(range(len(logits)), key=logits.__getitem__)
            predicted_id = class_ids[predicted_index]

            raw = generate(
                model,
                tokenizer,
                prompt=rendered_prompt(tokenizer, messages),
                max_tokens=max_tokens,
                sampler=sampler,
                verbose=False,
            )
            response = parse_response(clean_generation(raw))
            response_ok = teacher_response_accepted(response, record, label)
            accepted = (
                predicted_id == label["class_id"]
                and probabilities[predicted_index] >= min_probability
                and response_ok
            )
            target = response if accepted else label["target"]
            annotated = training_record(
                record,
                label,
                target=target,
                teacher_logits=dict(zip(class_ids, logits, strict=True)),
                teacher_accepted=accepted,
            )
            annotated.update(
                teacher_prediction=predicted_id,
                teacher_probabilities=dict(zip(class_ids, probabilities, strict=True)),
                teacher_raw_response=raw,
                teacher_latency_s=time.perf_counter() - start,
            )
            handle.write(json.dumps(annotated, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            print(
                f"[{index:04d}/{len(records):04d}] {record['sample_id']} "
                f"truth={label['class_id']} teacher={predicted_id} accepted={accepted}",
                flush=True,
            )

    annotations = load_jsonl(output_path)
    summary = teacher_quality_summary(
        records=records,
        labels=labels,
        annotations=annotations,
        model_path=model_path,
        split=split,
        min_probability=min_probability,
        samples_per_class=samples_per_class,
        selection_seed=selection_seed,
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
