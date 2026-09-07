#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TEACHER_MODEL=""
STUDENT_MODEL=""
INIT_ADAPTER=""
SEED="${SEED:-17}"
ITERATIONS="${ITERATIONS:-250}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
TEACHER_WEIGHT="${TEACHER_WEIGHT:-0.5}"
TEMPERATURE="${TEMPERATURE:-2.0}"
CKPTS="${CKPTS:-35 105 175 250}"
DATA_DIR="${DATA_DIR:-$ROOT/tests/fixtures/mini_cares}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/results/kd}"
DRY_RUN="${DRY_RUN:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --teacher-model) TEACHER_MODEL="$2"; shift 2 ;;
    --student-model) STUDENT_MODEL="$2"; shift 2 ;;
    --init-adapter) INIT_ADAPTER="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$TEACHER_MODEL" || -z "$STUDENT_MODEL" || -z "$INIT_ADAPTER" ]]; then
  echo "Usage: $0 --teacher-model <teacher-model> --student-model <student-model> --init-adapter <sft-adapter>" >&2
  exit 1
fi

if [[ ! -f "$DATA_DIR/train.jsonl" || ! -f "$DATA_DIR/validation.jsonl" ]]; then
  echo "[error] Incomplete dataset: $DATA_DIR" >&2
  exit 1
fi

run() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] $*"
  else
    echo "[run] $*"
    "$@"
  fi
}

ADAPTER_DIR="$OUTPUT_ROOT/adapters/08b-kd-seed${SEED}"

echo "=== LumiSense KD (seed=${SEED}, iters=${ITERATIONS}) ==="
echo "teacher=${TEACHER_MODEL}"
echo "student=${STUDENT_MODEL}"
echo "init_adapter=${INIT_ADAPTER}"
echo "output=${OUTPUT_ROOT}"

if [[ -f "$OUTPUT_ROOT/teacher/train.jsonl" && -f "$OUTPUT_ROOT/teacher/train_summary.json" ]]; then
  echo "[skip] teacher train annotations already exist"
else
  run python3 "$STAGE_DIR/annotate_teacher.py" \
    --model "$TEACHER_MODEL" \
    --data-dir "$DATA_DIR" \
    --split train \
    --output "$OUTPUT_ROOT/teacher/train.jsonl" \
    --summary "$OUTPUT_ROOT/teacher/train_summary.json"
fi

if [[ -f "$OUTPUT_ROOT/teacher/validation.jsonl" && -f "$OUTPUT_ROOT/teacher/validation_summary.json" ]]; then
  echo "[skip] teacher validation annotations already exist"
else
  run python3 "$STAGE_DIR/annotate_teacher.py" \
    --model "$TEACHER_MODEL" \
    --data-dir "$DATA_DIR" \
    --split validation \
    --output "$OUTPUT_ROOT/teacher/validation.jsonl" \
    --summary "$OUTPUT_ROOT/teacher/validation_summary.json"
fi

run python3 "$STAGE_DIR/check_teacher_gate.py" \
  --teacher-train "$OUTPUT_ROOT/teacher/train_summary.json" \
  --teacher-validation "$OUTPUT_ROOT/teacher/validation_summary.json" \
  --output "$OUTPUT_ROOT/teacher_gate.json"

run python3 "$STAGE_DIR/prepare_distill_data.py" \
  --data-dir "$DATA_DIR" \
  --teacher-file "$OUTPUT_ROOT/teacher/train.jsonl" \
  --output-dir "$OUTPUT_ROOT/training_data"

run python3 "$STAGE_DIR/train_distill_lora.py" \
  --model "$STUDENT_MODEL" \
  --data-dir "$OUTPUT_ROOT/training_data" \
  --adapter-dir "$ADAPTER_DIR" \
  --init-adapter "$INIT_ADAPTER" \
  --seed "$SEED" \
  --iterations "$ITERATIONS" \
  --teacher-weight "$TEACHER_WEIGHT" \
  --temperature "$TEMPERATURE" \
  --learning-rate "$LEARNING_RATE"

for CKPT in $CKPTS; do
  CKPT_FILE="$ADAPTER_DIR/$(printf '%07d' "$CKPT")_adapters.safetensors"
  if [[ "$DRY_RUN" == "1" || -f "$CKPT_FILE" ]]; then
    run python3 "$STAGE_DIR/evaluate_kd.py" \
      --model "$STUDENT_MODEL" \
      --adapter "$CKPT_FILE" \
      --data-dir "$DATA_DIR" \
      --split validation \
      --run-name "08b-kd-seed${SEED}-iter${CKPT}" \
      --output-dir "$OUTPUT_ROOT/quality"
  else
    echo "[warn] missing checkpoint: $CKPT_FILE"
  fi
done

run python3 "$STAGE_DIR/evaluate_kd.py" \
  --model "$STUDENT_MODEL" \
  --adapter "$ADAPTER_DIR" \
  --data-dir "$DATA_DIR" \
  --split validation \
  --run-name "08b-kd-seed${SEED}" \
  --output-dir "$OUTPUT_ROOT/quality"

echo "=== KD complete ==="
echo "adapter: $ADAPTER_DIR"
echo "summaries: $OUTPUT_ROOT/quality"
