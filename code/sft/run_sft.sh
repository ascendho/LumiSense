#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL=""
SEED="${SEED:-17}"
ITERATIONS="${ITERATIONS:-350}"
DATA_DIR="${DATA_DIR:-$ROOT/tests/fixtures/mini_cares}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/results/sft}"
DRY_RUN="${DRY_RUN:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "Usage: $0 --model /path/to/Qwen3.5-0.8B-MLX-4bit" >&2
  exit 1
fi

if [[ ! -f "$DATA_DIR/train.jsonl" || ! -f "$DATA_DIR/validation.jsonl" ]]; then
  echo "[error] Incomplete dataset: $DATA_DIR" >&2
  echo "        Expected train.jsonl, validation.jsonl, and labels/." >&2
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

echo "=== LumiSense SFT (seed=${SEED}, iters=${ITERATIONS}) ==="
echo "model=${MODEL}"
echo "data=${DATA_DIR}"
echo "output=${OUTPUT_ROOT}"

run python3 "$STAGE_DIR/prepare_training_data.py" \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_ROOT/training_data"

run python3 "$STAGE_DIR/train_lora.py" \
  --model "$MODEL" \
  --data-dir "$OUTPUT_ROOT/training_data" \
  --adapter-dir "$OUTPUT_ROOT/adapters/08b-sft-seed${SEED}" \
  --seed "$SEED" \
  --iterations "$ITERATIONS"

run python3 "$STAGE_DIR/evaluate_sft.py" \
  --model "$MODEL" \
  --adapter "$OUTPUT_ROOT/adapters/08b-sft-seed${SEED}" \
  --data-dir "$DATA_DIR" \
  --split validation \
  --run-name "08b-sft-seed${SEED}" \
  --output-dir "$OUTPUT_ROOT/quality"

echo "=== SFT complete ==="
echo "adapter: $OUTPUT_ROOT/adapters/08b-sft-seed${SEED}"
echo "summary: $OUTPUT_ROOT/quality/08b-sft-seed${SEED}_validation_summary.json"
