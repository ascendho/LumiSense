#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODELS=()
SHOTS_PER_CLASS="${SHOTS_PER_CLASS:-1}"
FEWSHOT_SEED="${FEWSHOT_SEED:-17}"
DATA_DIR="${DATA_DIR:-$ROOT/tests/fixtures/mini_cares}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/results/baselines}"
DRY_RUN="${DRY_RUN:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODELS+=("$2"); shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

run() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] $*"
  else
    echo "[run] $*"
    "$@"
  fi
}

DEMOS="$OUTPUT_ROOT/fewshot_demos_seed${FEWSHOT_SEED}.json"

echo "=== LumiSense Baselines ==="
echo "data=${DATA_DIR}"
echo "output=${OUTPUT_ROOT}"

run python3 "$STAGE_DIR/traditional_numeric.py" \
  --data-dir "$DATA_DIR" \
  --output "$OUTPUT_ROOT/traditional_numeric.json"

run python3 "$STAGE_DIR/traditional_text.py" \
  --data-dir "$DATA_DIR" \
  --output "$OUTPUT_ROOT/traditional_text.json"

if [[ ${#MODELS[@]} -eq 0 ]]; then
  echo "[skip] no --model supplied, skipping LLM baselines"
  exit 0
fi

if [[ -f "$DEMOS" ]]; then
  echo "[skip] fixed demos exist: ${DEMOS}"
else
  run python3 "$STAGE_DIR/export_fewshot_demos.py" \
    --data-dir "$DATA_DIR" \
    --shots-per-class "$SHOTS_PER_CLASS" \
    --seed "$FEWSHOT_SEED" \
    --output "$DEMOS"
fi

for spec in "${MODELS[@]}"; do
  name="${spec%%=*}"
  model_path="${spec#*=}"
  run python3 "$STAGE_DIR/evaluate_llm_baseline.py" \
    --model "$model_path" \
    --data-dir "$DATA_DIR" \
    --run-name "${name}-4shot" \
    --output-dir "$OUTPUT_ROOT/quality" \
    --demos "$DEMOS"
done

echo "=== Baselines complete ==="
echo "traditional: $OUTPUT_ROOT/traditional_numeric.json"
echo "traditional: $OUTPUT_ROOT/traditional_text.json"
echo "few-shot:    $DEMOS"
echo "llm:         $OUTPUT_ROOT/quality"
