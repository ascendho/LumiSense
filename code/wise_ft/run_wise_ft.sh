#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL=""
SFT_ADAPTER=""
KD_ADAPTER=""
ALPHA="${ALPHA:-0.5}"
DATA_DIR="${DATA_DIR:-$ROOT/tests/fixtures/mini_cares}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/results/wise_ft}"
DRY_RUN="${DRY_RUN:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --sft-adapter) SFT_ADAPTER="$2"; shift 2 ;;
    --kd-adapter) KD_ADAPTER="$2"; shift 2 ;;
    --alpha) ALPHA="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$MODEL" || -z "$SFT_ADAPTER" || -z "$KD_ADAPTER" ]]; then
  echo "Usage: $0 --model <0.8B-model> --sft-adapter <sft-adapter> --kd-adapter <kd-adapter> [--alpha 0.5]" >&2
  exit 1
fi

if [[ ! -f "$DATA_DIR/train.jsonl" || ! -f "$DATA_DIR/validation.jsonl" ]]; then
  echo "[error] Incomplete dataset: $DATA_DIR" >&2
  exit 1
fi

ALPHA_LABEL="$(python3 - "$ALPHA" <<'PY'
import sys
alpha = float(sys.argv[1])
if not 0.0 <= alpha <= 1.0:
    raise SystemExit("alpha must be in [0, 1]")
text = f"{alpha:.3f}".rstrip("0").rstrip(".")
print("a" + text.replace(".", ""))
PY
)"
ADAPTER_DIR="$OUTPUT_ROOT/adapters/08b-wise-${ALPHA_LABEL}"

run() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] $*"
  else
    echo "[run] $*"
    "$@"
  fi
}

echo "=== LumiSense WiSE-FT (alpha=${ALPHA}) ==="
echo "model=${MODEL}"
echo "sft=${SFT_ADAPTER}"
echo "kd=${KD_ADAPTER}"
echo "output=${OUTPUT_ROOT}"

if [[ -d "$ADAPTER_DIR" ]]; then
  echo "[skip] WiSE-FT adapter already exists: $ADAPTER_DIR"
else
  run python3 "$STAGE_DIR/interpolate_lora.py" \
    --sft-adapter "$SFT_ADAPTER" \
    --kd-adapter "$KD_ADAPTER" \
    --alpha "$ALPHA" \
    --output "$ADAPTER_DIR"
fi

run python3 "$STAGE_DIR/evaluate_wise.py" \
  --model "$MODEL" \
  --adapter "$ADAPTER_DIR" \
  --data-dir "$DATA_DIR" \
  --split validation \
  --run-name "08b-wise-${ALPHA_LABEL}" \
  --output-dir "$OUTPUT_ROOT/quality"

echo "=== WiSE-FT complete ==="
echo "adapter: $ADAPTER_DIR"
echo "summary: $OUTPUT_ROOT/quality/08b-wise-${ALPHA_LABEL}_validation_summary.json"
