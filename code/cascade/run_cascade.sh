#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EDGE_MODEL=""
SERVER=""
TOKENIZER_DIR=""
CLOUD_PREDICTIONS=""
PORT="${PORT:-18086}"
THREADS="${THREADS:-4}"
DATA_DIR="${DATA_DIR:-$ROOT/tests/fixtures/mini_cares}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/results/cascade}"
DRY_RUN="${DRY_RUN:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --edge-model) EDGE_MODEL="$2"; shift 2 ;;
    --server) SERVER="$2"; shift 2 ;;
    --tokenizer-dir) TOKENIZER_DIR="$2"; shift 2 ;;
    --cloud-predictions) CLOUD_PREDICTIONS="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$EDGE_MODEL" || -z "$SERVER" || -z "$TOKENIZER_DIR" || -z "$CLOUD_PREDICTIONS" ]]; then
  echo "Usage: $0 --edge-model <edge.gguf> --server <llama-server> --tokenizer-dir <tokenizer-dir> --cloud-predictions <cloud_validation_predictions.jsonl>" >&2
  exit 1
fi

if [[ "$DRY_RUN" != "1" && ! -f "$CLOUD_PREDICTIONS" ]]; then
  echo "[error] cloud predictions not found: $CLOUD_PREDICTIONS" >&2
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

echo "=== LumiSense Cascade (validation) ==="
echo "edge=${EDGE_MODEL}"
echo "cloud=${CLOUD_PREDICTIONS}"
echo "output=${OUTPUT_ROOT}"

CONF="$OUTPUT_ROOT/conf_validation.jsonl"
RISK="$OUTPUT_ROOT/risk_validation.json"
CASCADE="$OUTPUT_ROOT/cascade_validation.json"

run python3 "$STAGE_DIR/score_confidence_gguf.py" \
  --model "$EDGE_MODEL" \
  --server "$SERVER" \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --data-dir "$DATA_DIR" \
  --output "$CONF" \
  --port "$PORT" \
  --threads "$THREADS"

run python3 "$STAGE_DIR/select_threshold.py" \
  --confidence "$CONF" \
  --cloud-predictions "$CLOUD_PREDICTIONS" \
  --output "$RISK"

if [[ "$DRY_RUN" == "1" ]]; then
  THRESHOLD="0.8536"
  echo "[dry-run] threshold comes from $RISK; displaying example ${THRESHOLD}"
else
  THRESHOLD="$(python3 - "$RISK" <<'PY'
import json
import sys
print(f"{json.load(open(sys.argv[1]))['selected']['tau']:.4f}")
PY
)"
fi
echo "[freeze] threshold=${THRESHOLD}"

run python3 "$STAGE_DIR/compute_cascade.py" \
  --confidence "$CONF" \
  --cloud-predictions "$CLOUD_PREDICTIONS" \
  --threshold "$THRESHOLD" \
  --output "$CASCADE"

echo "=== Cascade complete ==="
echo "threshold: $THRESHOLD"
echo "risk:      $RISK"
echo "cascade:   $CASCADE"
