#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL=""
ADAPTER=""
LLAMA_CPP=""
QUANT="${QUANT:-Q4_K_M}"
IMAGE="${IMAGE:-ghcr.io/ggml-org/llama.cpp:full}"
SKIP_BENCH="${SKIP_BENCH:-0}"
DATA_DIR="${DATA_DIR:-$ROOT/tests/fixtures/mini_cares}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/results/quantization}"
DRY_RUN="${DRY_RUN:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --adapter) ADAPTER="$2"; shift 2 ;;
    --llama-cpp) LLAMA_CPP="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$MODEL" || -z "$ADAPTER" || -z "$LLAMA_CPP" ]]; then
  echo "Usage: $0 --model <official-HF-BF16-base> --adapter <WiSE-FT-adapter> --llama-cpp <llama.cpp-dir>" >&2
  exit 1
fi

if [[ -z "${MLX_SUMMARY:-}" ]]; then
  echo "[error] MLX_SUMMARY must point to the WiSE-FT validation summary." >&2
  exit 1
fi

QUANT_LOWER="$(printf '%s' "$QUANT" | tr '[:upper:]' '[:lower:]')"
EXPORT_DIR="$OUTPUT_ROOT/exports/${QUANT_LOWER}"
RUN_NAME="wise-${QUANT_LOWER}"
SERVER="$LLAMA_CPP/build/bin/llama-server"

run() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] $*"
  else
    echo "[run] $*"
    "$@"
  fi
}

echo "=== LumiSense Quantization (quant=${QUANT}) ==="
echo "model=${MODEL}"
echo "adapter=${ADAPTER}"
echo "llama_cpp=${LLAMA_CPP}"
echo "output=${OUTPUT_ROOT}"

run env PYTHONPATH="$STAGE_DIR${PYTHONPATH:+:$PYTHONPATH}" python3 "$STAGE_DIR/export_gguf.py" \
  --model "$MODEL" \
  --adapter "$ADAPTER" \
  --llama-cpp "$LLAMA_CPP" \
  --output-dir "$EXPORT_DIR" \
  --quantization "$QUANT"

run python3 "$STAGE_DIR/evaluate_gguf.py" \
  --server "$SERVER" \
  --model "$EXPORT_DIR/model-${QUANT_LOWER}.gguf" \
  --data-dir "$DATA_DIR" \
  --run-name "$RUN_NAME" \
  --output-dir "$OUTPUT_ROOT/quality"

run python3 "$STAGE_DIR/check_quantization_gate.py" \
  --mlx-summary "$MLX_SUMMARY" \
  --gguf-summary "$OUTPUT_ROOT/quality/${RUN_NAME}_validation_summary.json" \
  --output "$OUTPUT_ROOT/quantization_gate.json"

if [[ "$SKIP_BENCH" == "1" ]]; then
  echo "[skip] SKIP_BENCH=1, gateway benchmarks skipped"
else
  for PROFILE in min_gateway tiny_gateway standard_gateway; do
    run python3 "$STAGE_DIR/benchmark_gateway.py" \
      --model "$EXPORT_DIR/model-${QUANT_LOWER}.gguf" \
      --profile "$PROFILE" \
      --image "$IMAGE" \
      --output "$OUTPUT_ROOT/resources/${PROFILE}.json"
  done
fi

echo "=== Quantization complete ==="
echo "model: $EXPORT_DIR/model-${QUANT_LOWER}.gguf"
echo "gate:  $OUTPUT_ROOT/quantization_gate.json"
