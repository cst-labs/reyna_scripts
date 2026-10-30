#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PAIRWISE_INPUT="${1:-$ROOT_DIR/data/KTAS/test_pairwise.csv}"
RUN_ID="${RUN_ID:-ktas_pairwise_ollama_batch}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-999999}"
K_SHOTS="${K_SHOTS:-6}"
SERIALIZATION="${SERIALIZATION:-natural}"
STRATEGY="${STRATEGY:-KATE-PAIRWISE}"
SAVE_EVERY="${SAVE_EVERY:-25}"
ENCODER_RETRIES="${ENCODER_RETRIES:-3}"
ENCODER_RETRY_DELAY="${ENCODER_RETRY_DELAY:-2}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv313/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python binary not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -f "$PAIRWISE_INPUT" ]]; then
  echo "Pairwise input CSV not found: $PAIRWISE_INPUT" >&2
  exit 1
fi

OLLAMA_MODELS=()
while IFS= read -r MODEL; do
  [[ -n "$MODEL" ]] && OLLAMA_MODELS+=("$MODEL")
done < <(ollama list | awk 'NR>1 && $1 != "" {print $1}')

if [[ ${#OLLAMA_MODELS[@]} -eq 0 ]]; then
  echo "No Ollama models found." >&2
  exit 1
fi

for MODEL in "${OLLAMA_MODELS[@]}"; do
  echo "Running pairwise KTAS for model: $MODEL"
  "$PYTHON_BIN" llm-predict-triage.py \
    --dataset Triage-KTAS-Pairwise \
    --strategy "$STRATEGY" \
    --model "$MODEL" \
    --k_shots "$K_SHOTS" \
    --serialization "$SERIALIZATION" \
    --pairwise_input "$PAIRWISE_INPUT" \
    --start "$START_INDEX" \
    --end "$END_INDEX" \
    --save_every "$SAVE_EVERY" \
    --encoder_retries "$ENCODER_RETRIES" \
    --encoder_retry_delay "$ENCODER_RETRY_DELAY" \
    --run_id "$RUN_ID" \
    --auto_resume
done
