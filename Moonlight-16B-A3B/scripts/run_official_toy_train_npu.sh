#!/usr/bin/env bash
# Run official Moonlight toy Muon training on Ascend NPU.
# Matches upstream README flags; uses NPU device adapter.
set -euo pipefail
cd "$(dirname "$0")/.."

source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh 2>/dev/null || true

# Rebuild dataset Arrow if only split parts are present (GitHub <100MB limit).
bash scripts/reassemble_toy_prefetch.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-toy_prefetch/openwebtext-100k}"
TOK_DIR="${TOK_DIR:-toy_prefetch/Qwen2.5-0.5B-tokenizer}"
# Optional: MAX_STEPS=20 bash scripts/run_official_toy_train_npu.sh
MAX_STEPS="${MAX_STEPS:-}"

EXTRA=()
if [[ -d "$DATA_DIR" ]]; then
  EXTRA+=(--local-data-dir "$DATA_DIR")
fi
if [[ -d "$TOK_DIR" ]]; then
  EXTRA+=(--tokenizer-path "$TOK_DIR")
fi
if [[ -n "$MAX_STEPS" ]]; then
  EXTRA+=(--max-steps "$MAX_STEPS")
fi

mkdir -p logs toy_cache
export PYTHONUNBUFFERED=1
"$PYTHON_BIN" -u examples/toy_train_npu.py \
  --model qwen \
  --optimizer muon \
  --dataset openwebtext-100k \
  --hidden_size 896 \
  --lr 1e-3 \
  "${EXTRA[@]}"
