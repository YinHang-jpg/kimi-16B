#!/usr/bin/env bash
# Run official Moonlight toy Muon training on Ascend NPU.
# Matches upstream README flags; uses NPU device adapter.
set -euo pipefail
cd "$(dirname "$0")/.."

# Align OPP/kernel paths (910B3 + cann-9.0.1 often point OPP at a tree without kernel binaries).
# shellcheck disable=SC1091
source scripts/fix_cann_env.sh

# Rebuild dataset Arrow if only split parts are present (GitHub <100MB limit).
bash scripts/reassemble_toy_prefetch.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-toy_prefetch/openwebtext-100k}"
TOK_DIR="${TOK_DIR:-toy_prefetch/Qwen2.5-0.5B-tokenizer}"
# Optional:
#   MAX_STEPS=20 DTYPE=float16 bash scripts/run_official_toy_train_npu.sh
#   MAX_STEPS=20 EMBED_ON_CPU=1 bash scripts/run_official_toy_train_npu.sh
#   MAX_STEPS=20 DEVICE=cpu bash scripts/run_official_toy_train_npu.sh
MAX_STEPS="${MAX_STEPS:-}"
# Prefer float32 on NPU: fp16/bf16 + Muon often NaNs after step0 on Ascend.
DTYPE="${DTYPE:-float32}"
DEVICE="${DEVICE:-auto}"
EMBED_ON_CPU="${EMBED_ON_CPU:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/toy_muon}"
SAVE_STEPS="${SAVE_STEPS:-0}"

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
EXTRA+=(--dtype "$DTYPE")
EXTRA+=(--device "$DEVICE")
EXTRA+=(--output-dir "$OUTPUT_DIR")
EXTRA+=(--save-steps "$SAVE_STEPS")
if [[ "$EMBED_ON_CPU" == "1" || "$EMBED_ON_CPU" == "true" ]]; then
  EXTRA+=(--embed-on-cpu)
fi

mkdir -p logs toy_cache checkpoints
export PYTHONUNBUFFERED=1
"$PYTHON_BIN" -u examples/toy_train_npu.py \
  --model qwen \
  --optimizer muon \
  --dataset openwebtext-100k \
  --hidden_size 896 \
  --lr 1e-3 \
  "${EXTRA[@]}"
