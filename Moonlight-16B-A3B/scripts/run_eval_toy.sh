#!/usr/bin/env bash
# Evaluate saved toy checkpoint with smoke generate + lm-eval tasks.
set -euo pipefail
cd "$(dirname "$0")/.."

# shellcheck disable=SC1091
source scripts/fix_cann_env.sh 2>/dev/null || true
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh 2>/dev/null || true
if [[ -f "${HOME}/Ascend/ascend-toolkit/set_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/Ascend/ascend-toolkit/set_env.sh"
elif [[ -f "${HOME}/Ascend/ascend-toolkit/latest/set_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/Ascend/ascend-toolkit/latest/set_env.sh"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
CKPT="${CKPT:-checkpoints/toy_muon/final}"
TASKS="${TASKS:-wikitext,mmlu,math500}"
DTYPE="${DTYPE:-float32}"
LIMIT="${LIMIT:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"

EXTRA=()
if [[ "${SKIP_LM_EVAL:-0}" == "1" ]]; then
  EXTRA+=(--skip-lm-eval)
fi
if [[ -n "$LIMIT" && "$LIMIT" != "0" ]]; then
  EXTRA+=(--limit "$LIMIT")
fi

"$PYTHON_BIN" -u scripts/eval_toy_checkpoint.py \
  --checkpoint "$CKPT" \
  --tasks "$TASKS" \
  --dtype "$DTYPE" \
  --batch-size "$BATCH_SIZE" \
  "${EXTRA[@]}"
