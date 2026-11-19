#!/usr/bin/env bash
# Align / diagnose Ascend CANN env for 910B (fixes common 561103 ParseDynamicKernels).
#
# Usage:
#   source scripts/fix_cann_env.sh
#   python scripts/probe_npu_embedding.py
set -euo pipefail

_toolkit_candidates=(
  # Prefer a user-local full install (toolkit+kernels) when system image lacks binaries.
  "${HOME}/Ascend/ascend-toolkit/latest"
  "${HOME}/Ascend/ascend-toolkit"
  "${HOME}/Ascend/cann"
  /usr/local/Ascend/ascend-toolkit/latest
  /usr/local/Ascend/cann-9.0.1
  /usr/local/Ascend/nnae/latest
)

_source_set_env() {
  local root="$1"
  if [[ -f "$root/bin/setenv.bash" ]]; then
    # shellcheck disable=SC1090
    source "$root/bin/setenv.bash"
    echo "== sourced $root/bin/setenv.bash =="
    return 0
  fi
  if [[ -f "$root/set_env.sh" ]]; then
    # shellcheck disable=SC1090
    source "$root/set_env.sh"
    echo "== sourced $root/set_env.sh =="
    return 0
  fi
  return 1
}

_kernel_bytes() {
  local opp="$1"
  local kdir="$opp/built-in/op_impl/ai_core/tbe/kernel"
  if [[ ! -d "$kdir" ]]; then
    echo 0
    return
  fi
  # Prefer 910b tree (910B3); fall back to whole kernel dir.
  if [[ -d "$kdir/ascend910b" ]]; then
    du -sb "$kdir/ascend910b" 2>/dev/null | awk '{print $1}'
  else
    du -sb "$kdir" 2>/dev/null | awk '{print $1}'
  fi
}

echo "== fix_cann_env: before =="
echo "ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-}"
echo "ASCEND_OPP_PATH=${ASCEND_OPP_PATH:-}"
echo "ASCEND_AICPU_PATH=${ASCEND_AICPU_PATH:-}"

# Source toolkit env first (preferred), then cann-9.0.1 if present.
_sourced=0
for root in "${_toolkit_candidates[@]}"; do
  if [[ -d "$root" ]] && _source_set_env "$root"; then
    _sourced=1
    break
  fi
done
if (( _sourced == 0 )); then
  echo "WARNING: no Ascend set_env found under ${_toolkit_candidates[*]}"
fi

# Pick the OPP tree that actually has 910b kernel BINARIES (not just config/).
# Empty/stripped kernel trees cause: Parse dynamic kernel config fail / 561103.
_best_opp=""
_best_bytes=0
_opp_candidates=(
  "${ASCEND_OPP_PATH:-}"
  "${HOME}/Ascend/ascend-toolkit/latest/opp"
  "${HOME}/Ascend/ascend-toolkit/opp"
  "${HOME}/Ascend/cann/opp"
  /usr/local/Ascend/ascend-toolkit/latest/opp
  /usr/local/Ascend/cann-9.0.1/opp
  /usr/local/Ascend/nnae/latest/opp
)
for opp in "${_opp_candidates[@]}"; do
  [[ -z "$opp" || ! -d "$opp" ]] && continue
  bytes="$(_kernel_bytes "$opp")"
  bytes="${bytes:-0}"
  echo "== opp candidate: $opp  ascend910b_kernel_bytes=$bytes =="
  if (( bytes > _best_bytes )); then
    _best_bytes=$bytes
    _best_opp=$opp
  fi
done

if [[ -n "$_best_opp" ]]; then
  export ASCEND_OPP_PATH="$_best_opp"
  echo "== selected ASCEND_OPP_PATH=$ASCEND_OPP_PATH (kernel_bytes=$_best_bytes) =="
else
  echo "WARNING: no usable ASCEND_OPP_PATH found"
fi

# 910B3 hosts sometimes expose physical IDs 1,2 (not 0). Leave mapping alone unless set.
# export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-1}"

# Healthy 910b kernel trees are multi-GB. <50MB almost always means ops/kernels package missing.
if (( _best_bytes < 50000000 )); then
  echo ""
  echo "ERROR-HINT: Ascend 910b kernel binaries look missing/stripped (${_best_bytes} bytes)."
  echo "  This matches CANN error 561103 / ParseDynamicKernels / GatherV2AiCore."
  echo "  Ask ModelArts admin to install matching package, e.g.:"
  echo "    Ascend-cann-kernels-910b_*_linux-aarch64.run  (for CANN 9.0.1)"
  echo "  or the combined ops package for 910b."
  echo "  Until then use: MAX_STEPS=20 DEVICE=cpu  OR  EMBED_ON_CPU=1"
  echo ""
fi

echo "== fix_cann_env: after =="
echo "ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-}"
echo "ASCEND_OPP_PATH=${ASCEND_OPP_PATH:-}"
echo "ASCEND_AICPU_PATH=${ASCEND_AICPU_PATH:-}"
echo "LD_LIBRARY_PATH head: ${LD_LIBRARY_PATH:-}" | cut -c1-200
