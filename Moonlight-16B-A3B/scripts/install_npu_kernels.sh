#!/usr/bin/env bash
# Install Ascend 910B operator binaries (fixes 561103 / "Op Cast does not has any binary").
#
# Your ModelArts image has toolkit but only ~5MB of ascend910b kernels (need multi-GB).
# This script downloads the matching kernels/ops .run and installs it.
#
# Usage (on the NPU host, with network/proxy):
#   cd Moonlight-16B-A3B-Instruct
#   bash scripts/install_npu_kernels.sh
#
# Then:
#   source "$HOME/Ascend/ascend-toolkit/set_env.sh"   # or path printed by this script
#   python scripts/probe_npu_embedding.py
set -euo pipefail

ARCH="$(uname -m)"
if [[ "$ARCH" != "aarch64" && "$ARCH" != "arm64" ]]; then
  echo "ERROR: this helper targets aarch64 910B; got $ARCH"
  exit 1
fi

# Match the image's CANN major (you are on cann-9.0.1).
CANN_VER="${CANN_VER:-9.0.1}"
# Prefer modern ops package name; fall back to kernels package.
PKG_CANDIDATES=(
  "Ascend-cann-910b-ops_${CANN_VER}_linux-aarch64.run"
  "Ascend-cann-kernels-910b_${CANN_VER}_linux-aarch64.run"
)

# Install beside a writable toolkit. Prefer HOME to avoid needing root on /usr/local.
INSTALL_ROOT="${INSTALL_ROOT:-$HOME/Ascend}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-$HOME/cann_packages}"
mkdir -p "$DOWNLOAD_DIR" "$INSTALL_ROOT"
chmod 750 "$INSTALL_ROOT" 2>/dev/null || true

# Optional ModelArts proxy (override if yours differs).
export http_proxy="${http_proxy:-http://proxyma.huawei.com:8080}"
export https_proxy="${https_proxy:-http://proxyma.huawei.com:8080}"
export HTTP_PROXY="${HTTP_PROXY:-$http_proxy}"
export HTTPS_PROXY="${HTTPS_PROXY:-$https_proxy}"

echo "== arch=$ARCH cann=$CANN_VER install_root=$INSTALL_ROOT =="
echo "== proxy http_proxy=$http_proxy =="

download_one() {
  local name="$1"
  local url="https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%20${CANN_VER}/${name}"
  local out="$DOWNLOAD_DIR/$name"
  if [[ -f "$out" && -s "$out" ]]; then
    echo "== already have $out =="
    echo "$out"
    return 0
  fi
  echo "== downloading $url =="
  if command -v wget >/dev/null 2>&1; then
    wget --header="Referer: https://www.hiascend.com/" -O "$out" "$url" || return 1
  else
    curl -L -H "Referer: https://www.hiascend.com/" -o "$out" "$url" || return 1
  fi
  echo "$out"
}

PKG_PATH=""
for name in "${PKG_CANDIDATES[@]}"; do
  if path="$(download_one "$name")"; then
    PKG_PATH="$path"
    break
  else
    echo "WARN: failed to download $name, trying next..."
    rm -f "$DOWNLOAD_DIR/$name"
  fi
done

if [[ -z "$PKG_PATH" ]]; then
  echo "ERROR: could not download 910b ops/kernels for CANN ${CANN_VER}."
  echo "Download manually from https://www.hiascend.com/developer/download/community/result?module=cann"
  echo "Then: INSTALL_ROOT=$INSTALL_ROOT bash $0  (with the .run already in $DOWNLOAD_DIR)"
  exit 1
fi

chmod +x "$PKG_PATH"

# If system toolkit exists but is not writable, we still install a user toolkit first
# so kernels have a same-path partner. Skip if user already has toolkit under INSTALL_ROOT.
USER_TOOLKIT="$INSTALL_ROOT/ascend-toolkit"
if [[ ! -d "$USER_TOOLKIT" && ! -d "$INSTALL_ROOT/cann" ]]; then
  TK_NAME="Ascend-cann-toolkit_${CANN_VER}_linux-aarch64.run"
  echo "== no user toolkit under $INSTALL_ROOT; downloading toolkit (required same-path for kernels) =="
  if TK_PATH="$(download_one "$TK_NAME")"; then
    chmod +x "$TK_PATH"
    echo "== installing toolkit to $INSTALL_ROOT =="
    bash "$TK_PATH" --install --install-path="$INSTALL_ROOT" --quiet || \
      bash "$TK_PATH" --install --install-path="$INSTALL_ROOT"
  else
    echo "ERROR: need toolkit in the same --install-path as kernels."
    echo "Either install toolkit to $INSTALL_ROOT, or set INSTALL_ROOT to an existing writable Ascend root."
    exit 1
  fi
fi

echo "== installing kernels/ops: $PKG_PATH -> $INSTALL_ROOT =="
bash "$PKG_PATH" --install --install-path="$INSTALL_ROOT" --quiet || \
  bash "$PKG_PATH" --install --install-path="$INSTALL_ROOT"

# Locate set_env / opp after install
SET_ENV=""
for cand in \
  "$INSTALL_ROOT/ascend-toolkit/set_env.sh" \
  "$INSTALL_ROOT/ascend-toolkit/latest/set_env.sh" \
  "$INSTALL_ROOT/cann/set_env.sh" \
  "$INSTALL_ROOT/set_env.sh"
do
  if [[ -f "$cand" ]]; then
    SET_ENV="$cand"
    break
  fi
done

echo "== post-install kernel sizes =="
for opp in \
  "$INSTALL_ROOT/ascend-toolkit/latest/opp" \
  "$INSTALL_ROOT/ascend-toolkit/opp" \
  "$INSTALL_ROOT/cann/opp" \
  /usr/local/Ascend/ascend-toolkit/latest/opp
do
  k="$opp/built-in/op_impl/ai_core/tbe/kernel/ascend910b"
  if [[ -d "$k" ]]; then
    du -sh "$k" || true
  fi
done

if [[ -n "$SET_ENV" ]]; then
  echo ""
  echo "== SUCCESS path =="
  echo "Run EVERY session before python:"
  echo "  source $SET_ENV"
  echo "  source /path/to/.venv/bin/activate"
  echo "  python scripts/probe_npu_embedding.py"
  echo "  MAX_STEPS=20 bash scripts/run_official_toy_train_npu.sh"
else
  echo "WARN: install finished but set_env.sh not found under $INSTALL_ROOT; ls it and source manually."
  ls -la "$INSTALL_ROOT" || true
fi
