#!/usr/bin/env bash
# One-shot offline install for Ascend / air-gapped hosts (via GitHub wheels).
# Installs torch(+cpu), torch_npu, and the HuggingFace training stack.
#
#   cd Moonlight-16B-A3B
#   bash install_offline.sh
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -d offline_wheels ]]; then
  echo "ERROR: offline_wheels/ missing. Run: git pull"
  exit 1
fi

ARCH="$(uname -m)"
case "$ARCH" in
  x86_64|amd64) WHEEL_DIR="offline_wheels/linux_x86_64" ;;
  aarch64|arm64) WHEEL_DIR="offline_wheels/linux_aarch64" ;;
  *)
    echo "ERROR: unsupported arch: $ARCH"
    exit 1
    ;;
esac

if [[ ! -d "$WHEEL_DIR" ]]; then
  echo "ERROR: missing $WHEEL_DIR"
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=python
fi

echo "== interpreter: $($PYTHON_BIN -V) =="
echo "== arch: $ARCH =="
echo "== wheel dir: $WHEEL_DIR =="

# Reassemble GitHub-split wheels (>100MB files are stored as *.whl.part00, .part01, ...)
shopt -s nullglob
for part in "$WHEEL_DIR"/*.whl.part00; do
  prefix="${part%.part00}"
  out="${prefix}"
  if [[ -f "$out" ]]; then
    echo "== skip rebuild (exists): $(basename "$out") =="
    continue
  fi
  echo "== rebuilding $(basename "$out") from parts =="
  cat "${prefix}".part* > "$out"
done
shopt -u nullglob

echo "== installing torch + torch_npu from offline wheels =="
mapfile -t TORCH_WHLS < <(ls -1 "$WHEEL_DIR"/torch-2.10.0*cpu*.whl 2>/dev/null || true)
mapfile -t NPU_WHLS < <(ls -1 "$WHEEL_DIR"/torch_npu-*.whl 2>/dev/null || true)

if ((${#TORCH_WHLS[@]} == 0)); then
  echo "ERROR: torch 2.10.0+cpu wheel not found (did part reassembly fail?)"
  exit 1
fi
if ((${#NPU_WHLS[@]} == 0)); then
  echo "ERROR: torch_npu wheel not found"
  exit 1
fi

"$PYTHON_BIN" -m pip install --no-index --find-links="$WHEEL_DIR" "${TORCH_WHLS[0]}"
"$PYTHON_BIN" -m pip install --no-index --find-links="$WHEEL_DIR" "${NPU_WHLS[0]}"

echo "== installing remaining requirements =="
"$PYTHON_BIN" -m pip install --no-index --find-links="$WHEEL_DIR" -r requirements.txt

echo "== done =="
echo "Next: python scripts/smoke_test.py"

# Note: this file must use LF line endings for Linux bash.
