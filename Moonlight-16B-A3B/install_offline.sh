#!/usr/bin/env bash
# One-shot offline install for Ascend / air-gapped hosts.
# Assumes torch + torch_npu are ALREADY installed.
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

"$PYTHON_BIN" - <<'PY'
import importlib
for name in ("torch", "torch_npu"):
    try:
        m = importlib.import_module(name)
        print(f"  OK preinstalled {name}=={getattr(m, '__version__', '?')}")
    except ImportError:
        print(f"  WARN: {name} not importable (expected preinstalled)")
PY

echo "== installing from offline wheels (no network) =="
"$PYTHON_BIN" -m pip install --no-index --find-links="$WHEEL_DIR" -r requirements.txt

echo "== done =="
echo "Next: python scripts/smoke_test.py"
