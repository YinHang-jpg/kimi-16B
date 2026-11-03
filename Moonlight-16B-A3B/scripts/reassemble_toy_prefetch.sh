#!/usr/bin/env bash
# Rebuild GitHub-split Arrow shards under toy_prefetch/.
set -euo pipefail
cd "$(dirname "$0")/.."

ARROW="toy_prefetch/openwebtext-100k/train/data-00000-of-00001.arrow"
if [[ -f "$ARROW" ]]; then
  echo "== arrow already present: $ARROW =="
  exit 0
fi
if [[ ! -f "${ARROW}.part00" ]]; then
  echo "ERROR: missing ${ARROW}.part00 (git pull incomplete?)"
  exit 1
fi

echo "== reassembling $ARROW =="
cat "${ARROW}".part* > "$ARROW"
ls -lh "$ARROW"
echo "REASSEMBLE_OK"
