#!/usr/bin/env python3
"""Prefetch artifacts required by official Moonlight toy_train.

Upstream toy script uses:
  - dataset: Elriggs/openwebtext-100k
  - tokenizer: Qwen/Qwen2.5-0.5B

Run on a networked machine, then copy `toy_prefetch/` to the NPU host.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="toy_prefetch")
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    from transformers import Qwen2Tokenizer

    print("== downloading Elriggs/openwebtext-100k ==")
    ds = load_dataset("Elriggs/openwebtext-100k", trust_remote_code=True)
    ds_path = out / "openwebtext-100k"
    ds.save_to_disk(str(ds_path))
    print(f"saved dataset -> {ds_path}")

    print("== downloading Qwen/Qwen2.5-0.5B tokenizer ==")
    tok_path = out / "Qwen2.5-0.5B-tokenizer"
    tok = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
    tok.save_pretrained(str(tok_path))
    print(f"saved tokenizer -> {tok_path}")

    print("PREFETCH_OK")
    print("On NPU host, run toy train with:")
    print(
        "  python examples/toy_train_npu.py --optimizer muon --hidden_size 896 "
        f"--local-data-dir {ds_path} --tokenizer-path {tok_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
