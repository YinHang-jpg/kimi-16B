#!/usr/bin/env python3
"""Environment smoke test for Moonlight-16B-A3B.

Default: import deps + load tokenizer/config (cheap, CPU-safe).
Optional: --load-model to load weights and run a tiny generate.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


REQUIRED_MODULES = [
    "torch",
    "transformers",
    "accelerate",
    "peft",
    "datasets",
    "safetensors",
    "einops",
    "tiktoken",
    "loguru",
]


def detect_device(torch_mod):
    if hasattr(torch_mod, "npu") and torch_mod.npu.is_available():
        return torch_mod.device("npu:0"), "npu"
    if torch_mod.cuda.is_available():
        return torch_mod.device("cuda:0"), "cuda"
    return torch_mod.device("cpu"), "cpu"


def check_imports() -> None:
    print("== import check ==")
    for name in REQUIRED_MODULES:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "unknown")
        print(f"  OK  {name}=={version}")

    try:
        npu = importlib.import_module("torch_npu")
        print(f"  OK  torch_npu=={getattr(npu, '__version__', 'unknown')}")
    except ImportError:
        print("  --  torch_npu not installed (expected on non-Ascend hosts)")


def check_tokenizer_and_config(model_dir: Path) -> None:
    from transformers import AutoConfig, AutoTokenizer

    print("== tokenizer / config ==")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model dir not found: {model_dir}")

    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    print(f"  model_type={config.model_type}")
    print(f"  vocab_size={getattr(config, 'vocab_size', '?')}")
    print(f"  tokenizer_class={tokenizer.__class__.__name__}")

    sample = tokenizer("1+1=", return_tensors="pt")
    print(f"  encode_ok keys={list(sample.keys())} shape={tuple(sample['input_ids'].shape)}")


def check_model_generate(model_dir: Path, max_new_tokens: int) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("== model load + generate ==")
    device, backend = detect_device(torch)
    print(f"  device={device} backend={backend}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16 if backend != "cpu" else torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    inputs = tokenizer("1+1=", return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"  generate_ok: {text!r}")


def main() -> int:
    model_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=model_root,
        help="Path to Moonlight-16B-A3B weights/config (default: this package dir)",
    )
    parser.add_argument(
        "--load-model",
        action="store_true",
        help="Also load weights and run a tiny generate (needs lots of RAM/NPU)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()

    check_imports()
    check_tokenizer_and_config(args.model_dir)

    if args.load_model:
        check_model_generate(args.model_dir, args.max_new_tokens)
    else:
        print("== skip model load (pass --load-model to enable) ==")

    print("SMOKE_OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - surface clear failure for CI/smoke
        print(f"SMOKE_FAIL: {exc}", file=sys.stderr)
        raise
