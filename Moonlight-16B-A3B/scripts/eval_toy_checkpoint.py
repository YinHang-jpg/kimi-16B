#!/usr/bin/env python3
"""Evaluate a saved toy checkpoint (HF format under checkpoints/toy_muon/final).

Supported:
  - smoke generate
  - wikitext word perplexity (lm-eval task: wikitext)
  - mmlu (lm-eval)
  - math500 / MATH (lm-eval task name depends on version: math500 or minerva_math)
  - livecodebench (optional; requires livecodebench + lm-eval plugin if available)

IMPORTANT:
  The official toy is a ~331M Qwen2 LM trained on openwebtext-100k.
  It will NOT match Moonlight-16B paper scores on MMLU/MATH/LiveCodeBench.
  Use these numbers to compare Muon vs AdamW / your Muon under the same setup.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def detect_device() -> str:
    try:
        import torch

        if hasattr(torch, "npu") and torch.npu.is_available():
            return "npu:0"
        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def smoke_generate(ckpt: Path, device: str, dtype: str, max_new_tokens: int) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("== smoke generate ==")
    tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    torch_dtype = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "auto": torch.float32 if device == "cpu" else torch.bfloat16,
    }[dtype]
    model = AutoModelForCausalLM.from_pretrained(
        ckpt, trust_remote_code=True, torch_dtype=torch_dtype
    )
    model.to(device)
    model.eval()
    prompt = "The capital of France is"
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = tok.decode(out[0], skip_special_tokens=True)
    print(f"  prompt={prompt!r}")
    print(f"  output={text!r}")
    return {"prompt": prompt, "output": text}


def run_lm_eval(
    ckpt: Path,
    tasks: str,
    device: str,
    dtype: str,
    batch_size: int,
    output_path: Path,
    limit: int | None,
) -> dict:
    print(f"== lm_eval tasks={tasks} ==")
    cmd = [
        sys.executable,
        "-m",
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        f"pretrained={ckpt},trust_remote_code=True,dtype={dtype}",
        "--tasks",
        tasks,
        "--device",
        device,
        "--batch_size",
        str(batch_size),
        "--output_path",
        str(output_path),
        "--log_samples",
    ]
    if limit is not None and limit > 0:
        cmd.extend(["--limit", str(limit)])
    print("  exec:", " ".join(cmd))
    env = os.environ.copy()
    # Avoid accidental Hub downloads when offline caches are prepared.
    env.setdefault("HF_HUB_OFFLINE", env.get("HF_HUB_OFFLINE", "0"))
    proc = subprocess.run(cmd, env=env, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"lm_eval failed with exit code {proc.returncode}")
    # lm_eval writes a results JSON under output_path; collect newest.
    results = {"lm_eval_output": str(output_path), "returncode": proc.returncode}
    if output_path.is_dir():
        jsons = sorted(output_path.rglob("*.json"), key=lambda p: p.stat().st_mtime)
        if jsons:
            results["latest_json"] = str(jsons[-1])
            try:
                results["summary"] = json.loads(jsons[-1].read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                results["summary_error"] = str(exc)
    return results


def resolve_tasks(requested: list[str]) -> str:
    """Map user-facing names to lm-eval task ids (best-effort)."""
    mapping = {
        "wikitext": "wikitext",
        "mmlu": "mmlu",
        "math500": "math500",
        "math": "math500",
        "livecodebench": "livecodebench",
        "lcb": "livecodebench",
    }
    out = []
    for t in requested:
        key = t.strip().lower()
        out.append(mapping.get(key, t.strip()))
    return ",".join(out)


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=repo / "checkpoints" / "toy_muon" / "final",
        help="HF checkpoint dir saved by toy_train_npu.py",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="wikitext,mmlu,math500",
        help="comma list: wikitext,mmlu,math500,livecodebench (needs lm_eval)",
    )
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="optional per-task sample cap for a quick smoke (0 = full)",
    )
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--skip-lm-eval", action="store_true")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=repo / "logs" / "eval_toy",
    )
    args = parser.parse_args()

    ckpt = args.checkpoint
    if not ckpt.is_dir():
        raise FileNotFoundError(
            f"checkpoint not found: {ckpt}\n"
            "Re-run training with the updated toy_train_npu.py so it saves "
            "checkpoints/toy_muon/final/"
        )

    device = detect_device()
    print(f"device={device}")
    print(f"checkpoint={ckpt}")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"checkpoint": str(ckpt), "device": device}

    if not args.skip_smoke:
        report["smoke"] = smoke_generate(ckpt, device, args.dtype, max_new_tokens=16)

    if not args.skip_lm_eval:
        try:
            import lm_eval  # noqa: F401
        except ImportError as exc:
            raise SystemExit(
                "lm_eval not installed. On a networked machine:\n"
                "  pip install lm_eval[api]\n"
                "Then copy the env/wheels to the NPU host, or:\n"
                "  pip install --no-index --find-links=offline_wheels/... lm_eval\n"
                "For a save-only check, re-run with --skip-lm-eval.\n"
                f"Original error: {exc}"
            ) from exc

        tasks = resolve_tasks([t for t in args.tasks.split(",") if t.strip()])
        limit = args.limit if args.limit > 0 else None
        out = args.results_dir / "lm_eval"
        out.mkdir(parents=True, exist_ok=True)
        try:
            report["lm_eval"] = run_lm_eval(
                ckpt, tasks, device, args.dtype, args.batch_size, out, limit
            )
        except RuntimeError as exc:
            # Retry without livecodebench / math500 if task names differ by version.
            print(f"WARN: lm_eval failed ({exc}); trying wikitext,mmlu only")
            report["lm_eval_fallback"] = run_lm_eval(
                ckpt,
                "wikitext,mmlu",
                device,
                args.dtype,
                args.batch_size,
                out / "fallback",
                limit,
            )

    report_path = args.results_dir / "eval_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"EVAL_OK wrote {report_path}")
    print(
        "NOTE: toy (~331M, openwebtext) scores are for relative Muon validation, "
        "not Moonlight-16B paper baselines."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
