#!/usr/bin/env python3
"""Minimal Ascend Embedding / GatherV2 health check.

Run after: source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
"""

from __future__ import annotations

import os
import sys
import traceback

import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None


def _dir_bytes(path: str) -> int:
    total = 0
    if not os.path.isdir(path):
        return 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def check_opp_kernels() -> None:
    opp = os.environ.get("ASCEND_OPP_PATH", "")
    print(f"ASCEND_OPP_PATH={opp!r}")
    if not opp:
        print("OPP_WARN: ASCEND_OPP_PATH empty")
        return
    kroot = os.path.join(opp, "built-in", "op_impl", "ai_core", "tbe", "kernel")
    cfg = os.path.join(kroot, "config")
    print(f"kernel_root_exists={os.path.isdir(kroot)}")
    print(f"kernel_config_entries={os.listdir(cfg) if os.path.isdir(cfg) else []}")
    for soc in ("ascend910b", "ascend910_93", "ascend910"):
        p = os.path.join(kroot, soc)
        b = _dir_bytes(p)
        print(f"kernel_tree {soc}: exists={os.path.isdir(p)} bytes={b} (~{b/1e9:.2f}GB)")
    # Healthy 910b kernel binaries are multi-GB; config-only trees cause 561103.
    b910b = _dir_bytes(os.path.join(kroot, "ascend910b"))
    if b910b < 50_000_000:
        print(
            "OPP_FAIL: ascend910b kernel binaries missing/stripped "
            f"({b910b} bytes). Install Ascend-cann-kernels-910b for CANN 9.0.1."
        )
    else:
        print("OPP_OK: ascend910b kernel tree looks populated")


def main() -> int:
    print(f"torch={torch.__version__}")
    print(f"torch_npu={getattr(torch_npu, '__version__', None)}")
    print(f"ASCEND_HOME_PATH={os.environ.get('ASCEND_HOME_PATH', '')!r}")
    print(f"ASCEND_AICPU_PATH={os.environ.get('ASCEND_AICPU_PATH', '')!r}")
    check_opp_kernels()
    print(f"npu_available={hasattr(torch, 'npu') and torch.npu.is_available()}")
    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        print("PROBE_FAIL: NPU not available")
        return 1

    print(f"device_name={torch.npu.get_device_name(0)}")
    ok_any = False
    for dtype in (torch.float16, torch.float32, torch.bfloat16):
        for idx_dtype in (torch.int64, torch.int32):
            try:
                w = torch.randn(256, 64, device="npu:0", dtype=dtype)
                idx = torch.randint(0, 256, (4, 16), device="npu:0", dtype=idx_dtype)
                out = torch.nn.functional.embedding(idx, w)
                torch.npu.synchronize()
                s = float(out.float().sum().cpu())
                print(f"OK  dtype={dtype} idx={idx_dtype} sum={s:.4f}")
                ok_any = True
            except Exception as exc:  # noqa: BLE001
                print(f"FAIL dtype={dtype} idx={idx_dtype}: {exc}")
                traceback.print_exc(limit=2)

    if ok_any:
        print("PROBE_PARTIAL_OR_OK")
        return 0
    print(
        "PROBE_FAIL: Embedding broken on this NPU.\n"
        "Root cause is almost always missing Ascend-cann-kernels-910b binaries\n"
        "(config/ dirs can exist while kernel/*.o / binary tree is empty).\n"
        "Workarounds until kernels are installed:\n"
        "  MAX_STEPS=20 EMBED_ON_CPU=1 bash scripts/run_official_toy_train_npu.sh\n"
        "  MAX_STEPS=20 DEVICE=cpu bash scripts/run_official_toy_train_npu.sh"
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
