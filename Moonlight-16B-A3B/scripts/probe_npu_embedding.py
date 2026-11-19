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


def main() -> int:
    print(f"torch={torch.__version__}")
    print(f"torch_npu={getattr(torch_npu, '__version__', None)}")
    print(f"ASCEND_HOME_PATH={os.environ.get('ASCEND_HOME_PATH', '')!r}")
    print(f"ASCEND_OPP_PATH={os.environ.get('ASCEND_OPP_PATH', '')!r}")
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
        "Likely missing/mismatched Ascend-cann-kernels for your chip,\n"
        "or CANN env not fully sourced. Workarounds:\n"
        "  MAX_STEPS=20 EMBED_ON_CPU=1 bash scripts/run_official_toy_train_npu.sh\n"
        "  MAX_STEPS=20 DEVICE=cpu bash scripts/run_official_toy_train_npu.sh"
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
