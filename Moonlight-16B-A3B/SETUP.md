# Environment setup

Run all commands from this directory (`Moonlight-16B-A3B`).

## Ascend NPU via GitHub only (recommended)

Vendored under `offline_wheels/`:
- `torch==2.10.0+cpu` (split into `.whl.partXX` because GitHub max file is 100MB)
- `torch_npu==2.10.0`
- HuggingFace training stack (`transformers`, `accelerate`, `peft`, ...)

```bash
cd /path/to/Moonlight-16B-A3B
# Prefer this helper on ModelArts (aligns ASCEND_OPP_PATH to a tree that has 910b kernels):
source scripts/fix_cann_env.sh
# (fallback: source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh)

git pull
python3.12 -m venv .venv
source .venv/bin/activate

bash install_offline.sh
python scripts/smoke_test.py
python scripts/probe_npu_embedding.py   # must show OPP_OK + Embedding OK
```

If `probe_npu_embedding.py` reports missing `ascend910b` kernel binaries / error 561103,
install `Ascend-cann-kernels-910b` matching CANN 9.0.1 (ask ModelArts admin). Until then:

```bash
MAX_STEPS=20 EMBED_ON_CPU=1 bash scripts/run_official_toy_train_npu.sh
# or fully on CPU:
MAX_STEPS=20 DEVICE=cpu DTYPE=float32 bash scripts/run_official_toy_train_npu.sh
```


`install_offline.sh` rebuilds split torch wheels, then `pip install --no-index` for the matching arch (`linux_x86_64` / `linux_aarch64`).

## Local (CPU, no NPU)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -U pip setuptools wheel
pip install -r requirements-cpu.txt
python scripts/smoke_test.py
```

## Official toy Muon training (what Moonshot actually open-sourced)

Read `docs/REPRODUCIBILITY.md` first. Full 16B / 5.7T training is **not** open-sourced.

```bash
# toy_prefetch/ is vendored in this repo (Arrow file split into *.partXX).
# NPU host:
git pull
bash scripts/run_official_toy_train_npu.sh
# (script auto-runs scripts/reassemble_toy_prefetch.sh first)
```
