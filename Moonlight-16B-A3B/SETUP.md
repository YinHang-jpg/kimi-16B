# Environment setup

Run all commands from this directory (`Moonlight-16B-A3B`).

## Ascend NPU via GitHub only (recommended)

`torch` / `torch_npu` are assumed already installed.  
All other deps are vendored under `offline_wheels/` in this repo.

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
cd /path/to/Moonlight-16B-A3B
git pull
bash install_offline.sh
python scripts/smoke_test.py
```

`install_offline.sh` picks `linux_x86_64` or `linux_aarch64` from `uname -m` and installs with `--no-index` (no PyPI).

## Local (CPU, no NPU)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -U pip setuptools wheel
pip install -r requirements-cpu.txt
python scripts/smoke_test.py
```
