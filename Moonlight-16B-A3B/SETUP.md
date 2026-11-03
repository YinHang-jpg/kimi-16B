# Environment setup

Run all commands from this directory (`Moonlight-16B-A3B`).

## Local (CPU, no NPU)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -U pip setuptools wheel
pip install -r requirements-cpu.txt
python scripts/smoke_test.py
```

Default smoke test checks imports + tokenizer/config only.  
Full weight load (needs lots of RAM / NPU):

```bash
python scripts/smoke_test.py --load-model
```

## Ascend NPU target (CANN 9.0.1, Python 3.12)

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
cd /path/to/Moonlight-16B-A3B
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements-npu.txt
python scripts/smoke_test.py
# optional:
# python scripts/smoke_test.py --load-model
```

If `torch-npu==2.10.0` fails on CANN 9.0.1, try `torch-npu==2.10.0.post2`.

## Offline install (no PyPI access)

Wheels under `offline_wheels/` can be installed without network:

```bash
pip install --no-index --find-links=./offline_wheels accelerate==1.3.0
```

If other packages also time out, download their Linux/`py3-none-any` wheels on a networked machine into `offline_wheels/`, sync, then install the same way.
