# Environment setup

## Local (CPU, no NPU) — this machine

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
python3.12 -m venv ~/venvs/moonlight
source ~/venvs/moonlight/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements-npu.txt
python scripts/smoke_test.py
# optional:
# python scripts/smoke_test.py --load-model
```

If `torch-npu==2.10.0` fails on CANN 9.0.1, try `torch-npu==2.10.0.post2`.
