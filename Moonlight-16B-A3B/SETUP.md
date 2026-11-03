# Environment setup

Run all commands from this directory (`Moonlight-16B-A3B`).

## Ascend NPU (air-gapped) — recommended

`torch` / `torch_npu` are assumed already installed on the machine.  
Only the HuggingFace training stack is shipped as offline wheels.

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
cd /path/to/Moonlight-16B-A3B

# After copying offline_wheels/ here (see below):
bash install_offline.sh
python scripts/smoke_test.py
# optional full load:
# python scripts/smoke_test.py --load-model
```

### Getting `offline_wheels/` onto the machine

These wheels are too large for GitHub. On a networked PC they live under
`Moonlight-16B-A3B/offline_wheels/linux_<arch>/`. Pack and copy:

```bash
# on networked machine (example)
tar -czf moonlight-offline-wheels.tar.gz -C Moonlight-16B-A3B offline_wheels install_offline.sh requirements.txt

# on NPU host, inside Moonlight-16B-A3B:
tar -xzf moonlight-offline-wheels.tar.gz
bash install_offline.sh
```

`install_offline.sh` auto-selects `linux_x86_64` or `linux_aarch64` via `uname -m`.

## Local (CPU, no NPU)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -U pip setuptools wheel
pip install -r requirements-cpu.txt
python scripts/smoke_test.py
```
