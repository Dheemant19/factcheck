# Tested environments (T02)

Every machine that trains or runs the app gets one row here after `python scripts/check_env.py`
and `python scripts/smoke_test.py` pass. PyTorch is not in `requirements.txt`: each machine
installs the build that matches its own driver (the CUDA version `nvidia-smi` reports).

| Machine | Owner | GPU | Driver (`nvidia-smi`) | Driver CUDA | PyTorch build | Smoke test (val acc, train time) | Date |
|---|---|---|---|---|---|---|---|
| Laptop, Windows 11 | M1 | RTX 5060 Laptop, 8 GB (sm_120) | 610.88 | 13.3 | 2.14.1+cu132 | 0.938, 4.5 s, 1.7 GB peak | 2026-10-20 |
| MLDA gpu21, Ubuntu 22.04 | M1 + M2 | 4x RTX 3090, 24 GB (sm_86) | 565.57.01 | 12.7 | 2.14.1+cu126 | 0.936, 4.2 s, 1.7 GB peak (GPU 1) | 2026-10-20 |

Torch installs used (each picks the newest build at or below the driver's CUDA version):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu132   # M1 laptop, driver CUDA 13.3
pip install torch --index-url https://download.pytorch.org/whl/cu126   # MLDA gpu21, driver CUDA 12.7
```

## MLDA gpu21 notes

* System Python is 3.10, so the venv is built from a conda Python 3.11:
  `conda create -n ee6405 python=3.11 git`, then
  `PYTHON=~/miniforge3/envs/ee6405/bin/python bash jobs/setup_cluster.sh`.
  The repo lives in `~/factcheck` with its `.venv`; the server has no system `git`, so
  activate the conda env (`conda activate ee6405`) to get one.
* `setup_cluster.sh` makes the venv use conda's libstdc++; without it optuna fails to import
  (`CXXABI_1.3.15 not found`).
* Datasets are in `~/factcheck/data/raw/`, Hugging Face downloads in `~/hf_cache` (`HF_HOME`).
* The account and GPUs are shared: check `nvidia-smi` and pick a free GPU with
  `CUDA_VISIBLE_DEVICES` (GPU 0 already had about 3 GB in use on 2026-10-20).

On a new machine, read the CUDA version from `nvidia-smi` and pick the newest PyTorch build at
or below it from https://pytorch.org/get-started/locally/. RTX 50-series cards need CUDA 12.8+.
