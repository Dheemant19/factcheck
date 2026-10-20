# Tested environments (T02)

Every machine that trains or runs the app gets one row here after `python scripts/check_env.py`
and `python scripts/smoke_test.py` pass, so we know which setup produced which numbers.
PyTorch is not in `requirements.txt`: each machine installs the build that matches its own
driver (the CUDA version `nvidia-smi` reports).

| Machine | Owner | GPU | Driver (`nvidia-smi`) | Driver CUDA | PyTorch build | Smoke test (val acc, train time) | Date |
|---|---|---|---|---|---|---|---|
| Laptop, Windows 11 | M1 | RTX 5060 Laptop, 8 GB (sm_120) | 610.88 | 13.3 | 2.14.1+cu132 | 0.938, 4.5 s, 1.7 GB peak | 2026-10-20 |
| GPU server, Ubuntu 22.04 | M1 | RTX 3090, 24 GB (sm_86) | 565.57.01 | 12.7 | 2.14.1+cu126 | 0.936, 4.2 s, 1.7 GB peak | 2026-10-20 |

## Adding your machine

1. Run `nvidia-smi` and read the CUDA version (newer drivers label it "CUDA UMD Version").
2. Install the newest PyTorch build at or below it from
   https://pytorch.org/get-started/locally/ (RTX 50-series cards need CUDA 12.8+). For example:

   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu132   # driver CUDA 13.2 or newer
   pip install torch --index-url https://download.pytorch.org/whl/cu126   # driver CUDA 12.6 to 12.9
   ```

3. `pip install -r requirements.txt`, then `python scripts/check_env.py` and
   `python scripts/smoke_test.py` (README "Setup").
4. Add a row to the table above.

## Tips for shared GPU servers

* `bash jobs/setup_cluster.sh` does steps 1 to 3 on Linux. It needs Python 3.11+; if the
  system Python is older, make one with conda (`conda create -n ee6405 python=3.11`) and run
  `PYTHON=<that env>/bin/python bash jobs/setup_cluster.sh`.
* No `git` and no sudo? `conda install git` into that env and link it onto your PATH:
  `ln -s <env>/bin/git ~/.local/bin/git` (Ubuntu adds `~/.local/bin` at login).
* The repo is private. To `git pull` on a server, add a read-only deploy key for that server
  (repo Settings > Deploy keys) instead of copying your personal GitHub key there.
* With a conda Python, `setup_cluster.sh` points the venv at conda's libstdc++; without it
  optuna fails to import (`CXXABI_1.3.15 not found`).
* Set `HF_HOME` to a folder on the server's big disk so models do not fill a home quota.
* GPUs are often shared: check `nvidia-smi` first and pick a free one with
  `CUDA_VISIBLE_DEVICES=<id>`.
