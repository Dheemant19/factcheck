# Tested environments (T02)

Every machine that trains or runs the app gets one row here after `python scripts/check_env.py`
and `python scripts/smoke_test.py` pass. PyTorch is not in `requirements.txt`: each machine
installs the build that matches its own driver (the CUDA version `nvidia-smi` reports).

| Machine | Owner | GPU | Driver (`nvidia-smi`) | Driver CUDA | PyTorch build | Smoke test (val acc, train time) | Date |
|---|---|---|---|---|---|---|---|
| Laptop, Windows 11 | M1 | RTX 5060 Laptop, 8 GB (sm_120) | 610.88 | 13.3 | 2.14.1+cu132 | 0.938, 4.5 s, 1.7 GB peak | 2026-10-20 |
| GPU server | M1 + M2 | | | | | | |

Torch install used on the M1 laptop (its driver supports CUDA 13.3, so the CUDA 13.2 build):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu132
```

On a new machine, read the CUDA version from `nvidia-smi` and pick the newest PyTorch build at
or below it from https://pytorch.org/get-started/locally/. RTX 50-series cards need CUDA 12.8+.
