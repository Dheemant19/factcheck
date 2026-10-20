#!/usr/bin/env bash
# One-time environment setup on a Linux GPU server, using python -m venv and pip.
# Run from the repository root after `git clone`:   bash jobs/setup_cluster.sh
# No admin rights needed. (Python 3.11+; set PYTHON=python3.12 to pick one).
# Picks the newest PyTorch CUDA wheel the server's driver supports.
set -euo pipefail

PYTHON=${PYTHON:-python3}

driver_cuda=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]*\.[0-9]*\).*/\1/p' | head -1)
echo "Driver supports CUDA ${driver_cuda}"
pick() { awk -v v="$driver_cuda" -v t="$1" 'BEGIN { exit !(v + 0 >= t + 0) }'; }
if   pick 13.2; then wheel=cu132
elif pick 13.0; then wheel=cu130
elif pick 12.8; then wheel=cu128
elif pick 12.6; then wheel=cu126
elif pick 12.4; then wheel=cu124
elif pick 12.1; then wheel=cu121
else wheel=cu118; fi
echo "Using PyTorch wheel index: ${wheel}"

[ -d .venv ] || "$PYTHON" -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url "https://download.pytorch.org/whl/${wheel}"
pip install -r requirements.txt

# Keep models and datasets on the server's disk, not in a home quota you might hit.
grep -q HF_HOME ~/.bashrc || echo 'export HF_HOME=~/hf_cache' >> ~/.bashrc
python scripts/check_env.py
