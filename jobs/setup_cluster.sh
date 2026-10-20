#!/usr/bin/env bash
# One-time environment setup on an MLDA GPU server (4x RTX 3090), using uv like the laptop.
# Run from the repository root after `git clone`:   bash jobs/setup_cluster.sh
# No conda or admin rights needed: uv installs to ~/.local/bin and downloads Python itself.
# Picks the newest PyTorch CUDA wheel the server's driver supports.
set -euo pipefail

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

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

[ -d .venv ] || uv venv --python 3.11
source .venv/bin/activate
uv pip install torch --index-url "https://download.pytorch.org/whl/${wheel}"
uv pip install -e ".[llm,explain,dev]"

# Keep models and datasets on the server's disk, not in a home quota you might hit.
grep -q HF_HOME ~/.bashrc || echo 'export HF_HOME=~/hf_cache' >> ~/.bashrc
python scripts/check_env.py
