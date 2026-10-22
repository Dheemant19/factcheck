#!/usr/bin/env bash
# Job list for GPU 3. Start inside a screen session so it survives logout:
#   screen -S gpu3
#   CUDA_VISIBLE_DEVICES=3 bash jobs/gpu3.sh      # Ctrl+A then Ctrl+D to detach; screen -r gpu3
# Each line is one run; results append to results/runs.csv. Check nvidia-smi first.
set -uo pipefail   # no -e: one failed run should not stop the rest of the queue
cd "$(dirname "$0")/.."
mkdir -p logs

run() {  # run <config> [extra args...]
  local name; name=$(basename "$1" .yaml)
  echo "[$(date '+%F %T')] start $name"
  python -m factcheck.train --config "$@" > "logs/${name}_gpu3.log" 2>&1     && echo "[$(date '+%F %T')] done  $name" || echo "[$(date '+%F %T')] FAILED $name (see logs/)"
}

# Example (E1, 3 seeds). Replace with this GPU's share of the queue:
# run configs/encoder/bert_liar.yaml --seeds 13 42 2026
