#!/usr/bin/env bash
# Job list for GPU 2. Start inside a screen session so it survives logout:
#   screen -S gpu2
#   CUDA_VISIBLE_DEVICES=2 bash jobs/gpu2.sh      # Ctrl+A then Ctrl+D to detach; screen -r gpu2
# Each line is one run; results append to results/runs.csv. Check nvidia-smi first.
set -uo pipefail   # no -e: one failed run should not stop the rest of the queue
cd "$(dirname "$0")/.."
mkdir -p logs

run() {  # run <config> [extra args...]
  local name; name=$(basename "$1" .yaml)
  echo "[$(date '+%F %T')] start $name"
  python -m factcheck.train --config "$@" > "logs/${name}_gpu2.log" 2>&1     && echo "[$(date '+%F %T')] done  $name" || echo "[$(date '+%F %T')] FAILED $name (see logs/)"
}

# Example (E1, 3 seeds). Replace with this GPU's share of the queue:
# run configs/encoder/bert_base_liar.yaml --seeds 13 42 2026
