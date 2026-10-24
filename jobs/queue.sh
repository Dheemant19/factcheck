#!/usr/bin/env bash
# Shared job queue: every GPU worker takes the next line of one queue file, so all GPUs stay
# busy until the queue is empty, however long each run takes. Start one worker per free GPU
# (check nvidia-smi first), each in its own screen session or with setsid nohup:
#   CUDA_VISIBLE_DEVICES=0 bash jobs/queue.sh jobs/t17_encoders.txt
#   CUDA_VISIBLE_DEVICES=1 bash jobs/queue.sh jobs/t17_encoders.txt   # ...and so on
# Each queue line is one `python -m factcheck.train --config` call: a config and its options,
# e.g. "configs/encoder/bert_liar.yaml --seeds 13 42 2026". Blank lines and # comments are
# skipped. Put the longest runs first. Workers pop lines from logs/<queue>.todo, a copy made
# by the first worker; delete it to start the queue again from the top.
set -uo pipefail   # no -e: one failed run should not stop the rest of the queue
cd "$(dirname "$0")/.."
mkdir -p logs

queue="$1"
todo="logs/$(basename "$queue" .txt).todo"
gpu="${CUDA_VISIBLE_DEVICES:-0}"
exec 9>"$todo.lock"

flock 9
[ -f "$todo" ] || grep -v -E '^\s*(#|$)' "$queue" > "$todo"
flock -u 9

while true; do
  flock 9
  line=$(head -n 1 "$todo")
  sed -i '1d' "$todo"
  flock -u 9
  [ -n "$line" ] || break
  name=$(basename "${line%% *}" .yaml)
  echo "[$(date '+%F %T')] gpu$gpu start $name"
  # Appends: one config's seeds can be separate lines that end up on the same GPU.
  # shellcheck disable=SC2086  # the line holds several arguments
  if python -m factcheck.train --config $line >> "logs/${name}_gpu${gpu}.log" 2>&1; then
    echo "[$(date '+%F %T')] gpu$gpu done  $name"
  else
    echo "[$(date '+%F %T')] gpu$gpu FAILED $name (see logs/${name}_gpu${gpu}.log)"
  fi
done
echo "[$(date '+%F %T')] gpu$gpu queue empty"
