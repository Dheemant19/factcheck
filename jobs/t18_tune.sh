#!/usr/bin/env bash
# T18 Optuna search (E2): DeBERTa-v3-base, one study per dataset, 30 trials each. Start one
# worker per free GPU (check nvidia-smi first); the workers share each study and move on to the
# next dataset once the current one has 30 trials:
#   CUDA_VISIBLE_DEVICES=0 setsid nohup bash jobs/t18_tune.sh > logs/t18_tune_gpu0.log 2>&1 < /dev/null &
# When every worker has finished, write the trial tables, importances and tuned configs:
#   for d in covid liar pubhealth; do
#     python -m factcheck.tune --config configs/encoder/deberta_v3_base_$d.yaml --export
#   done
# ISOT is left out: every encoder already scores 0.999 or more on it.
set -uo pipefail   # no -e: one failed study should not stop the next
cd "$(dirname "$0")/.."
mkdir -p logs

for dataset in covid liar pubhealth; do
  echo "[$(date '+%F %T')] gpu${CUDA_VISIBLE_DEVICES:-0} start $dataset"
  python -m factcheck.tune --config "configs/encoder/deberta_v3_base_${dataset}.yaml" --trials 30
done
echo "[$(date '+%F %T')] gpu${CUDA_VISIBLE_DEVICES:-0} done"
