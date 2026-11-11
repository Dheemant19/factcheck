#!/usr/bin/env bash
# T31 + T32: top-3 ensembles, significance tests and the extra-effort table, from the saved
# predictions. Run it where results/predictions/ is complete (the cluster), after pulling:
#   bash jobs/t31_ensemble.sh
# No GPU needed: about a minute on a CPU. After the results freeze (T43) it runs the
# significance tests only (--skip-ensemble), so the frozen runs.csv is not changed; drop the
# flag only if the team agrees to add the optional ensemble rows. It writes
# results/significance.csv, results/ensemble_candidates.csv, results/e5_extra_effort.{csv,md}
# and results/figures/e5_extra_effort.png; commit those.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m factcheck.ensemble --skip-ensemble
python -m factcheck.extra_effort
