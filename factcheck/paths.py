"""Every filesystem location the project uses, in one place.

No other module hard-codes a path. Each location can be moved with an environment
variable (handy on the cluster, where data should live on the big disk):

    FACTCHECK_ROOT          repository root (default: the folder above this package)
    FACTCHECK_DATA_DIR      raw + processed datasets      (default: <root>/data)
    FACTCHECK_CACHE_DIR     tokenised data, embeddings    (default: <data>/cache)
    FACTCHECK_OUTPUT_DIR    checkpoints and adapters      (default: <root>/outputs)
    FACTCHECK_RESULTS_DIR   runs.csv, predictions, figures (default: <root>/results)

Set them before importing the package (for example in a `.env` file loaded by your shell).
"""

from __future__ import annotations

import os
from pathlib import Path


def _env_path(var: str, default: Path) -> Path:
    value = os.environ.get(var)
    return Path(value).expanduser().resolve() if value else default


PROJECT_ROOT = _env_path("FACTCHECK_ROOT", Path(__file__).resolve().parents[1])
CONFIG_DIR = PROJECT_ROOT / "configs"

DATA_DIR = _env_path("FACTCHECK_DATA_DIR", PROJECT_ROOT / "data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = _env_path("FACTCHECK_CACHE_DIR", DATA_DIR / "cache")

OUTPUT_DIR = _env_path("FACTCHECK_OUTPUT_DIR", PROJECT_ROOT / "outputs")
RESULTS_DIR = _env_path("FACTCHECK_RESULTS_DIR", PROJECT_ROOT / "results")
RUNS_CSV = RESULTS_DIR / "runs.csv"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
FIGURES_DIR = RESULTS_DIR / "figures"
