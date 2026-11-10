"""Reading and writing results/runs.csv (T04; M1, agreed with M4).

``runs.csv`` is the project's scoreboard: one row per run, seed and split, written by
``factcheck.evaluate.log_run`` at the end of every training or scoring run. The app's Results
tab, the specs check and the results notebook all read it through ``load_runs``.

A row is identified by ``run_name, seed, split``. Logging the same run again (a re-run after
a fix) replaces its old rows instead of adding duplicates, so the file never double counts.
Writes take a file lock, so the four cluster GPUs can log to the same file at once, and go
to a temporary file that then replaces runs.csv in one step, so a crash mid-write leaves the
old file whole and a reader never sees half a table.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
from filelock import FileLock

from factcheck.paths import RUNS_CSV

KEY = ["run_name", "seed", "split"]

# Column order of runs.csv. Metrics a run does not have (e.g. latency for an LLM) stay empty.
COLUMNS = [
    "timestamp",
    "run_name",
    "seed",
    "dataset",
    "variant",
    "family",
    "model_name",
    "method",
    "split",
    "subset",  # rows used when a debug run trains on a data subset; empty for real runs
    "n",
    "macro_f1",
    "accuracy",
    "weighted_f1",
    "precision_fake",
    "recall_fake",
    "roc_auc_fake",
    "pr_auc_fake",
    "ece",
    "n_binary",
    "confusion_matrix",
    "trainable_params",
    "train_seconds",
    "peak_gpu_mem_mb",
    "latency_ms",
    "best_epoch",
    "device",
    "git_commit",
    "config",
]


def load_runs() -> pd.DataFrame:
    """Every logged run (an empty table with the right columns if nothing is logged yet)."""
    if not RUNS_CSV.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(RUNS_CSV)


def append_runs(rows: list[dict]) -> pd.DataFrame:
    """Add rows to runs.csv, replacing older rows with the same run_name, seed and split."""
    new = pd.DataFrame(rows)
    extra = [c for c in new.columns if c not in COLUMNS]
    new = new.reindex(columns=COLUMNS + extra)
    RUNS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(RUNS_CSV) + ".lock"):
        old = load_runs()
        if not old.empty:
            keys = set(new[KEY].astype(str).itertuples(index=False, name=None))
            stale = old[KEY].astype(str).apply(tuple, axis=1).isin(keys)
            old = old[~stale]
        # concat with an empty frame is a no-op that pandas warns about, so skip it
        runs = pd.concat([old, new], ignore_index=True) if not old.empty else new
        write_atomically(runs, RUNS_CSV)
    return runs


def write_atomically(runs: pd.DataFrame, path: Path, tries: int = 20) -> None:
    """Write next to ``path``, then swap it in with ``os.replace`` (atomic on one drive).

    On Windows the swap fails while another process has ``path`` open (the app reading it),
    so it retries for a few seconds before giving up; the temporary file is removed on failure.
    """
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        runs.to_csv(tmp, index=False)
        for attempt in range(tries):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt == tries - 1:
                    raise
                time.sleep(0.25)
    finally:
        tmp.unlink(missing_ok=True)
