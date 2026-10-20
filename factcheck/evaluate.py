"""The ONE metrics function, also checks the specs S1-S10 (owner M4, T04).

Every run appends a row to ``results/runs.csv``. The app's Results tab calls
``check_specs(runs)``, which returns one row per spec with columns
``id, spec, target, value, status`` (status: "PASS", "FAIL", "no runs yet" or "check by hand").
"""

from __future__ import annotations

import pandas as pd


def check_specs(runs: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError("T04 (M4): metrics and specs module")
