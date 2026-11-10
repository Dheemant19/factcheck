"""Tests for the extra-effort (E5) table (T32)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import factcheck.extra_effort as e5


def runs(**scores: list[float]) -> pd.DataFrame:
    """runs.csv test rows: run name -> test macro-F1 per seed."""
    rows = []
    for name, values in scores.items():
        dataset = next(d for d in e5.DATASETS if f"_{d}" in f"_{name}")
        rows += [{"dataset": dataset, "run_name": name, "seed": s, "split": "test",
                  "subset": np.nan, "macro_f1": v} for s, v in zip((13, 42, 2026), values, strict=True)]  # fmt: skip
    return pd.DataFrame(rows)


def test_gains_are_measured_against_the_run_they_change(tmp_path, monkeypatch):
    monkeypatch.setattr(e5, "SIGNIFICANCE_CSV", tmp_path / "none.csv")
    table = e5.build_table(runs(
        deberta_v3_base_liar=[0.27, 0.28, 0.29],
        deberta_v3_base_liar_tuned=[0.28, 0.28, 0.28],
        deberta_v3_base_liar_tuned_meta=[0.30, 0.30, 0.30],
    )).set_index(["dataset", "step"])  # fmt: skip
    tuning = table.loc[("liar", "Optuna tuning")]
    assert tuning["gain"] == pytest.approx(0.0) and tuning["reference"] == "deberta_v3_base_liar"
    meta = table.loc[("liar", "+ speaker metadata")]
    assert meta["gain"] == pytest.approx(0.02) and meta["reference"] == "deberta_v3_base_liar_tuned"
    assert tuning["gain_se"] == pytest.approx(np.sqrt(0.01**2 / 3), abs=1e-4)
    assert table.loc[("liar", "+ evidence sentences"), "status"] == "not applicable"
    assert table.loc[("liar", "+ class weights"), "status"] == "not run yet"
    assert table.loc[("covid", "Optuna tuning"), "status"] == "not run yet"


def test_significance_is_attached_when_t31_has_run(tmp_path, monkeypatch):
    sig = tmp_path / "significance.csv"
    pd.DataFrame([{"model_a": "deberta_v3_base_liar_tuned_meta",
                   "model_b": "deberta_v3_base_liar_tuned", "p_bootstrap_holm": 0.012}]).to_csv(
        sig, index=False)  # fmt: skip
    monkeypatch.setattr(e5, "SIGNIFICANCE_CSV", sig)
    table = e5.build_table(runs(
        deberta_v3_base_liar_tuned=[0.28, 0.28, 0.28],
        deberta_v3_base_liar_tuned_meta=[0.30, 0.30, 0.30],
    )).set_index(["dataset", "step"])  # fmt: skip
    assert table.loc[("liar", "+ speaker metadata"), "p_holm"] == pytest.approx(0.012)
