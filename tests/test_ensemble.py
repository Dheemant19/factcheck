"""End-to-end test of the T31 ensemble job on synthetic predictions (no real runs needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import factcheck.ensemble as ens
import factcheck.evaluate as evaluate
import factcheck.results as results
from factcheck.data import get_dataset

N = {"validation": 400, "test": 400}
# run name -> (model_name, accuracy): three DeBERTa variants are the strongest
MODELS = {
    "deberta_v3_base_covid_tuned": ("microsoft/deberta-v3-base", 0.90),
    "deberta_v3_base_covid_tuned_cw": ("microsoft/deberta-v3-base", 0.895),
    "deberta_v3_base_covid": ("microsoft/deberta-v3-base", 0.89),
    "roberta_covid": ("roberta-base", 0.885),  # a new backbone within the 0.01 margin
    "bert_covid": ("bert-base-uncased", 0.80),
}


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    splits = {
        s: pd.DataFrame({"id": [f"{s}-{i}" for i in range(n)], "label": rng.integers(0, 2, n),
                         "in_llm_subset": False})
        for s, n in N.items()
    }  # fmt: skip
    splits["train"] = splits["validation"].iloc[:0]
    pred_dir = tmp_path / "predictions"
    rows = []
    for run, (model_name, acc) in MODELS.items():
        for seed in (13, 42, 2026):
            for split, df in splits.items():
                if split == "train":
                    continue
                y = df["label"].to_numpy()
                pred = np.where(rng.random(len(y)) < acc, y, 1 - y)
                p_fake = np.clip(np.where(pred == 1, 0.8, 0.2) + rng.normal(0, 0.05, len(y)), 0, 1)
                out = pd.DataFrame({"id": df["id"], "label": y, "pred": pred,
                                    "prob_real": 1 - p_fake, "prob_fake": p_fake})  # fmt: skip
                path = pred_dir / run / f"seed{seed}" / f"{split}.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                out.sample(frac=1, random_state=seed).to_csv(path, index=False)  # any row order
                rows.append({"run_name": run, "seed": seed, "dataset": "covid", "variant": "raw",
                             "family": "encoder", "model_name": model_name, "method": "full",
                             "split": split, "subset": np.nan})  # fmt: skip
    monkeypatch.setattr(ens, "PREDICTIONS_DIR", pred_dir)
    monkeypatch.setattr(evaluate, "PREDICTIONS_DIR", pred_dir)
    monkeypatch.setattr(results, "RUNS_CSV", tmp_path / "runs.csv")
    monkeypatch.setattr(ens, "load_splits", lambda name, variant: splits)
    return pd.DataFrame(rows), tmp_path


def test_members_are_picked_on_validation_with_a_diverse_backbone(workspace):
    runs, _ = workspace
    models, context = ens.load_models(runs, "covid")
    assert set(models) == set(MODELS)
    members = ens.select_members(models, context["labels"], k=2)
    names = [m.run_name for m in members]
    assert "roberta_covid" in names  # within the margin, so it replaces a third DeBERTa
    assert "bert_covid" not in names
    assert len({m.backbone for m in members}) >= 2


def test_ensemble_is_logged_and_compared(workspace):
    runs, tmp_path = workspace
    candidates, tests = ens.run_dataset("covid", runs, temperature=False)
    logged = pd.read_csv(tmp_path / "runs.csv")
    assert set(logged["run_name"]) == {"ensemble_top3_covid"}
    assert set(logged["split"]) == {"validation", "test"}
    assert (tmp_path / "predictions" / "ensemble_top3_covid" / "seed0" / "test.csv").exists()
    assert sum(c["member_rank"] is not None for c in candidates) == 3
    questions = {t["comparison"] for t in tests}
    assert "ensemble vs best single model" in questions
    assert "Optuna-tuned vs default DeBERTa-v3-base" in questions
    assert "+ class weights (tuned DeBERTa-v3-base)" in questions
    for t in tests:
        assert t["ci_low"] <= t["diff"] <= t["ci_high"]


def test_runs_without_predictions_are_skipped(workspace, capsys):
    runs, _ = workspace
    extra = runs[runs["run_name"] == "bert_covid"].assign(run_name="ghost_covid")
    models, _ = ens.load_models(pd.concat([runs, extra]), "covid")
    assert "ghost_covid" not in models
    assert "skip ghost_covid" in capsys.readouterr().out
    assert get_dataset("covid").labels == ("real", "fake")


def test_significance_only_leaves_runs_csv_alone(workspace):
    runs, tmp_path = workspace
    candidates, tests = ens.run_dataset("covid", runs, temperature=False, with_ensemble=False)
    assert not (tmp_path / "runs.csv").exists()
    assert all(c["member_rank"] is None for c in candidates)
    questions = {t["comparison"] for t in tests}
    assert "ensemble vs best single model" not in questions
    assert "Optuna-tuned vs default DeBERTa-v3-base" in questions
