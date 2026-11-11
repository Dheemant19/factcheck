"""Tests for the metrics, the runs.csv log and the specs check (T04). No data needed."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import factcheck.evaluate as evaluate
import factcheck.results as results
from factcheck.data import get_dataset
from factcheck.evaluate import (
    check_specs,
    compute_metrics,
    expected_calibration_error,
    load_specs,
    log_run,
    parse_target,
)


@pytest.fixture
def tmp_results(tmp_path, monkeypatch):
    """Point runs.csv and the predictions folder at a temporary directory."""
    monkeypatch.setattr(results, "RUNS_CSV", tmp_path / "runs.csv")
    monkeypatch.setattr(evaluate, "PREDICTIONS_DIR", tmp_path / "predictions")
    return tmp_path


def onehot(pred, num_labels, confidence=0.9):
    probs = np.full((len(pred), num_labels), (1 - confidence) / (num_labels - 1))
    probs[np.arange(len(pred)), pred] = confidence
    return probs


def test_perfect_binary_predictions():
    covid = get_dataset("covid")  # labels ("real", "fake")
    labels = np.array([0, 1, 1, 0])
    m = compute_metrics(labels, onehot(labels, 2), covid)
    for key in ("macro_f1", "accuracy", "weighted_f1", "precision_fake", "recall_fake"):
        assert m[key] == 1.0
    assert m["roc_auc_fake"] == m["pr_auc_fake"] == 1.0
    assert m["ece"] == pytest.approx(0.1)  # 90% confident, 100% right
    assert json.loads(m["confusion_matrix"]) == [[2, 0], [0, 2]]
    assert m["n"] == m["n_binary"] == 4


def test_ece_by_hand():
    # four rows at 80% confidence, three right: |0.75 - 0.8| = 0.05
    probs = np.array([[0.8, 0.2]] * 4)
    assert expected_calibration_error([0, 0, 0, 1], probs) == pytest.approx(0.05)


def test_fake_metrics_use_the_binary_mapping():
    liar = get_dataset("liar")  # fake: pants-fire, false, barely-true (ids 0-2)
    labels = np.array([0, 2, 3, 5])
    pred = np.array([1, 4, 3, 5])  # "false" for pants-fire still flags it as fake
    m = compute_metrics(labels, onehot(pred, 6), liar)
    assert m["accuracy"] == 0.5
    assert m["recall_fake"] == 0.5  # caught pants-fire, missed barely-true
    assert m["precision_fake"] == 1.0


def test_rows_outside_the_binary_mapping_are_left_out():
    pubhealth = get_dataset("pubhealth")  # false, mixture, true, unproven; fake = false only
    labels = np.array([0, 1, 2, 3])
    m = compute_metrics(labels, onehot(labels, 4), pubhealth)
    assert m["n"] == 4 and m["n_binary"] == 2


def test_wrong_probability_shape_is_an_error():
    with pytest.raises(ValueError):
        compute_metrics([0, 1], np.ones((2, 3)) / 3, get_dataset("covid"))


def test_parse_target():
    assert parse_target(">= 0.95") == (">=", 0.95)
    assert parse_target("<= 50 ms") == ("<=", 50.0)
    with pytest.raises(ValueError):
        parse_target("one config, one command")


def test_specs_file_lists_s1_to_s10():
    assert [s["id"] for s in load_specs()] == [f"S{i}" for i in range(1, 11)]


def cfg(name, dataset, seed=42, family="lstm"):
    return {
        "run": {"name": name, "seed": seed},
        "data": {"dataset": dataset, "variant": "raw"},
        "model": {"family": family, "name": "bilstm-glove300"},
    }


def split_df(labels, llm_subset=None):
    return pd.DataFrame(
        {
            "id": [f"r{i}" for i in range(len(labels))],
            "label": labels,
            "in_llm_subset": llm_subset if llm_subset is not None else [False] * len(labels),
        }
    )


def test_log_run_writes_rows_and_predictions(tmp_results):
    covid = get_dataset("covid")
    df = split_df([0, 1, 1, 0])
    log_run(cfg("lstm_covid", "covid"), covid, {"test": (df, onehot([0, 1, 0, 0], 2))})
    runs = results.load_runs()
    assert list(runs.columns[: len(results.COLUMNS)]) == results.COLUMNS
    assert runs.loc[0, "accuracy"] == 0.75
    pred = pd.read_csv(tmp_results / "predictions" / "lstm_covid" / "seed42" / "test.csv")
    assert list(pred.columns) == ["id", "label", "pred", "prob_real", "prob_fake"]


def test_logging_again_replaces_the_old_row(tmp_results):
    covid = get_dataset("covid")
    df = split_df([0, 1])
    log_run(cfg("lstm_covid", "covid"), covid, {"test": (df, onehot([1, 1], 2))})
    log_run(cfg("lstm_covid", "covid", seed=13), covid, {"test": (df, onehot([1, 1], 2))})
    log_run(cfg("lstm_covid", "covid"), covid, {"test": (df, onehot([0, 1], 2))})
    runs = results.load_runs()
    assert len(runs) == 2
    assert runs.set_index("seed").loc[42, "accuracy"] == 1.0


def test_a_crash_while_writing_leaves_runs_csv_whole(tmp_results, monkeypatch):
    covid = get_dataset("covid")
    df = split_df([0, 1])
    log_run(cfg("lstm_covid", "covid"), covid, {"test": (df, onehot([1, 1], 2))})
    before = (tmp_results / "runs.csv").read_bytes()

    to_csv = pd.DataFrame.to_csv

    def half_written(self, path, **kwargs):  # noqa: ANN001  (only the runs.csv write dies)
        if "runs.csv" not in str(path):
            return to_csv(self, path, **kwargs)
        with open(path, "w", encoding="utf-8") as f:
            f.write("timestamp,run_na")
        raise OSError("killed mid-write")

    monkeypatch.setattr(pd.DataFrame, "to_csv", half_written)
    with pytest.raises(OSError, match="mid-write"):
        log_run(cfg("lstm_covid", "covid", seed=13), covid, {"test": (df, onehot([1, 1], 2))})
    assert (tmp_results / "runs.csv").read_bytes() == before
    assert not list(tmp_results.glob("*.tmp")) and not list(tmp_results.glob(".*.tmp"))


def test_isot_also_logs_the_llm_subset(tmp_results):
    isot = get_dataset("isot")
    df = split_df([0, 1, 1, 0], llm_subset=[True, True, False, False])
    rows = log_run(cfg("lstm_isot", "isot"), isot, {"test": (df, onehot([0, 1, 0, 1], 2))})
    assert [r["split"] for r in rows] == ["test", "test_llm_subset"]
    assert rows[1]["n"] == 2 and rows[1]["accuracy"] == 1.0


def runs_table(rows):
    base = {"split": "test", "subset": np.nan, "variant": "raw", "family": "encoder"}
    return pd.DataFrame([base | r for r in rows]).reindex(columns=results.COLUMNS)


def test_check_specs():
    runs = runs_table(
        [
            # S2: averaged over seeds, the best config passes (0.96) even though one seed is 0.94
            {"run_name": "a", "seed": 13, "dataset": "covid", "weighted_f1": 0.94},
            {"run_name": "a", "seed": 42, "dataset": "covid", "weighted_f1": 0.98},
            {"run_name": "b", "seed": 42, "dataset": "covid", "weighted_f1": 0.90},
            # S3 fails; a debug subset run with a high score is ignored
            {"run_name": "c", "seed": 42, "dataset": "liar", "accuracy": 0.26},
            {"run_name": "d", "seed": 42, "dataset": "liar", "accuracy": 0.99, "subset": 300},
            # S6 best latency; S9 judged by the slowest run
            {"run_name": "e", "seed": 42, "dataset": "covid", "latency_ms": 12.0},
            {"run_name": "f", "seed": 42, "dataset": "liar", "train_seconds": 7 * 3600},
            {"run_name": "g", "seed": 42, "dataset": "liar", "train_seconds": 60},
        ]
    )
    specs = check_specs(runs).set_index("id")
    assert list(specs.columns) == ["spec", "target", "value", "status", "source"]
    assert specs.loc["S1", "status"] == "FAIL"  # only one family
    assert specs.loc["S2", ["value", "status", "source"]].tolist() == [
        "0.960",
        "PASS",
        "a (2 seeds)",
    ]
    assert specs.loc["S3", "status"] == "FAIL"
    assert specs.loc["S4", "status"] == "no runs yet"
    assert specs.loc["S6", ["value", "status"]].tolist() == ["12.0 ms", "PASS"]
    assert specs.loc["S7", "status"] == "check by hand"
    assert specs.loc["S9", ["value", "status", "source"]].tolist() == ["7.00 h", "FAIL", "worst: f"]


def test_check_specs_with_no_runs():
    specs = check_specs(results.load_runs().iloc[0:0])
    assert set(specs["status"]) == {"no runs yet", "check by hand"}
