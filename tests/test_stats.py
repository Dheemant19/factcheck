"""Tests for the ensemble and significance statistics (T31)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import f1_score

from factcheck.stats import (
    fit_temperature,
    holm,
    macro_f1,
    mcnemar_p,
    paired_bootstrap,
    soft_vote,
    temperature_scale,
)

rng = np.random.default_rng(0)


@pytest.mark.parametrize("k", [2, 4, 6])
def test_macro_f1_matches_sklearn(k):
    y, pred = rng.integers(0, k, 500), rng.integers(0, k, 500)
    pred[pred == k - 1] = 0  # one class never predicted
    expected = f1_score(y, pred, average="macro", labels=range(k), zero_division=0)
    assert macro_f1(y, pred, k) == pytest.approx(expected)


def test_soft_vote_averages_and_checks_shapes():
    a, b = np.array([[0.9, 0.1], [0.2, 0.8]]), np.array([[0.5, 0.5], [0.4, 0.6]])
    np.testing.assert_allclose(soft_vote([a, b]), [[0.7, 0.3], [0.3, 0.7]])
    with pytest.raises(ValueError, match="shape"):
        soft_vote([a, np.ones((3, 2)) / 2])


def test_temperature_softens_an_overconfident_model():
    y = rng.integers(0, 2, 2000)
    correct = rng.random(2000) < 0.7  # 70% accurate but always 99% sure
    pred = np.where(correct, y, 1 - y)
    probs = np.where(pred[:, None] == np.arange(2), 0.99, 0.01)
    t = fit_temperature(y, probs)
    assert t > 1.5
    scaled = temperature_scale(probs, t)
    np.testing.assert_allclose(scaled.sum(axis=1), 1.0)
    assert scaled.max(axis=1).mean() == pytest.approx(0.7, abs=0.03)  # now about calibrated
    assert (scaled.argmax(axis=1) == pred).all()  # scaling never changes the prediction


def test_bootstrap_of_a_model_against_itself_is_zero():
    y, pred = rng.integers(0, 3, 300), rng.integers(0, 3, 300)
    out = paired_bootstrap(y, pred, pred, n_classes=3, n_boot=200)
    assert out == {"diff": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_bootstrap": 1.0}


def test_bootstrap_detects_a_clearly_better_model():
    y = rng.integers(0, 2, 1000)
    good = np.where(rng.random(1000) < 0.9, y, 1 - y)
    bad = np.where(rng.random(1000) < 0.6, y, 1 - y)
    out = paired_bootstrap(y, good, bad, n_boot=1000)
    assert out["diff"] > 0.2 and out["ci_low"] > 0 and out["p_bootstrap"] < 0.01
    assert out == paired_bootstrap(y, good, bad, n_boot=1000)  # seeded: reproducible


def test_mcnemar():
    y = np.zeros(40, dtype=int)
    a = np.zeros(40, dtype=int)  # always right
    b = a.copy()
    b[:10] = 1  # wrong on 10 rows where A is right
    p, n = mcnemar_p(y, a, b)
    assert n == 10 and p == pytest.approx(2 * 0.5**10)  # exact test: 10 vs 0
    assert mcnemar_p(y, a, a) == (1.0, 0)


def test_holm():
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    np.testing.assert_allclose(holm([0.5, 0.9]), [1.0, 1.0])
