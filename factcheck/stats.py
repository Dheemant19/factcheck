"""Ensembling and significance tests on saved predictions (owner M3, T31; plan Section 11).

All functions take plain NumPy arrays: ``y`` true class ids, ``pred`` predicted class ids,
``probs`` (rows, classes) probabilities. They know nothing about runs or files; see
``factcheck.ensemble`` for that.

    soft_vote([probs_a, probs_b, probs_c])        # average the probability matrices
    paired_bootstrap(y, pred_a, pred_b)            # mean and 95% CI of macro-F1(A) - macro-F1(B)
    mcnemar_p(y, pred_a, pred_b)                   # do A and B make different errors?
    holm([0.01, 0.04, 0.03])                       # p-values adjusted for several comparisons
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import binomtest, chi2

N_BOOT = 10_000
EXACT_BELOW = 25  # McNemar: exact binomial test when fewer discordant pairs than this


def soft_vote(prob_list: Sequence[np.ndarray]) -> np.ndarray:
    """Average of several probability matrices of the same shape (rows still sum to 1)."""
    shapes = {np.shape(p) for p in prob_list}
    if len(shapes) != 1:
        raise ValueError(f"probability matrices differ in shape: {sorted(shapes)}")
    return np.mean(np.stack(prob_list), axis=0)


def temperature_scale(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Soften (T > 1) or sharpen (T < 1) probabilities: softmax(log p / T)."""
    logits = np.log(np.clip(probs, 1e-12, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(y: np.ndarray, probs: np.ndarray) -> float:
    """The temperature that minimises the log loss on a validation split (Guo et al., 2017)."""

    def nll(log_t: float) -> float:
        scaled = temperature_scale(probs, float(np.exp(log_t)))
        return -float(np.mean(np.log(np.clip(scaled[np.arange(len(y)), y], 1e-12, None))))

    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


def macro_f1(y: np.ndarray, pred: np.ndarray, n_classes: int) -> float:
    """Macro-F1 from a confusion count (same value as sklearn's average="macro" with every
    class in ``range(n_classes)``; a class never predicted nor present scores 0)."""
    cm = np.bincount(y * n_classes + pred, minlength=n_classes**2).reshape(n_classes, n_classes)
    tp = np.diag(cm).astype(float)
    denom = cm.sum(axis=0) + cm.sum(axis=1)  # predicted + actual = 2tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros(n_classes), where=denom > 0)
    return float(f1.mean())


def paired_bootstrap(
    y: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    n_classes: int | None = None,
    n_boot: int = N_BOOT,
    seed: int = 0,
) -> dict:
    """Macro-F1(A) - macro-F1(B) on the test set and on ``n_boot`` resampled test sets.

    Both models are scored on the SAME resampled rows each time, so the spread reflects only
    which examples happen to be in the test set. Returns the observed difference, the 2.5 and
    97.5 percentiles of the resampled differences, and a two-sided p-value (twice the share
    of resamples on the far side of zero).
    """
    y, pred_a, pred_b = (np.asarray(a, dtype=np.int64) for a in (y, pred_a, pred_b))
    k = n_classes or int(max(y.max(), pred_a.max(), pred_b.max())) + 1
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        diffs[b] = macro_f1(y[i], pred_a[i], k) - macro_f1(y[i], pred_b[i], k)
    observed = macro_f1(y, pred_a, k) - macro_f1(y, pred_b, k)
    tail = min((diffs <= 0).mean(), (diffs >= 0).mean())
    low, high = np.percentile(diffs, [2.5, 97.5])
    return {
        "diff": observed,
        "ci_low": float(low),
        "ci_high": float(high),
        "p_bootstrap": float(min(1.0, 2 * tail)),
    }


def mcnemar_p(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> tuple[float, int]:
    """McNemar test on the rows where exactly one model is right; returns (p, discordant).

    Exact binomial test when the discordant count is small, otherwise the chi-squared test with
    continuity correction.
    """
    a_ok, b_ok = np.asarray(pred_a) == y, np.asarray(pred_b) == y
    only_a, only_b = int(np.sum(a_ok & ~b_ok)), int(np.sum(~a_ok & b_ok))
    n = only_a + only_b
    if n == 0:
        return 1.0, 0
    if n < EXACT_BELOW:
        return float(binomtest(only_a, n, 0.5).pvalue), n
    stat = (abs(only_a - only_b) - 1) ** 2 / n
    return float(chi2.sf(stat, df=1)), n


def holm(p_values: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (same order as the input), capped at 1."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted
