"""Tests for the LDA topic analysis (T56) on a small synthetic corpus."""

from __future__ import annotations

import numpy as np

from factcheck.topics import fit_topics, one_sided_share, top_words, topic_probe, topic_table

POLITICS = ["senate", "election", "vote", "campaign", "congress", "ballot", "candidate"]
HEALTH = ["vaccine", "virus", "hospital", "doctor", "patient", "disease", "clinic"]


def corpus(n: int, seed: int) -> tuple[list[str], np.ndarray]:
    """Fake texts are about politics, real ones about health: topic gives the label away."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, n)
    texts = [" ".join(rng.choice(POLITICS if y else HEALTH, 12)) for y in labels]
    return texts, labels


def test_topics_that_follow_the_label_are_found():
    texts, y = corpus(300, seed=0)
    vec, lda = fit_topics(texts, n_topics=2)
    theta = lda.transform(vec.transform(texts))
    words = top_words(vec, lda, n=5)
    table = topic_table(theta, y, words)
    assert list(table.columns) == ["topic", "top_words", "share_real", "share_fake",
                                   "fake_minus_real"]  # fmt: skip
    assert set(table["top_words"].iloc[0].split()) <= set(POLITICS)  # most fake-leaning first
    assert table["fake_minus_real"].iloc[0] > 0.8
    val_texts, y_val = corpus(100, seed=1)
    probe = topic_probe(theta, y, lda.transform(vec.transform(val_texts)), y_val)
    assert probe["probe_macro_f1"] > 0.95 > probe["majority_macro_f1"]
    assert one_sided_share(theta, y) == 1.0


def test_one_sided_share_is_zero_when_topics_are_mixed():
    theta = np.array([[0.9, 0.1]] * 4 + [[0.1, 0.9]] * 4)
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    assert one_sided_share(theta, labels) == 0.0
