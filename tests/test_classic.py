"""Tests for the traditional baselines (T54). None of them need the datasets or the GloVe file."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from factcheck.data import get_dataset
from factcheck.models import classic

REAL = ["CDC data shows cases rising in two states", "WHO publishes new testing data today",
        "Health officials report case data by state", "State data on tests released by CDC"]  # fmt: skip
FAKE = ["Miracle cure kills corona overnight!!", "Garlic cure stops corona, doctors stunned!",
        "Secret corona cure they hide from you!", "This cure ends corona in one day!!"]  # fmt: skip


def mcfg(method: str = "tfidf_lr", **overrides) -> dict:
    cfg = classic.with_defaults({"model": {"method": method}, "train": {}})["model"]
    return {**cfg, "word_min_df": 1, "char_min_df": 1, **overrides}


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="model.method"):
        classic.with_defaults({"model": {"method": "tfidf_rf"}})


def test_joined_tfidf_rows_stay_unit_length_and_name_both_kinds():
    features = classic.TfidfFeatures(mcfg())
    x = features.fit_transform(REAL + FAKE)
    norms = np.sqrt(np.asarray(x.multiply(x).sum(axis=1))).ravel()
    assert np.allclose(norms, 1.0, atol=1e-5)
    names = features.names()
    assert len(names) == x.shape[1]
    assert "cure" in names and any(n.startswith("'") for n in names)


def test_word_only_features_when_character_ngrams_are_off():
    features = classic.TfidfFeatures(mcfg(char_ngrams=None))
    features.fit_transform(REAL + FAKE)
    assert not any(n.startswith("'") for n in features.names())


def test_grid_covers_regularisation_and_class_weights():
    tcfg = classic.DEFAULTS["train"]
    assert len(classic.candidates("tfidf_nb", tcfg)) == len(tcfg["nb_alpha"])
    lr = classic.candidates("tfidf_lr", tcfg)
    assert len(lr) == len(tcfg["lr_C"]) * 2 and {c["class_weight"] for c in lr} == {
        None,
        "balanced",
    }


def test_top_words_point_the_right_way():
    info = get_dataset("covid")  # labels: real, fake
    features = classic.TfidfFeatures(mcfg(char_ngrams=None))
    x = features.fit_transform(REAL + FAKE)
    y = [info.labels.index("real")] * 4 + [info.labels.index("fake")] * 4
    for clf in (LogisticRegression(C=10).fit(x, y), MultinomialNB(alpha=0.1).fit(x, y)):
        words = classic.top_words(clf, features.names(), info, n=3)
        top = {label: words[words["label"] == label]["feature"].tolist() for label in info.labels}
        assert "cure" in top["fake"] or "corona" in top["fake"]
        assert "data" in top["real"]
        assert not set(top["fake"]) & set(top["real"])


def test_averaged_glove_is_the_mean_of_known_word_vectors():
    features = classic.GloveFeatures(mcfg("glove_svm"))
    features.word2id = {"<pad>": 0, "<unk>": 1, "cure": 2, "corona": 3}
    features.embeddings = np.array([[0, 0], [0, 0], [2, 0], [0, 4]], dtype=np.float32)
    means = features._mean(["cure corona", "cure unknownword", ""])
    assert np.allclose(means, [[1, 2], [1, 0], [0, 0]])


@pytest.mark.parametrize("method", ["tfidf_nb", "tfidf_lr", "tfidf_svm"])
def test_train_end_to_end_on_a_tiny_dataset(method, tmp_path, monkeypatch):
    info = get_dataset("covid")
    fake_id, real_id = info.labels.index("fake"), info.labels.index("real")

    def frame(texts_real, texts_fake):
        return pd.DataFrame({"text": texts_real + texts_fake,
                             "label": [real_id] * len(texts_real) + [fake_id] * len(texts_fake)})  # fmt: skip

    splits = {"train": frame(REAL * 3, FAKE * 3), "validation": frame(REAL, FAKE),
              "test": frame(REAL[:2], FAKE[:2])}  # fmt: skip
    logged = {}
    monkeypatch.setattr(classic, "load_inputs", lambda cfg: splits)
    monkeypatch.setattr(classic, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(classic, "log_run", lambda cfg, info, scored, extra: logged.update(scored))
    cfg = {"run": {"name": f"tiny_{method}", "seed": 42}, "data": {"dataset": "covid"},
           "model": {"family": "classic", "method": method, "word_min_df": 1, "char_min_df": 1},
           "train": {"calibration_folds": 2}}  # fmt: skip
    summary = classic.train_classic(cfg)

    assert summary["best_val_macro_f1"] == 1.0
    _, probs = logged["test"]
    assert probs.shape == (4, 2) and np.allclose(probs.sum(axis=1), 1.0)
    assert (probs.argmax(axis=1) == splits["test"]["label"].to_numpy()).all()
    run_dir = tmp_path / f"tiny_{method}" / "seed42"
    assert (run_dir / "top_words.csv").exists() and (run_dir / "train_summary.json").exists()
    saved = classic.ClassicPredictor.load(run_dir / "model" / "model.joblib")
    assert saved.predict_proba(["Miracle corona cure!!"])[0].argmax() == fake_id
    assert summary["trainable_params"] > 0


def test_svm_grid_scores_the_calibrated_model_it_returns():
    # Imbalanced, noisy three-class data, where balanced weights and calibration disagree.
    rng = np.random.default_rng(0)
    y = rng.choice(3, size=300, p=[0.7, 0.2, 0.1])
    x = rng.normal(size=(300, 5)) + 0.8 * np.eye(3, 5)[y]
    tcfg = {**classic.DEFAULTS["train"], "svm_C": [0.01, 1.0]}
    best, setting, grid = classic.select("glove_svm", tcfg, x[:200], y[:200], x[200:], y[200:])

    assert isinstance(best, classic.CalibratedClassifierCV)
    assert best.estimator.get_params()["C"] == setting["C"]
    f1 = classic.f1_score(y[200:], best.predict_proba(x[200:]).argmax(axis=1), average="macro")
    assert f1 == pytest.approx(max(g["val_macro_f1"] for g in grid))


def test_sparse_join_keeps_csr():
    a = sp.csr_matrix(np.eye(2, dtype=np.float32))
    assert sp.isspmatrix_csr(classic.TfidfFeatures._join([a, a]))
