"""Traditional baselines: TF-IDF with Naive Bayes, logistic regression or a linear SVM, and
averaged GloVe vectors with a linear SVM (owner M1, T54).

Added after the Week 12 consultation (plan Section 8.0): the bottom rungs of the ladder
TF-IDF -> word embeddings -> RNN family -> transformers. They use the course's Week 3, 4.1 and
4.2 methods, train in seconds to minutes on a CPU and read the same splits and inputs as every
other model (``factcheck.inputs.load_inputs``).

    python -m factcheck.train --config configs/classic/tfidf_lr_liar.yaml

``model.method`` picks the model:

    tfidf_nb     TF-IDF word 1-2 grams (+ character 2-5 grams) -> multinomial Naive Bayes
    tfidf_lr     the same features -> logistic regression (its weights are the readable words)
    tfidf_svm    the same features -> linear SVM, probabilities by sigmoid calibration
    glove_svm    the mean of the GloVe 300-d vectors of the words -> linear SVM, calibrated

The regularisation strength (and, for LR and SVM, plain vs balanced class weights) is picked
on validation macro-F1 from a small grid (for the SVMs, the calibrated model's); the model is fitted on the training split only and
the test split is scored once. Fitting is deterministic, so one run (seed 42) is enough, like
the kNN vote.

Each run writes, under ``outputs/<name>/seed<N>/``:

    config.yaml              the full config used (defaults filled in)
    model/model.joblib       features + classifier (load with ``ClassicPredictor``)
    top_words.csv            TF-IDF methods: per label, the 30 word n-grams and the 30 character
                             n-grams that push hardest towards it
    train_summary.json       the validation grid, the chosen setting, time and size

and logs the validation and test scores to ``results/runs.csv`` through
``factcheck.evaluate.log_run``, which also saves the per-row probabilities.
"""

from __future__ import annotations

import copy
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight

from factcheck.data import get_dataset
from factcheck.data.registry import DatasetInfo
from factcheck.evaluate import log_run, measure_latency_ms
from factcheck.inputs import load_inputs, single_text
from factcheck.models.lstm import encode, vocab_and_embeddings
from factcheck.paths import OUTPUT_DIR, RAW_DIR

METHODS = ("tfidf_nb", "tfidf_lr", "tfidf_svm", "glove_svm")
TOP_N = 30

DEFAULTS = {
    "model": {
        # TF-IDF (plan Section 8.0): sublinear tf, smoothed idf, l2-normalised rows.
        "word_ngrams": [1, 2],
        "word_min_df": 2,
        "word_max_features": 200_000,
        "char_ngrams": [2, 5],  # null turns the character features off (long ISOT articles)
        "char_min_df": 3,
        "char_max_features": 200_000,
        # Averaged GloVe: the same frozen vectors and tokenizer as the Bi-LSTM.
        "embedding_path": str(RAW_DIR / "glove" / "glove.6B.300d.txt"),
        "embedding_dim": 300,
        "max_words": 2000,  # words averaged per text (ISOT articles are long)
    },
    "train": {
        # Validation grid. Naive Bayes: Laplace/Lidstone smoothing alpha. LR and SVM: the
        # inverse regularisation strength C, each with and without balanced class weights.
        "nb_alpha": [0.01, 0.03, 0.1, 0.3, 1.0],
        "lr_C": [0.3, 1.0, 3.0, 10.0, 30.0, 100.0],
        "svm_C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
        "class_weights": [None, "balanced"],
        "calibration_folds": 3,
    },
}


def with_defaults(cfg: dict) -> dict:
    """The run config with every setting filled in; checks the method name."""
    cfg = copy.deepcopy(cfg)
    for section, defaults in DEFAULTS.items():
        cfg[section] = {**defaults, **(cfg.get(section) or {})}
    if cfg["model"]["method"] not in METHODS:
        raise ValueError(f"model.method must be one of {METHODS}, not {cfg['model']['method']!r}")
    return cfg


# ---------------------------------------------------------------------------------------------
# Features


class TfidfFeatures:
    """Word n-gram TF-IDF, optionally side by side with character n-gram TF-IDF."""

    def __init__(self, mcfg: dict):
        self.word = TfidfVectorizer(
            ngram_range=tuple(mcfg["word_ngrams"]),
            min_df=mcfg["word_min_df"],
            max_features=mcfg["word_max_features"],
            sublinear_tf=True,
            lowercase=True,
            dtype=np.float32,
        )
        self.char = None
        if mcfg.get("char_ngrams"):
            self.char = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=tuple(mcfg["char_ngrams"]),
                min_df=mcfg["char_min_df"],
                max_features=mcfg["char_max_features"],
                sublinear_tf=True,
                lowercase=True,
                dtype=np.float32,
            )

    def fit_transform(self, texts: Sequence[str]) -> sp.csr_matrix:
        parts = [self.word.fit_transform(texts)]
        if self.char is not None:
            parts.append(self.char.fit_transform(texts))
        return self._join(parts)

    def transform(self, texts: Sequence[str]) -> sp.csr_matrix:
        parts = [self.word.transform(texts)]
        if self.char is not None:
            parts.append(self.char.transform(texts))
        return self._join(parts)

    def names(self) -> np.ndarray:
        """Readable feature names: word n-grams as they are, character n-grams quoted."""
        names = [self.word.get_feature_names_out()]
        if self.char is not None:
            names.append(np.array([f"'{g}'" for g in self.char.get_feature_names_out()]))
        return np.concatenate(names)

    @staticmethod
    def _join(parts: list[sp.csr_matrix]) -> sp.csr_matrix:
        if len(parts) == 1:
            return parts[0]
        # Each part is l2-normalised; scaling by 1/sqrt(2) keeps the joined rows at length 1.
        return (sp.hstack(parts, format="csr") / np.sqrt(len(parts))).astype(np.float32)


class GloveFeatures:
    """The mean GloVe vector of a text's words (words without a vector count as zeros)."""

    def __init__(self, mcfg: dict):
        self.mcfg = mcfg
        self.word2id: dict[str, int] = {}
        self.embeddings: np.ndarray | None = None
        self.scaler = StandardScaler()

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        vocab, embeddings = vocab_and_embeddings(
            list(texts), Path(self.mcfg["embedding_path"]), self.mcfg["embedding_dim"], 1
        )
        self.word2id = {w: i for i, w in enumerate(vocab)}
        self.embeddings = embeddings.numpy()
        return self.scaler.fit_transform(self._mean(texts))

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        return self.scaler.transform(self._mean(texts))

    def _mean(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.embeddings.shape[1]), dtype=np.float32)
        for i, text in enumerate(texts):
            ids = encode(text, self.word2id, self.mcfg["max_words"])
            out[i] = self.embeddings[ids].mean(axis=0)  # PAD and UNK rows are zero vectors
        return out


def make_features(mcfg: dict) -> TfidfFeatures | GloveFeatures:
    return GloveFeatures(mcfg) if mcfg["method"] == "glove_svm" else TfidfFeatures(mcfg)


# ---------------------------------------------------------------------------------------------
# Classifiers and the validation grid


def candidates(method: str, tcfg: dict) -> list[dict]:
    """Every setting of the validation grid for this method."""
    if method == "tfidf_nb":
        return [{"alpha": float(a)} for a in tcfg["nb_alpha"]]
    key = "lr_C" if method == "tfidf_lr" else "svm_C"
    return [{"C": float(c), "class_weight": w} for c in tcfg[key] for w in tcfg["class_weights"]]


def make_classifier(method: str, setting: dict):
    if method == "tfidf_nb":
        return MultinomialNB(alpha=setting["alpha"])
    if method == "tfidf_lr":
        return LogisticRegression(
            C=setting["C"], class_weight=setting["class_weight"], max_iter=3000
        )
    return LinearSVC(C=setting["C"], class_weight=setting["class_weight"], random_state=0)


def scores_of(clf, x) -> np.ndarray:
    """Class scores for argmax: probabilities, or SVM margins (one column per class)."""
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(x)
    margins = clf.decision_function(x)
    return np.column_stack([-margins, margins]) if margins.ndim == 1 else margins


def select(method: str, tcfg: dict, x_train, y_train, x_val, y_val) -> tuple[object, dict, list]:
    """Fit every grid setting on train; keep the best by validation macro-F1.

    The SVMs are scored as the calibrated model that gets logged and saved, not as the plain
    SVM, so the grid picks the setting for the model it reports (T44 review).
    """
    grid, best, best_f1, best_setting = [], None, -1.0, None
    for setting in candidates(method, tcfg):
        start = time.perf_counter()
        if method in ("tfidf_svm", "glove_svm"):
            clf = calibrated(method, setting, tcfg["calibration_folds"], x_train, y_train)
        else:
            clf = make_classifier(method, setting).fit(x_train, y_train)
        f1 = f1_score(y_val, scores_of(clf, x_val).argmax(axis=1), average="macro")
        grid.append({**setting, "val_macro_f1": f1, "seconds": time.perf_counter() - start})
        print(f"  {setting}  val macro-F1 {f1:.4f}")
        if f1 > best_f1:
            best, best_f1, best_setting = clf, f1, setting
    return best, best_setting, grid


def calibrated(method: str, setting: dict, folds: int, x_train, y_train):
    """An SVM with sigmoid calibration, so it outputs probabilities (for ROC-AUC and ECE).
    Naive Bayes and LR already do.

    Balanced class weights go in as sample weights, so the sigmoid fit is balanced too, like
    LR's balanced probabilities. With ``class_weight`` on the SVM alone, the unweighted
    sigmoid fit undoes most of the balancing and the model leans to the majority labels again.
    """
    balanced = setting["class_weight"] == "balanced"
    svm = make_classifier(method, {**setting, "class_weight": None})
    weights = compute_sample_weight("balanced", y_train) if balanced else None
    clf = CalibratedClassifierCV(svm, method="sigmoid", cv=folds)
    return clf.fit(x_train, y_train, sample_weight=weights)


def class_weights_of(clf, n_labels: int) -> np.ndarray:
    """One weight row per label: how much each feature pushes towards that label."""
    if isinstance(clf, MultinomialNB):
        logp = clf.feature_log_prob_  # log P(feature | label); compare with the other labels
        return np.stack(
            [logp[i] - np.delete(logp, i, axis=0).mean(axis=0) for i in range(n_labels)]
        )
    coef = clf.coef_
    return np.vstack([-coef[0], coef[0]]) if coef.shape[0] == 1 and n_labels == 2 else coef


def top_words(clf, names: np.ndarray, info: DatasetInfo, n: int = TOP_N) -> pd.DataFrame:
    """Per label, the ``n`` word n-grams and the ``n`` character n-grams (quoted names) with
    the largest weight towards it. Word n-grams are the ones to read next to SHAP."""
    weights = class_weights_of(clf, info.num_labels)
    is_char = np.char.startswith(names.astype(str), "'")
    rows = []
    for label_id, row in enumerate(weights):
        for kind, mask in (("word", ~is_char), ("char", is_char)):
            idx = np.flatnonzero(mask)
            for rank, j in enumerate(idx[np.argsort(-row[idx])][:n], start=1):
                rows.append({"label": info.labels[label_id], "kind": kind, "rank": rank,
                             "feature": names[j], "weight": round(float(row[j]), 4)})  # fmt: skip
    return pd.DataFrame(rows)


def count_params(clf) -> int:
    """Learned numbers: weights and biases (Naive Bayes: one log-probability per feature and
    label, plus the label priors)."""
    if isinstance(clf, MultinomialNB):
        return int(clf.feature_log_prob_.size + clf.class_log_prior_.size)
    if isinstance(clf, CalibratedClassifierCV):
        return sum(count_params(c.estimator) for c in clf.calibrated_classifiers_)
    return int(clf.coef_.size + np.size(clf.intercept_))


# ---------------------------------------------------------------------------------------------
# Training


def train_classic(cfg: dict) -> dict:
    """Fit one traditional baseline from a config dict (see ``configs/classic/``)."""
    cfg = with_defaults(cfg)
    run, data, mcfg, tcfg = cfg["run"], cfg["data"], cfg["model"], cfg["train"]
    method, seed = mcfg["method"], int(run["seed"])
    info = get_dataset(data["dataset"])
    splits = {split: df.assign(text=single_text(df)) for split, df in load_inputs(cfg).items()}
    texts = {split: df["text"].tolist() for split, df in splits.items()}
    labels = {split: df["label"].to_numpy() for split, df in splits.items()}
    print(f"{run['name']}: {info.name} ({info.num_labels} labels), {method}, "
          f"{len(texts['train']):,} train texts")  # fmt: skip

    start = time.perf_counter()
    features = make_features(mcfg)
    x = {"train": features.fit_transform(texts["train"])}
    x["validation"] = features.transform(texts["validation"])
    print(f"  {x['train'].shape[1]:,} features in {time.perf_counter() - start:.0f} s")
    chosen, setting, grid = select(
        method, tcfg, x["train"], labels["train"], x["validation"], labels["validation"]
    )
    final = chosen
    if method in ("tfidf_svm", "glove_svm"):
        # The calibrated SVM has no single weight row to read; top_words reads the plain SVM
        # with the chosen setting, fitted on the whole training split.
        chosen = make_classifier(method, setting).fit(x["train"], labels["train"])
    train_seconds = time.perf_counter() - start
    x["test"] = features.transform(texts["test"])

    run_dir = OUTPUT_DIR / run["name"] / f"seed{seed}"
    (run_dir / "model").mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    joblib.dump(
        {"features": features, "classifier": final, "labels": list(info.labels), "method": method},
        run_dir / "model" / "model.joblib",
    )
    if isinstance(features, TfidfFeatures):
        words = top_words(chosen, features.names(), info)
        words.to_csv(run_dir / "top_words.csv", index=False)
        for label in info.labels:
            mine = words[(words["label"] == label) & (words["kind"] == "word")]
            top = mine["feature"].head(12).tolist()
            print(f"  towards {label}: {', '.join(top)}")

    predictor = ClassicPredictor(features, final, list(info.labels))
    latency = measure_latency_ms(predictor.predict_proba, texts["test"][:200])
    summary = {
        "run": run["name"],
        "seed": seed,
        "dataset": info.name,
        "variant": data.get("variant", "raw"),
        "family": "classic",
        "method": method,
        "setting": setting,
        "best_val_macro_f1": max(g["val_macro_f1"] for g in grid),
        "grid": grid,
        "n_features": int(x["train"].shape[1]),
        "trainable_params": count_params(final),
        "train_seconds": train_seconds,
        "latency_ms": latency,
        "device": "cpu",
    }
    with open(run_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  chosen {setting}; fitted in {train_seconds:.0f} s; {latency:.2f} ms per claim")
    log_run(
        cfg,
        info,
        {split: (splits[split], final.predict_proba(x[split])) for split in ("validation", "test")},
        extra={
            k: summary[k] for k in ("trainable_params", "train_seconds", "latency_ms", "device")
        },
    )
    return summary


# ---------------------------------------------------------------------------------------------
# Inference (app, analysis)


@dataclass
class ClassicPredictor:
    """A fitted traditional baseline as ``texts -> probabilities``.

    p = ClassicPredictor.load("outputs/tfidf_lr_covid/seed42/model/model.joblib")
    p.predict_proba(["5G spreads the coronavirus."])
    """

    features: TfidfFeatures | GloveFeatures
    classifier: object
    labels: list[str]

    @classmethod
    def load(cls, path: str | Path) -> ClassicPredictor:
        saved = joblib.load(path)
        return cls(saved["features"], saved["classifier"], saved["labels"])

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        return self.classifier.predict_proba(self.features.transform(list(texts)))
