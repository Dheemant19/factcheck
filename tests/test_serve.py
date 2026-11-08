"""The app's engine (factcheck.serve): loading a run, scoring a claim, evidence and SHAP."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from factcheck import serve
from factcheck.data import get_dataset
from factcheck.retrieval import EvidenceStore

TINY = "google/bert_uncased_L-2_H-128_A-2"
CLAIM = "Drinking hot water kills the coronavirus."


def save_run(tmp_path, dataset: str, name: str = "tiny", features: dict | None = None):
    """A run folder like factcheck.train writes: model/ plus config.yaml next to it."""
    transformers = pytest.importorskip("transformers")
    info = get_dataset(dataset)
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(TINY)
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            TINY, num_labels=info.num_labels, id2label=info.id2label, label2id=info.label2id
        )
    except OSError:
        pytest.skip(f"{TINY} not downloadable here")
    run_dir = tmp_path / name / "seed42"
    model.save_pretrained(run_dir / "model")
    tokenizer.save_pretrained(run_dir / "model")
    cfg = {
        "run": {"name": name, "seed": 42},
        "data": {"dataset": dataset, "max_length": 64},
        "model": {"family": "encoder", "name": TINY},
        "features": features or {},
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return run_dir / "model"


def test_servable_runs(tmp_path):
    plain = save_run(tmp_path, "covid")
    meta = save_run(tmp_path, "liar", "meta", {"use_metadata": True})
    assert serve.is_servable(plain) and not serve.is_qlora(plain)
    assert not serve.is_servable(meta)  # a pasted claim has no speaker metadata
    assert not serve.is_servable(tmp_path / "missing" / "seed42" / "model")


def test_prediction_is_a_distribution_over_the_dataset_labels(tmp_path):
    clf = serve.Classifier(save_run(tmp_path, "liar"), device="cpu")
    pred = clf.predict(CLAIM)
    info = get_dataset("liar")
    assert list(pred.label_probs) == list(info.labels)
    assert sum(pred.label_probs.values()) == pytest.approx(1.0, abs=1e-5)
    fake = sum(pred.label_probs[info.labels[i]] for i in info.fake_label_ids)
    assert pred.fake_prob == pytest.approx(fake)
    assert pred.seconds > 0


def test_evidence_runs_read_the_claim_with_its_evidence(tmp_path):
    clf = serve.Classifier(
        save_run(tmp_path, "pubhealth", features={"use_evidence": True}), device="cpu"
    )
    assert clf.uses_evidence
    no_evidence = clf.encode([CLAIM])[0]  # an empty second segment, as in training
    paired = clf.encode([CLAIM], ["Hot water does not kill viruses in the body."])[0]
    sep = clf.tokenizer.sep_token_id
    claim_part = no_evidence[: no_evidence.index(sep) + 1]
    assert paired.count(sep) == 2 and len(paired) > len(no_evidence)
    assert paired[: len(claim_part)] == claim_part  # the claim comes first, untouched


def test_explanation_covers_every_token(tmp_path):
    pytest.importorskip("shap")
    clf = serve.Classifier(save_run(tmp_path, "covid"), device="cpu")
    why = serve.explain(clf, CLAIM, max_evals=40)
    assert len(why.tokens) == len(why.towards_false) > 3
    assert "".join(why.tokens).strip().lower().replace(" ", "") == CLAIM.lower().replace(" ", "")
    assert sum(why.probs) == pytest.approx(1.0, abs=0.05)


def test_evidence_search_merges_the_pools_best_first(monkeypatch):
    import factcheck.retrieval as retrieval

    def unit(rows):
        rows = np.asarray(rows, dtype=np.float32)
        return rows / np.linalg.norm(rows, axis=1, keepdims=True)

    def store(name, sentences, emb):
        return EvidenceStore(name, sentences, unit(emb).astype(np.float16), {}, {})

    search = serve.EvidenceSearch()
    search.__dict__["stores"] = {  # skip the cached pools: two tiny ones
        "pubhealth": store("pubhealth", ["health a", "health b"], [[1, 0], [0.6, 0.8]]),
        "climate_fever": store("climate_fever", ["climate a"], [[0.8, 0.6]]),
    }
    monkeypatch.setattr(retrieval, "embed", lambda texts, **_: unit([[1, 0]] * len(texts)))
    found = search.find(CLAIM, dataset=None, k_sentences=2)
    assert found.similar == []  # no dataset, no claim index
    assert [s.text for s in found.sentences] == ["health a", "climate a"]
    assert [s.source for s in found.sentences] == ["pubhealth", "climate_fever"]
    assert found.sentence_texts(1) == ["health a"]


def test_sub_word_pieces_join_into_words():
    why = serve.Explanation(
        tokens=["", "Vitamin ", "cures ", "CO", "VID", "-", "19 ", "fast", "."],
        towards_false=np.array([0.0, 0.3, 0.2, 0.1, 0.1, 0.05, 0.05, -0.1, 0.0]),
        base=0.0,
        probs=[0.5, 0.5],
        seconds=0.0,
        evals=10,
    )
    words = why.words()
    assert [w for w, _ in words] == ["Vitamin ", "cures ", "COVID-19 ", "fast."]
    assert words[2][1] == pytest.approx(0.3)


# --------------------------------------------------------------------------- baselines (T58)

REAL = ["CDC data shows cases rising in two states", "WHO publishes new testing data today",
        "Health officials report case data by state", "State data on tests released by CDC"]  # fmt: skip
FAKE = ["Miracle cure kills corona overnight", "Garlic cure stops corona, doctors stunned",
        "Secret corona cure they hide from you", "This cure ends corona in one day"]  # fmt: skip


def save_baseline(tmp_path, dataset: str = "covid", char_ngrams=(2, 4), labels=None):
    """A TF-IDF + logistic regression run folder like factcheck.models.classic writes."""
    import joblib
    from sklearn.linear_model import LogisticRegression

    from factcheck.models import classic

    info = get_dataset(dataset)
    mcfg = classic.with_defaults({"model": {"method": "tfidf_lr"}})["model"]
    mcfg = {**mcfg, "word_min_df": 1, "char_min_df": 1, "char_ngrams": char_ngrams}
    features = classic.TfidfFeatures(mcfg)
    x = features.fit_transform(REAL + FAKE)
    y = labels or [info.labels.index("real")] * 4 + [info.labels.index("fake")] * 4
    clf = LogisticRegression(C=10).fit(x, y)
    run_dir = tmp_path / f"tfidf_lr_{dataset}" / "seed42"
    (run_dir / "model").mkdir(parents=True)
    joblib.dump({"features": features, "classifier": clf, "labels": info.labels,
                 "method": "tfidf_lr"}, run_dir / "model" / "model.joblib")  # fmt: skip
    cfg = {"run": {"name": f"tfidf_lr_{dataset}", "seed": 42}, "data": {"dataset": dataset},
           "model": {"family": "classic", "name": "tfidf", "method": "tfidf_lr"}}  # fmt: skip
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return run_dir / "model"


def test_only_logistic_regression_baselines_are_servable(tmp_path):
    model_dir = save_baseline(tmp_path)
    assert serve.is_baseline(model_dir) and serve.is_servable(model_dir)
    cfg_path = model_dir.parent / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["model"]["method"] = "tfidf_nb"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert not serve.is_baseline(model_dir) and not serve.is_servable(model_dir)


def test_baseline_predicts_like_a_classifier(tmp_path):
    base = serve.load(save_baseline(tmp_path))
    assert isinstance(base, serve.Baseline) and not base.uses_evidence and not base.qlora
    pred = base.predict("Miracle garlic cure for corona")
    assert set(pred.label_probs) == set(base.info.labels)
    assert sum(pred.label_probs.values()) == pytest.approx(1.0)
    assert pred.fake_prob == pytest.approx(pred.label_probs["fake"]) and pred.fake_prob > 0.5


def test_linear_why_adds_up_to_the_log_odds_of_false(tmp_path):
    base = serve.load(save_baseline(tmp_path))
    claim = "  Miracle cure: garlic stops corona, CDC data says."
    why = serve.explain_linear(base, claim)
    p = base.predict(claim).fake_prob
    total = why.base + sum(v for _, v in why.features) + why.rest
    assert total == pytest.approx(np.log(p / (1 - p)), abs=1e-4)
    # The word shares cover the word n-grams exactly; the pieces rebuild the claim.
    assert why.towards_false.sum() == pytest.approx(sum(v for _, v in why.features), abs=1e-6)
    assert "".join(why.tokens) == claim
    assert why.evals == 0 and abs(why.rest) > 0
    pushes = dict(why.features)
    assert pushes["cure"] > 0 and pushes["data"] < 0


def test_linear_why_without_character_ngrams(tmp_path):
    base = serve.load(save_baseline(tmp_path, char_ngrams=None))
    why = serve.explain_linear(base, "Garlic cure stops corona")
    assert why.rest == 0.0
    words = dict(why.words())
    assert words["cure "] > 0  # the unigram plus half of each pair it is in


def test_linear_why_on_a_many_label_dataset(tmp_path):
    info = get_dataset("liar")
    labels = [info.labels.index(x) for x in ["true", "mostly-true", "half-true", "true"]] + [
        info.labels.index(x) for x in ["false", "pants-fire", "barely-true", "false"]
    ]
    base = serve.load(save_baseline(tmp_path, "liar", labels=labels))
    why = serve.explain_linear(base, "Secret corona cure")
    assert base.direction.shape == (len(base.names),)
    assert len(why.probs) == info.num_labels
    assert dict(why.features)["cure"] > 0
