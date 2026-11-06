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
