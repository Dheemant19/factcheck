"""Tests for the feature switches: LIAR metadata and evidence (T15)."""

from __future__ import annotations

import pandas as pd
import pytest

from factcheck.data import load_splits
from factcheck.inputs import (
    FEATURE_DEFAULTS,
    build_inputs,
    features,
    load_inputs,
    metadata_prefix,
    single_text,
)
from factcheck.paths import RAW_DIR


def cfg(dataset: str, **feats) -> dict:
    return {"data": {"dataset": dataset, "variant": "raw"}, "features": feats}


def liar_rows() -> pd.DataFrame:
    return pd.DataFrame({
        "id": ["1", "2"], "text": ["Taxes went up.", "Jobs went down."], "label": [1, 4],
        "meta_speaker": ["rick-perry", ""], "meta_party": ["republican", "democrat"],
        "meta_subject": ["taxes,economy", "jobs"], "meta_context": ["a debate", ""],
    })  # fmt: skip


def test_switches_are_off_by_default():
    assert features(cfg("liar")) == FEATURE_DEFAULTS
    assert not FEATURE_DEFAULTS["use_metadata"] and not FEATURE_DEFAULTS["use_evidence"]


@pytest.mark.parametrize(
    ("dataset", "feats", "message"),
    [
        ("covid", {"use_metadata": True}, "use_metadata is for"),
        ("liar", {"use_evidence": True}, "use_evidence is for"),
        ("liar", {"use_metadat": True}, "unknown features"),
    ],
)
def test_wrong_switches_fail_clearly(dataset, feats, message):
    with pytest.raises(ValueError, match=message):
        features(cfg(dataset, **feats))


def test_metadata_prefix():
    rows = liar_rows()
    assert metadata_prefix(rows.iloc[0]) == (
        "[speaker: rick perry | party: republican | subject: taxes, economy | context: a debate]"
    )
    assert "speaker: unknown" in metadata_prefix(rows.iloc[1])
    assert "context: unknown" in metadata_prefix(rows.iloc[1])


def test_metadata_changes_only_the_text():
    rows = liar_rows()
    out = build_inputs(rows, cfg("liar", use_metadata=True))
    assert out["text"][0] == metadata_prefix(rows.iloc[0]) + " Taxes went up."
    assert out["claim"].tolist() == rows["text"].tolist()
    assert out.drop(columns=["text", "claim"]).equals(rows.drop(columns="text"))
    assert rows["text"][0] == "Taxes went up."  # the input frame is not changed


def test_evidence_becomes_the_text_pair(monkeypatch):
    calls = {}

    def fake_evidence_for(name, df, k, source):
        calls.update(name=name, k=k, source=source)
        return [[f"e{i}a", f"e{i}b"][:k] for i in range(len(df))]

    monkeypatch.setattr("factcheck.retrieval.evidence_for", fake_evidence_for)
    rows = pd.DataFrame({"id": ["a", "b"], "text": ["claim a", "claim b"], "label": [0, 2]})
    out = build_inputs(rows, cfg("pubhealth", use_evidence=True, evidence_k=2))
    assert out["text_pair"].tolist() == ["e0a e0b", "e1a e1b"]
    assert out["text"].tolist() == ["claim a", "claim b"]  # the claim stays the first input
    assert calls == {"name": "pubhealth", "k": 2, "source": "own"}
    assert single_text(out).tolist() == ["claim a e0a e0b", "claim b e1a e1b"]
    assert single_text(rows).tolist() == ["claim a", "claim b"]


@pytest.mark.skipif(not (RAW_DIR / "liar").exists(), reason="no raw LIAR data")
def test_liar_metadata_on_the_real_data():
    plain, meta = load_splits("liar"), load_inputs(cfg("liar", use_metadata=True))
    for split in plain:
        assert meta[split]["text"].str.startswith("[speaker: ").all()
        assert (meta[split]["label"] == plain[split]["label"]).all()
        assert (meta[split]["claim"] == plain[split]["text"]).all()
        assert not [c for c in meta[split].columns if c.endswith("_counts")]


@pytest.mark.parametrize("name", ["covid", "liar", "pubhealth", "isot", "climate_fever"])
def test_switches_off_leave_the_text_alone(name):
    if not (RAW_DIR / name).exists():
        pytest.skip(f"no raw {name} data")
    plain, inputs = load_splits(name), load_inputs(cfg(name))
    for split in plain:
        assert (inputs[split]["text"] == plain[split]["text"]).all()
        assert "text_pair" not in inputs[split]
