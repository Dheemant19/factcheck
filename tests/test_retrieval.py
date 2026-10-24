"""Tests for the retrieval index (T08).

They use a tiny bag-of-words hashing "encoder" instead of bge-small, so they run in seconds,
download nothing and need no GPU. Tests on the real datasets are skipped without the raw data.
"""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd
import pytest

from factcheck.data import get_dataset, load_splits
from factcheck.paths import RAW_DIR
from factcheck.retrieval import (
    ClaimIndex,
    build_claim_index,
    build_evidence,
    index_texts,
    top_k_rows,
)


def fake_encoder(texts: list[str]) -> np.ndarray:
    """Unit-length hashed bag of words: same text -> same vector, shared words -> similar."""
    out = np.full((len(texts), 256), 1e-3, dtype=np.float32)
    for i, text in enumerate(texts):
        for word in text.lower().split():
            out[i, zlib.crc32(word.encode()) % 256] += 1
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def needs_data(name: str):  # noqa: ANN201
    return pytest.mark.skipif(not (RAW_DIR / name).exists(), reason=f"no raw {name} data")


def toy_index() -> ClaimIndex:
    texts = ["vaccines cause autism", "the moon is made of cheese", "taxes rose last year",
             "vaccines are safe and tested"]  # fmt: skip
    return ClaimIndex(
        info=get_dataset("covid"),
        ids=np.array(["a", "b", "c", "d"]),
        texts=texts,
        labels=np.array([1, 1, 0, 0]),
        emb=fake_encoder(texts),
        encoder=fake_encoder,
    )


def test_top_k_rows_is_sorted_and_clamped():
    scores = np.array([[0.1, 0.9, 0.5], [0.3, 0.2, 0.8]])
    top, values = top_k_rows(scores, 2)
    assert top.tolist() == [[1, 2], [2, 0]]
    assert values.tolist() == [[0.9, 0.5], [0.8, 0.3]]
    assert top_k_rows(scores, 10)[0].shape == (2, 3)  # k larger than the index


def test_search_finds_the_closest_claim_and_excludes_itself():
    index = toy_index()
    hits = index.search_texts(["vaccines cause autism"], k=2)[0]
    assert hits[0].id == "a" and hits[0].score == pytest.approx(1.0, abs=1e-5)
    assert hits[0].label_name == "fake"
    positions, _ = index.search(index.emb, k=1, exclude_ids=list(index.ids))
    assert (positions[:, 0] != np.arange(len(index))).all()  # nobody retrieves itself


def test_vote_is_a_probability_distribution():
    index = toy_index()
    probs = index.vote(index.embed_queries(["vaccines cause autism in kids"]), k=3)
    assert probs.shape == (1, 2)
    assert probs.sum() == pytest.approx(1.0)
    assert probs.argmax() == 1  # nearest claim "vaccines cause autism" is fake


def test_isot_queries_use_the_start_of_the_article():
    df = pd.DataFrame({"text": [" ".join(f"w{i}" for i in range(500))]})
    assert len(index_texts(df, "isot")[0].split()) == 100
    assert index_texts(df, "liar")[0] == df["text"][0]


@needs_data("liar")
def test_claim_index_holds_train_claims_only():
    index = build_claim_index("liar", encoder=fake_encoder)
    splits = load_splits("liar")
    assert set(index.ids) == set(splits["train"]["id"])
    held_out = set(splits["validation"]["id"]) | set(splits["test"]["id"])
    assert not held_out & set(index.ids)  # a prompt can never show a val/test verdict
    sample = slice(0, 500)
    positions, _ = index.search(index.emb[sample], k=8, exclude_ids=list(index.ids[sample]))
    assert (positions != np.arange(500)[:, None]).all()


@needs_data("climate_fever")
def test_climate_fever_evidence_belongs_to_its_claim():
    store = build_evidence("climate_fever", encoder=fake_encoder)
    test = load_splits("climate_fever")["test"]
    assert set(store.members) == set(test["id"])
    assert all(len(own) == 5 for own in store.members.values())
    assert len(store.sentences) <= 5 * len(test)  # shared evidence sentences are stored once
    row = test.iloc[0]
    own = set(store.sentences[p] for p in store.members[row["id"]])
    top = store.top_k(row["id"], k=5)
    assert len(top) == len(own) and set(top) <= own
    assert len(store.search_pool([row["text"]], k=3)[0]) == 3
