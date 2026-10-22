"""Tests for the encoder trainer (T06). They need a small Hugging Face tokenizer (downloaded
once, about 1 MB) but no datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factcheck.models.encoder import encode_texts, softmax, stratified_subset, with_defaults

TINY = "google/bert_uncased_L-2_H-128_A-2"


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(TINY)
    except OSError:
        pytest.skip(f"{TINY} not downloadable here")


def test_plain_truncation_keeps_the_head(tokenizer):
    (ids,) = encode_texts([" ".join(["word"] * 50)], tokenizer, max_length=10)
    assert len(ids) == 10
    assert ids[0] == tokenizer.cls_token_id and ids[-1] == tokenizer.sep_token_id


def test_head_and_tail_truncation(tokenizer):
    text = "start " + " ".join(["middle"] * 50) + " finish"
    (ids,) = encode_texts([text], tokenizer, max_length=8, head_tokens=2)
    tokens = tokenizer.convert_ids_to_tokens(ids)
    # [CLS] + 2 head tokens + 4 tail tokens + [SEP]
    assert tokens == ["[CLS]", "start", "middle", "middle", "middle", "middle", "finish", "[SEP]"]


def test_short_text_is_not_changed_by_head_and_tail(tokenizer):
    plain = encode_texts(["a short claim"], tokenizer, max_length=512)
    assert encode_texts(["a short claim"], tokenizer, max_length=512, head_tokens=128) == plain


def test_softmax_rows_sum_to_one():
    probs = softmax(np.array([[1000.0, 1000.0], [0.0, 1.0]]))  # no overflow
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    np.testing.assert_allclose(probs[0], [0.5, 0.5])


def test_stratified_subset_keeps_the_balance():
    df = pd.DataFrame({"label": [0] * 80 + [1] * 20})
    sub = stratified_subset(df, 50, seed=42)
    assert len(sub) == 50 and (sub["label"] == 1).sum() == 10


def test_defaults_fill_and_coerce():
    cfg = with_defaults(
        {"run": {}, "data": {}, "model": {"name": "x"}, "train": {"learning_rate": "3e-5"}}
    )
    assert cfg["train"]["learning_rate"] == 3e-5
    assert cfg["train"]["warmup_ratio"] == 0.06
    assert cfg["model"]["method"] == "full"
