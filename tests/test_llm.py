"""Tests for the LLM label scoring (T07) and its kNN and RAG modes (T20). They use the Qwen3
tokenizer (a few MB, downloaded once) with a fake model, so no GPU or 8B weights are needed."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from factcheck.data import get_dataset
from factcheck.models.llm import (
    PROMPTS,
    LLMScorer,
    score_llm,
    scored_splits,
    with_defaults,
    zero_shot_message,
)


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    except OSError:
        pytest.skip("Qwen3 tokenizer not downloadable here")


class FakeLM:
    """Next-token logits that depend only on the real (unpadded) tokens of each row."""

    device = torch.device("cpu")

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask, logits_to_keep, use_cache):
        score = (input_ids * attention_mask).sum(dim=1).double()
        logits = torch.zeros(len(input_ids), 1, self.vocab_size)
        for k in range(self.vocab_size):
            logits[:, 0, k] = torch.sin(score * (k + 1)).float()
        return SimpleNamespace(logits=logits)


def test_prompts_list_every_label_in_registry_order():
    for name, spec in PROMPTS.items():
        assert tuple(spec["labels"]) == get_dataset(name).labels


def test_message_contains_text_and_labels():
    msg = zero_shot_message("Vaccines contain microchips.", "covid")
    assert "Text: Vaccines contain microchips." in msg
    assert "Answer with exactly one label from: real, fake." in msg
    assert msg.endswith("with no other text or formatting.")


@pytest.mark.parametrize("dataset", ["covid", "liar", "pubhealth", "isot", "climate_fever"])
def test_label_first_tokens_differ(tokenizer, dataset):
    scorer = LLMScorer("x", get_dataset(dataset), model=FakeLM(len(tokenizer)), tokenizer=tokenizer)
    assert len(set(scorer.answer_ids)) == get_dataset(dataset).num_labels


def test_prompt_turns_thinking_off_and_ends_at_the_answer(tokenizer):
    scorer = LLMScorer("x", get_dataset("liar"), model=FakeLM(len(tokenizer)), tokenizer=tokenizer)
    assert scorer.prompt("A claim.").endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_long_text_is_cut_to_max_length(tokenizer):
    scorer = LLMScorer(
        "x", get_dataset("covid"), max_length=5, model=FakeLM(len(tokenizer)), tokenizer=tokenizer
    )
    assert "word " * 6 not in scorer.prompt("word " * 50)


def test_batched_scores_keep_the_input_order(tokenizer):
    scorer = LLMScorer("x", get_dataset("liar"), model=FakeLM(len(tokenizer)), tokenizer=tokenizer)
    texts = ["short", "a much much longer claim about taxes and jobs", "mid length claim", "x y"]
    batched = scorer.predict_proba(texts, batch_size=3)
    one_by_one = np.vstack([scorer.predict_proba([t], batch_size=1) for t in texts])
    np.testing.assert_allclose(batched, one_by_one, rtol=1e-5)
    np.testing.assert_allclose(batched.sum(axis=1), 1.0, rtol=1e-5)
    assert len(scorer.coverage) == 2 * len(texts)


def scorer_for(tokenizer, dataset: str, method: str) -> LLMScorer:
    return LLMScorer(
        "x", get_dataset(dataset), model=FakeLM(len(tokenizer)), tokenizer=tokenizer, method=method
    )


def test_knn_prompt_shows_the_examples_most_similar_last(tokenizer):
    examples = [("Most similar claim.", "false"), ("Less similar claim.", "true")]
    prompt = scorer_for(tokenizer, "liar", "knn").prompt("A claim.", examples)
    assert "Text: Most similar claim.\nLabel: false" in prompt
    assert prompt.index("Less similar claim.") < prompt.index("Most similar claim.")
    assert prompt.index("Most similar claim.") < prompt.index("Text: A claim.")
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_rag_prompt_numbers_the_evidence(tokenizer):
    evidence = ["Vitamin C does not cure colds.", "Trials found no effect."]
    prompt = scorer_for(tokenizer, "pubhealth", "rag").prompt("Vitamin C cures colds.", evidence)
    assert "1. Vitamin C does not cure colds.\n2. Trials found no effect." in prompt
    assert "Text: Vitamin C cures colds." in prompt


def test_retrieval_methods_need_a_context_per_text(tokenizer):
    scorer = scorer_for(tokenizer, "liar", "knn")
    with pytest.raises(ValueError, match="needs a context"):
        scorer.predict_proba(["A claim."])
    with pytest.raises(ValueError, match="one entry per text"):
        scorer.predict_proba(["A claim.", "Another."], contexts=[[("x", "true")]])


def test_contexts_follow_their_texts_through_the_batching(tokenizer):
    scorer = scorer_for(tokenizer, "pubhealth", "rag")
    texts = ["short", "a much longer claim about vaccines and side effects", "mid claim"]
    contexts = [["one sentence."], ["two", "sentences here."], ["a third, longer sentence."]]
    batched = scorer.predict_proba(texts, batch_size=2, contexts=contexts)
    one_by_one = np.vstack(
        [scorer.predict_proba([t], 1, [c]) for t, c in zip(texts, contexts, strict=True)]
    )
    np.testing.assert_allclose(batched, one_by_one, rtol=1e-5)


def test_unknown_method_is_refused(tokenizer):
    with pytest.raises(ValueError, match="method must be one of"):
        scorer_for(tokenizer, "liar", "fewshot")


def splits_frame(n: int, subset: int = 0) -> pd.DataFrame:
    return pd.DataFrame(
        {"id": [str(i) for i in range(n)], "in_llm_subset": [i < subset for i in range(n)]}
    )


def test_scored_splits_skip_empty_splits():
    cfg = with_defaults({"data": {"dataset": "climate_fever"}})
    splits = {"train": splits_frame(0), "validation": splits_frame(0), "test": splits_frame(5)}
    assert list(scored_splits(cfg, splits)) == ["test"]


def test_llm_subset_keeps_only_the_fixed_test_subset():
    cfg = with_defaults({"data": {"dataset": "isot", "llm_subset": True}})
    splits = {"validation": splits_frame(4), "test": splits_frame(6, subset=2)}
    scored = scored_splits(cfg, splits)
    assert list(scored) == ["test"]
    assert scored["test"]["id"].tolist() == ["0", "1"]
    with pytest.raises(ValueError, match="no LLM test subset"):
        scored_splits(cfg, {"validation": splits_frame(4), "test": splits_frame(6)})


def llm_cfg(dataset: str, method: str, **features) -> dict:
    return {
        "run": {"name": "t", "seed": 42},
        "data": {"dataset": dataset},
        "model": {"family": "llm", "name": "x", "method": method},
        "features": features,
    }


@pytest.mark.parametrize(
    ("cfg", "error"),
    [
        (llm_cfg("pubhealth", "rag"), "RAG mode"),  # rag without evidence
        (llm_cfg("pubhealth", "zeroshot", use_evidence=True), "RAG mode"),  # evidence, no rag
        (llm_cfg("liar", "rag", use_evidence=True), "use_evidence is for"),  # no evidence
    ],
)
def test_rag_and_evidence_switches_go_together(cfg, error):
    with pytest.raises(ValueError, match=error):
        score_llm(cfg)  # fails before any model or data is loaded


def test_unknown_llm_method_is_refused():
    with pytest.raises(ValueError, match="model.method must be"):
        score_llm(llm_cfg("liar", "fewshot"))
