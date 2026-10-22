"""Tests for the LLM label scoring (T07). They use the Qwen3 tokenizer (a few MB, downloaded
once) with a fake model, so no GPU or 8B weights are needed."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from factcheck.data import get_dataset
from factcheck.models.llm import PROMPTS, LLMScorer, zero_shot_message


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

    def __call__(self, input_ids, attention_mask, logits_to_keep):
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
    assert msg.endswith("Answer with exactly one label from: real, fake.")


@pytest.mark.parametrize("dataset", ["covid", "liar", "pubhealth", "isot"])
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
