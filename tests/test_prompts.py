"""Tests for the prompt and input builders (T14)."""

from __future__ import annotations

import pytest

from factcheck.data import get_dataset, load_splits
from factcheck.models.encoder import encode_texts
from factcheck.paths import CACHE_DIR
from factcheck.prompts import (
    PROMPTS,
    answer_instruction,
    encoder_pair,
    knn_messages,
    rag_messages,
    shorten,
    zero_shot_message,
)

EXAMPLES = [  # most similar first, as the retrieval index returns them
    ("Vaccines cause autism in children.", "fake"),
    ("The MMR vaccine was linked to autism in a 1998 study.", "real"),
    ("Masks reduce the spread of the virus.", "real"),
]


def content(messages: list[dict]) -> str:
    assert len(messages) == 1 and messages[0]["role"] == "user"
    return messages[0]["content"]


def test_every_dataset_has_a_prompt_in_registry_order():
    for name in ("covid", "liar", "pubhealth", "isot", "climate_fever"):
        assert tuple(PROMPTS[name]["labels"]) == get_dataset(name).labels


def test_zero_shot_message_is_unchanged_from_t07():
    msg = zero_shot_message("Vaccines contain microchips.", "covid")
    assert "Text: Vaccines contain microchips." in msg
    assert msg.endswith(answer_instruction("covid"))


def test_knn_prompt_puts_the_most_similar_example_last():
    text = content(knn_messages("Do vaccines cause autism?", EXAMPLES, "covid"))
    positions = [text.index(example) for example, _ in EXAMPLES]
    assert positions == sorted(positions, reverse=True)  # most similar is closest to the claim
    assert text.index(EXAMPLES[0][0]) < text.index("Text: Do vaccines cause autism?")
    assert "Label: fake" in text and "Label: real" in text
    assert text.endswith(answer_instruction("covid"))


def test_knn_prompt_is_deterministic_and_shortens_long_examples():
    long_example = [(" ".join(f"w{i}" for i in range(300)), "fake")]
    a = content(knn_messages("A claim.", long_example, "isot"))
    assert a == content(knn_messages("A claim.", long_example, "isot"))
    assert "w59 ..." in a and "w60" not in a
    assert shorten("short text") == "short text"


def test_rag_prompt_numbers_the_evidence():
    evidence = ["The vaccine was tested on 40,000 people.", "No link to autism was found."]
    text = content(rag_messages("Vaccines cause autism.", evidence, "pubhealth"))
    assert "1. The vaccine was tested" in text and "2. No link to autism" in text
    assert "using only this evidence" in text
    assert text.index("2. No link") < text.index("Text: Vaccines cause autism.")


def test_encoder_pair_keeps_the_order():
    assert encoder_pair("claim", ["best", "second"]) == ("claim", "best second")


@pytest.fixture(scope="module")
def tiny_tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
    except OSError:
        pytest.skip("tiny BERT tokenizer not downloadable here")


def test_pairs_cut_only_the_evidence(tiny_tokenizer):
    claim, evidence = encoder_pair("vaccines cause autism", ["no " * 50, "unused"])
    (ids,) = encode_texts([claim], tiny_tokenizer, max_length=12, pairs=[evidence])
    tokens = tiny_tokenizer.convert_ids_to_tokens(ids)
    assert len(ids) == 12
    assert tokens[:6] == ["[CLS]", "vaccines", "cause", "autism", "[SEP]", "no"]  # claim intact
    assert tokens[-1] == "[SEP]"
    with pytest.raises(ValueError, match="head_tokens"):
        encode_texts([claim], tiny_tokenizer, max_length=12, head_tokens=4, pairs=[evidence])


def test_knn_prompt_fits_the_token_budget():
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    except OSError:
        pytest.skip("Qwen3 tokenizer not downloadable here")
    article = "Senator says the bill will cost billions of dollars over ten years. " * 60
    examples = [(article, "real" if i % 2 else "fake") for i in range(8)]
    query = tok.decode(tok.encode(article)[:512])  # the LLM scorer cuts the claim to max_length
    prompt = tok.apply_chat_template(
        knn_messages(query, examples, "isot"), tokenize=True, add_generation_prompt=True
    )
    ids = prompt["input_ids"] if isinstance(prompt, dict) else prompt
    assert len(ids) < 2000


@pytest.mark.skipif(
    not list((CACHE_DIR / "retrieval").glob("liar-claims-*.npy")),
    reason="no LIAR claim index cached (python -m factcheck.retrieval build --datasets liar)",
)
def test_knn_examples_never_leak_held_out_claims():
    from factcheck.retrieval import knn_examples

    splits = load_splits("liar")
    held_out = set(splits["validation"]["id"]) | set(splits["test"]["id"])
    val = splits["validation"].head(200)
    for hits in knn_examples("liar", val, "validation", k=8):
        assert len(hits) == 8 and not {h.id for h in hits} & held_out
    train = splits["train"].head(300)
    for cid, hits in zip(train["id"], knn_examples("liar", train, "train", k=8), strict=True):
        assert cid not in {h.id for h in hits}  # a training claim never retrieves itself
