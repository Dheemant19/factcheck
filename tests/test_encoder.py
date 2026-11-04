"""Tests for the encoder trainer (T06). They need a small Hugging Face tokenizer (downloaded
once, about 1 MB) but no datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factcheck.models.encoder import (
    encode_texts,
    load_model,
    softmax,
    stratified_subset,
    with_defaults,
)

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


def test_fp16_checkpoints_load_with_fp32_weights(tokenizer, tmp_path):
    """DeBERTa-v3 is saved in fp16; training fp16 weights with AdamW gives NaN on step 1."""
    import torch
    from transformers import AutoModelForSequenceClassification

    from factcheck.data import get_dataset

    AutoModelForSequenceClassification.from_pretrained(TINY, num_labels=2).half().save_pretrained(
        tmp_path
    )
    model = load_model(str(tmp_path), get_dataset("covid"))
    assert {p.dtype for p in model.parameters()} == {torch.float32}


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


# 2 layers x (query, key, value) / 2 layers x 6 linear layers + BERT's pooler
@pytest.mark.parametrize(("targets", "adapted"), [("attention", 6), ("all_linear", 13)])
def test_lora_adapts_only_the_chosen_encoder_layers(tokenizer, targets, adapted):
    pytest.importorskip("peft")
    from factcheck.data import get_dataset
    from factcheck.models.encoder import add_lora, count_trainable

    model = load_model(TINY, get_dataset("liar"))
    full = count_trainable(model)
    cfg = with_defaults({"model": {"method": "lora", "lora_r": 4, "lora_targets": targets}})
    lora = add_lora(model, cfg["model"])
    layers = {n.rsplit(".lora_A", 1)[0] for n, _ in lora.named_parameters() if ".lora_A" in n}
    assert len(layers) == adapted
    assert all(".bert." in n for n in layers)  # never the classification head
    assert 0 < count_trainable(lora) < full / 10
    assert any(p.requires_grad for n, p in lora.named_parameters() if "classifier" in n)


def test_merged_lora_model_saves_as_a_plain_model(tokenizer, tmp_path):
    pytest.importorskip("peft")
    import torch
    from transformers import AutoModelForSequenceClassification

    from factcheck.data import get_dataset
    from factcheck.models.encoder import add_lora

    cfg = with_defaults({"model": {"method": "lora", "lora_r": 4}})
    lora = add_lora(load_model(TINY, get_dataset("covid")), cfg["model"])
    for name, p in lora.named_parameters():
        if "lora_B" in name:  # B starts at zero; make the adapters actually change the output
            torch.nn.init.normal_(p, std=0.1)
    batch = tokenizer(["a claim"], return_tensors="pt")
    lora.eval()
    with torch.no_grad():
        before = lora(**batch).logits
    lora.merge_and_unload().save_pretrained(tmp_path)
    plain = AutoModelForSequenceClassification.from_pretrained(tmp_path).eval()
    with torch.no_grad():
        torch.testing.assert_close(plain(**batch).logits, before, atol=1e-5, rtol=1e-4)


def test_lora_on_a_qwen3_classifier_adapts_attention_and_trains_the_head():
    """T27's QLoRA target: a tiny random Qwen3ForSequenceClassification (no download)."""
    pytest.importorskip("peft")
    from transformers import Qwen3Config, Qwen3ForSequenceClassification

    from factcheck.models.encoder import add_lora

    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_labels=6,
        pad_token_id=0,
    )
    cfg = with_defaults({"model": {"method": "qlora", "lora_r": 4}})
    lora = add_lora(Qwen3ForSequenceClassification(config), cfg["model"])
    layers = {n.rsplit(".lora_A", 1)[0] for n, _ in lora.named_parameters() if ".lora_A" in n}
    assert len(layers) == 6  # q, k, v in each of 2 layers
    assert any(p.requires_grad for n, p in lora.named_parameters() if "score" in n)


def test_balanced_weights_are_inverse_frequency():
    from factcheck.models.encoder import balanced_weights

    weights = balanced_weights([0, 0, 0, 1], 2)  # n / (classes x count) = 4/6 and 4/2
    np.testing.assert_allclose(weights.numpy(), [4 / 6, 2.0], rtol=1e-6)
    np.testing.assert_allclose(balanced_weights([0, 1, 2], 3).numpy(), 1.0)
    with pytest.raises(ValueError, match="no training rows"):
        balanced_weights([0, 0], 2)


def test_weighted_loss_matches_a_hand_computed_value():
    from types import SimpleNamespace

    import torch

    from factcheck.models.encoder import WeightedLossTrainer

    logits = torch.tensor([[2.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    labels = torch.tensor([0, 1, 1])
    weights = torch.tensor([1.0, 3.0])
    trainer = WeightedLossTrainer.__new__(WeightedLossTrainer)  # only compute_loss is used
    trainer.class_weights = weights
    loss = trainer.compute_loss(lambda **_: SimpleNamespace(logits=logits), {"labels": labels})
    nll = -torch.log_softmax(logits, dim=-1)[torch.arange(3), labels]
    expected = (weights[labels] * nll).sum() / weights[labels].sum()
    torch.testing.assert_close(loss, expected)
