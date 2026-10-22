"""Tests for the Bi-LSTM baseline (T05). None of them need the datasets or the GloVe file."""

from __future__ import annotations

import numpy as np
import torch

from factcheck.models.lstm import (
    PAD,
    UNK,
    BiLSTMClassifier,
    LSTMPredictor,
    build_vocab,
    count_trainable,
    encode,
    glove_words_and_vectors,
    make_loader,
    predict_proba,
    tokenize,
    with_defaults,
)


def test_tokenize_matches_glove_style():
    assert tokenize("Don't panic, it's 100% FAKE!") == [
        "do",
        "n't",
        "panic",
        ",",
        "it",
        "'s",
        "100",
        "%",
        "fake",
        "!",
    ]


def test_vocab_keeps_only_train_words_with_vectors():
    vocab = build_vocab(["the cat sat", "the dog"], glove_words={"the", "cat", "dog"})
    assert vocab[:2] == [PAD, UNK]
    assert vocab[2] == "the"  # most frequent first
    assert set(vocab[2:]) == {"the", "cat", "dog"}
    word2id = {w: i for i, w in enumerate(vocab)}
    assert encode("the sat cat", word2id, max_length=2) == [2, 1]
    assert encode("", word2id, max_length=5) == [1]  # never empty


def test_glove_reader(tmp_path):
    path = tmp_path / "glove.txt"
    path.write_text("the 0.1 0.2\ncat 0.3 0.4\n", encoding="utf-8")
    words, vectors = glove_words_and_vectors(path, wanted={"cat"})
    assert words == {"the", "cat"}
    np.testing.assert_allclose(vectors["cat"], [0.3, 0.4])


def test_model_shapes_and_frozen_embeddings():
    model = BiLSTMClassifier(torch.randn(50, 300), num_labels=6)
    assert not model.embedding.weight.requires_grad
    assert 2_000_000 < count_trainable(model) < 3_000_000  # plan Section 8.1
    loader = make_loader([[2, 3, 4], [5], [6, 7]], [0, 1, 2], batch_size=2, shuffle=False)
    probs = predict_proba(model, loader, torch.device("cpu"))
    assert probs.shape == (3, 6)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)


def test_padding_does_not_change_the_output():
    model = BiLSTMClassifier(torch.randn(20, 300), num_labels=2).eval()
    alone = make_loader([[2, 3]], [0], batch_size=1, shuffle=False)
    padded = make_loader([[2, 3], [4, 5, 6, 7, 8, 9]], [0, 0], batch_size=2, shuffle=False)
    np.testing.assert_allclose(
        predict_proba(model, alone, torch.device("cpu"))[0],
        predict_proba(model, padded, torch.device("cpu"))[0],
        rtol=1e-5,
    )


def test_predictor_round_trip(tmp_path):
    vocab = [PAD, UNK, "vaccine", "cures"]
    model = BiLSTMClassifier(torch.randn(len(vocab), 300), num_labels=2).eval()
    path = tmp_path / "model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocab": vocab,
            "labels": ["real", "fake"],
            "max_length": 128,
            "hidden_size": 256,
            "num_layers": 2,
            "dropout": 0.3,
        },
        path,
    )
    probs = LSTMPredictor(path, device="cpu").predict_proba(["Vaccine cures", "unknown words"])
    assert probs.shape == (2, 2)


def test_defaults_fill_and_coerce():
    cfg = with_defaults(
        {"run": {}, "data": {}, "model": {"family": "lstm"}, "train": {"learning_rate": "3e-4"}}
    )
    assert cfg["train"]["learning_rate"] == 3e-4  # YAML can read 3e-4 as a string
    assert cfg["train"]["batch_size"] == 64
    assert cfg["model"]["hidden_size"] == 256
