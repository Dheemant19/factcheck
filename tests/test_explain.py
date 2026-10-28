"""Tests for the SHAP pipeline (T16). They use a tiny, untrained BERT saved as a fake run
folder (downloaded once, about 20 MB) but no datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

pytest.importorskip("shap")

from factcheck.data import get_dataset  # noqa: E402
from factcheck.explain import (  # noqa: E402
    check_logged_predictions,
    explain_claim,
    explain_dataset_batch,
    get_shap_explainer,
    resolve_checkpoint,
    stratified_examples,
    word_tokenizer,
)

TINY = "google/bert_uncased_L-2_H-128_A-2"


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    """A run folder like factcheck.train writes: model/ and config.yaml (LIAR, 6 labels)."""
    transformers = pytest.importorskip("transformers")
    info = get_dataset("liar")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(TINY)
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            TINY, num_labels=info.num_labels, id2label=info.id2label, label2id=info.label2id
        )
    except OSError:
        pytest.skip(f"{TINY} not downloadable here")
    path = tmp_path_factory.mktemp("tiny_liar") / "seed7"
    model.save_pretrained(path / "model")
    tokenizer.save_pretrained(path / "model")
    cfg = {
        "run": {"name": "tiny_liar", "seed": 7},
        "data": {"dataset": "liar", "variant": "raw", "max_length": 32, "head_tokens": None},
    }
    (path / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def explainer(run_dir):
    return get_shap_explainer(str(run_dir), device="cpu")


def test_run_folder_settings_are_used(run_dir, explainer):
    assert resolve_checkpoint(str(run_dir / "model"))[1]["run"]["name"] == "tiny_liar"
    assert explainer.name == "tiny_liar_seed7"  # model and seed in the file name
    assert explainer.info.name == "liar" and explainer.max_length == 32


def test_missing_local_checkpoint_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(str(tmp_path / "outputs" / "nope"))


def test_explain_claim_adds_up_to_the_prediction(explainer):
    result = explain_claim("Vaccines cause autism, says the senator.", explainer, max_evals=60)
    tokens = result["token_attributions"]
    assert tokens and {"token", "values", "towards_false"} <= tokens[0].keys()
    assert len(tokens[0]["values"]) == 6
    assert sum(result["probs"]) == pytest.approx(1.0, abs=1e-3)
    # towards_false is the sum over LIAR's false labels (pants-fire, false, barely-true)
    assert tokens[0]["towards_false"] == pytest.approx(sum(tokens[0]["values"][:3]), abs=1e-6)
    p_false = explainer.shap.model(np.array([result["text"]]))[0][:3].sum()
    assert result["p_false"] == pytest.approx(float(p_false), abs=1e-3)


def test_batch_saves_tokens_aligned_with_values(explainer, tmp_path):
    examples = pd.DataFrame(
        {
            "id": ["a", "b"],
            "text": ["The economy grew.", "Taxes doubled last year."],
            "label": [5, 1],
        }
    )
    html, npz = explain_dataset_batch(examples, explainer, tmp_path, max_evals=40)
    assert html.name == "tiny_liar_seed7_shap.html" and "<" in html.read_text(encoding="utf-8")
    z = np.load(npz, allow_pickle=True)
    assert list(z["ids"]) == ["a", "b"] and list(z["output_names"]) == list(explainer.info.labels)
    for tokens, values in zip(z["tokens"], z["values"], strict=True):
        assert values.shape == (len(tokens), 6)
    assert z["probs"].shape == (2, 6)


def test_stratified_examples_are_fixed_and_balanced():
    df = pd.DataFrame({"id": [str(i) for i in range(200)], "label": [0] * 150 + [1] * 50})
    first, second = stratified_examples(df, 20), stratified_examples(df, 20)
    assert list(first["id"]) == list(second["id"])
    assert (first["label"] == 1).sum() == 5


def test_stale_checkpoint_is_flagged(explainer, tmp_path, capsys):
    """A checkpoint whose probabilities differ from the logged run's gets a warning."""
    examples = pd.DataFrame({"id": ["a", "b"], "text": ["Taxes went up.", "Jobs fell."]})
    probs = explainer.shap.model(np.asarray(examples["text"]))
    logged = pd.DataFrame(probs, columns=[f"prob_{x}" for x in explainer.info.labels])
    logged.insert(0, "id", examples["id"])
    explainer.predictions = tmp_path / "test.csv"
    try:
        logged.to_csv(explainer.predictions, index=False)
        assert check_logged_predictions(explainer, examples) == pytest.approx(0, abs=1e-5)
        logged["prob_true"] += 0.5  # as if another run of the same name and seed was logged
        logged.to_csv(explainer.predictions, index=False)
        assert check_logged_predictions(explainer, examples) > 0.05
        assert "WARNING" in capsys.readouterr().out
    finally:
        explainer.predictions = None


def test_word_tokenizer_keeps_every_character():
    text = "Massachusetts lawmakers weigh 'Medicare for all' bill."
    out = word_tokenizer(text)
    assert out["input_ids"][-2:] == ["bill", "."]  # SHAP's default Text() drops the "."
    assert [text[a:b] for a, b in out["offset_mapping"]] == out["input_ids"]


@pytest.fixture(scope="module")
def lstm_run(tmp_path_factory):
    """A tiny untrained Bi-LSTM saved like factcheck.models.lstm does (PUBHEALTH, 4 labels)."""
    import torch

    from factcheck.models.lstm import BiLSTMClassifier

    torch.manual_seed(0)
    vocab = ["<pad>", "<unk>", "lawmakers", "weigh", "'", "medicare", "for", "all", "bill", "."]
    embeddings = torch.randn(len(vocab), 8)
    embeddings[:2] = 0
    model = BiLSTMClassifier(embeddings, 4, hidden_size=8, num_layers=1, dropout=0.0)
    path = tmp_path_factory.mktemp("tiny_lstm") / "seed1"
    (path / "model").mkdir(parents=True)
    saved = {
        "state_dict": model.state_dict(),
        "vocab": vocab,
        "labels": list(get_dataset("pubhealth").labels),
    }
    saved.update(max_length=64, hidden_size=8, num_layers=1, dropout=0.0)
    torch.save(saved, path / "model" / "model.pt")
    cfg = {"run": {"name": "tiny_lstm", "seed": 1}, "data": {"dataset": "pubhealth"}}
    (path / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_lstm_values_add_up_to_the_real_prediction(lstm_run):
    """With the default masker the explained text lost its final "." and this was off."""
    explainer = get_shap_explainer(str(lstm_run), device="cpu")
    assert explainer.name == "tiny_lstm_seed1" and explainer.tokenizer is None
    text = "Lawmakers weigh 'Medicare for all' bill."
    result = explain_claim(text, explainer, max_evals=100)
    assert [t["token"].strip() for t in result["token_attributions"]][-1] == "."
    real = explainer.shap.model(np.array([text]))[0]
    np.testing.assert_allclose(result["probs"], real, atol=1e-5)
