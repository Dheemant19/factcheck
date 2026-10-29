"""Tests for the Optuna search (T18) and the encoder settings it varies. The training tests use
a tiny, untrained BERT (downloaded once, about 20 MB) on a made-up 2-label dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

optuna = pytest.importorskip("optuna")
pytest.importorskip("transformers")

from factcheck import tune  # noqa: E402
from factcheck.data import get_dataset  # noqa: E402
from factcheck.models import encoder  # noqa: E402

TINY = "google/bert_uncased_L-2_H-128_A-2"


def test_apply_params_puts_each_setting_where_the_trainer_reads_it():
    base = {"run": {"name": "x"}, "data": {"dataset": "covid"}, "model": {}, "train": {}}
    params = {
        "learning_rate": 3e-5,
        "batch_size": 64,
        "epochs": 3,
        "warmup_ratio": 0.1,
        "weight_decay": 0.05,
        "dropout": 0.2,
        "freeze_layers": 4,
        "max_length": 64,
    }
    cfg = tune.apply_params(base, params)
    assert cfg["train"]["batch_size"] == 32 and cfg["train"]["grad_accum"] == 2
    assert cfg["model"] == {"dropout": 0.2, "freeze_layers": 4}
    assert cfg["data"]["max_length"] == 64 and cfg["train"]["learning_rate"] == 3e-5
    assert base["train"] == {}  # the base config is not changed


def test_max_length_is_searched_only_where_it_cuts_texts():
    for dataset, searched in (("covid", True), ("liar", False), ("pubhealth", False)):
        trial = optuna.create_study().ask()
        assert ("max_length" in tune.sample_params(trial, dataset)) is searched


@pytest.fixture(scope="module")
def tiny_model():
    try:
        return encoder.load_model(TINY, get_dataset("covid"))
    except OSError:
        pytest.skip(f"{TINY} not downloadable here")


def test_dropout_replaces_every_rate(tiny_model):
    model = encoder.load_model(TINY, get_dataset("covid"), dropout=0.0)
    assert model.config.hidden_dropout_prob == 0.0
    assert model.config.attention_probs_dropout_prob == 0.0
    assert tiny_model.config.hidden_dropout_prob == 0.1  # the pretrained rate by default


def test_freeze_bottom_layers(tiny_model):
    model = encoder.load_model(TINY, get_dataset("covid"))
    encoder.freeze_bottom_layers(model, 1)
    bert = model.base_model
    assert not any(p.requires_grad for p in bert.embeddings.parameters())
    assert not any(p.requires_grad for p in bert.encoder.layer[0].parameters())
    assert all(p.requires_grad for p in bert.encoder.layer[1].parameters())
    assert all(p.requires_grad for p in model.classifier.parameters())
    with pytest.raises(ValueError):
        encoder.freeze_bottom_layers(model, 3)  # the tiny BERT has 2 layers


@pytest.fixture
def tiny_covid(tiny_model, tmp_path, monkeypatch):
    """A base config for the tiny BERT on a made-up COVID-like dataset; nothing outside tmp."""
    rng = np.random.default_rng(0)

    def split(n: int) -> pd.DataFrame:
        labels = rng.integers(0, 2, n)
        words = np.where(labels == 1, "fake hoax", "real news")
        return pd.DataFrame(
            {
                "id": [f"r{i}" for i in range(n)],
                "text": [f"claim {w}" for w in words],
                "label": labels,
            }
        )

    splits = {"train": split(32), "validation": split(16), "test": split(16)}
    monkeypatch.setattr(
        encoder, "load_inputs", lambda cfg: {k: v.copy() for k, v in splits.items()}
    )
    monkeypatch.setattr(encoder, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(encoder, "OUTPUT_DIR", tmp_path / "outputs")
    monkeypatch.setattr(tune, "STUDY_DIR", tmp_path / "optuna")
    monkeypatch.setattr(tune, "CONFIG_DIR", tmp_path / "configs")
    monkeypatch.setattr(tune, "MAX_FROZEN", 2)  # the tiny BERT has 2 layers
    (tmp_path / "configs" / "encoder").mkdir(parents=True)
    cfg = {
        "run": {"name": "tiny_covid", "seed": 42},
        "data": {"dataset": "covid", "variant": "raw"},
        "model": {"family": "encoder", "name": TINY},
        "train": {"precision": "fp32"},
    }
    path = tmp_path / "tiny_covid.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_pruned_trial_stops_after_its_first_epoch(tiny_covid):
    base = yaml.safe_load(tiny_covid.read_text(encoding="utf-8"))
    study = optuna.create_study(
        direction="maximize", pruner=optuna.pruners.ThresholdPruner(lower=1.1)
    )  # prunes every trial at its first report
    study.optimize(lambda trial: tune.objective(trial, base), n_trials=1)
    (trial,) = study.trials
    assert trial.state == optuna.trial.TrialState.PRUNED
    assert list(trial.intermediate_values) == [1]


def test_tune_and_export(tiny_covid):
    """Two workers' worth of trials share one study, and the export writes the tuned config."""
    tune.tune(str(tiny_covid), n_trials=2)
    tune.tune(str(tiny_covid), n_trials=3)  # a second call joins the study and adds 1 trial
    study = tune.load_study("tiny_covid")
    assert len(study.trials) == 3
    assert not (tune.STUDY_DIR.parent / "outputs").exists()  # trials save no checkpoint

    paths = tune.export("tiny_covid")
    trials = pd.read_csv(paths["trials"])
    assert len(trials) == 3 and "params_learning_rate" in trials
    importance = yaml.safe_load(paths["importance"].read_text(encoding="utf-8"))
    assert set(importance) in (set(), set(study.best_params))  # empty if all scores tie
    tuned = yaml.safe_load(paths["config"].read_text(encoding="utf-8"))
    assert tuned["run"]["name"] == "tiny_covid_tuned"
    assert tuned["model"]["dropout"] == study.best_params["dropout"]
    assert encoder.with_defaults(tuned)["train"]["learning_rate"] == pytest.approx(
        study.best_params["learning_rate"]
    )
