"""Optuna search on DeBERTa-v3-base, one study per dataset (owners M1 + M2, T18, E2).

    python -m factcheck.tune --config configs/encoder/deberta_v3_base_liar.yaml
    python -m factcheck.tune --config configs/encoder/deberta_v3_base_liar.yaml --export

Plan Section 10: a TPE sampler, and a median pruner that stops a trial after any epoch where
its validation macro-F1 is below the median of the earlier trials at that epoch; about 30
trials per dataset. Several workers, one per GPU (``jobs/t18_tune.sh``), can share one study:
each takes the next trial until the study has ``--trials`` of them. Every trial uses the
config's seed, so trials differ only in their settings. Trials are scored on the validation
split, save no checkpoint and write nothing to results/runs.csv.

The search space (plan Section 9, E2): learning rate, batch size, epochs, warmup, weight decay,
dropout, frozen bottom layers, and the max length where it cuts texts (COVID tweets, ISOT
articles; every LIAR and PUBHEALTH claim is under 64 tokens). Pooling is not searched: each
architecture's head pools its own way, and another pooling would give checkpoints that the app
and explain.py cannot load.

The study is kept in ``results/optuna/<run name>.log``, an Optuna journal file (plain text,
safe for several workers on one machine; it is committed). ``--export`` writes next to it
``<run name>_trials.csv`` and ``<run name>_importance.json`` (fANOVA importances), and the best
trial as a run config, ``configs/encoder/<run name>_tuned.yaml``, for T29 and the E1 table.
The importance and slice plots (T21):

    import optuna.visualization.matplotlib as vis
    from factcheck.tune import load_study
    study = load_study("deberta_v3_base_liar")
    vis.plot_param_importances(study)
    vis.plot_slice(study)
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path

import optuna
import torch
import yaml
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.trial import TrialState
from transformers import TrainerCallback

from factcheck.paths import CONFIG_DIR, RESULTS_DIR
from factcheck.train import load_config

STUDY_DIR = RESULTS_DIR / "optuna"
# Max lengths worth trying: only where the registry's length cuts some texts.
MAX_LENGTHS = {"covid": [64, 128], "isot": [256, 512]}
PER_DEVICE_BATCH = 32  # a batch of 64 runs as 2 x 32 with gradient accumulation
MAX_FROZEN = 6  # of DeBERTa-v3-base's 12 layers


def sample_params(trial: optuna.Trial, dataset: str) -> dict:
    """One trial's settings."""
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 5e-6, 1e-4, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "epochs": trial.suggest_int("epochs", 2, 5),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        "freeze_layers": trial.suggest_int("freeze_layers", 0, MAX_FROZEN, step=2),
    }
    if dataset in MAX_LENGTHS:
        params["max_length"] = trial.suggest_categorical("max_length", MAX_LENGTHS[dataset])
    return params


def apply_params(cfg: dict, params: dict) -> dict:
    """The run config with a trial's settings put where ``train_encoder`` reads them."""
    cfg = copy.deepcopy(cfg)
    data, model, train = (cfg.setdefault(k, {}) for k in ("data", "model", "train"))
    for key in ("learning_rate", "epochs", "warmup_ratio", "weight_decay"):
        train[key] = params[key]
    train["batch_size"] = min(params["batch_size"], PER_DEVICE_BATCH)
    train["grad_accum"] = params["batch_size"] // train["batch_size"]
    model["dropout"] = params["dropout"]
    model["freeze_layers"] = params["freeze_layers"]
    if "max_length" in params:
        data["max_length"] = params["max_length"]
    return cfg


class ReportToOptuna(TrainerCallback):
    """Reports each epoch's validation macro-F1 to the trial; stops the trial if it is pruned."""

    def __init__(self, trial: optuna.Trial):
        self.trial = trial

    def on_evaluate(self, args, state, control, metrics, **kwargs):  # noqa: ANN001
        epoch = round(state.epoch)
        self.trial.report(metrics["eval_macro_f1"], epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned(f"pruned after epoch {epoch}")


def objective(trial: optuna.Trial, base: dict) -> float:
    """Train one trial and return its best validation macro-F1."""
    from factcheck.models.encoder import train_encoder

    params = sample_params(trial, base["data"]["dataset"])
    cfg = apply_params(base, params)
    cfg["run"]["name"] = f"{base['run']['name']}_trial{trial.number}"
    try:
        summary = train_encoder(cfg, callbacks=[ReportToOptuna(trial)], save=False)
    except optuna.TrialPruned:
        summary = None
    # Free the trial's model before the next one starts (the exception kept it alive).
    gc.collect()
    torch.cuda.empty_cache()
    if summary is None:
        raise optuna.TrialPruned()
    for key in ("best_epoch", "epochs_run", "train_seconds", "peak_gpu_mem_mb", "trainable_params"):
        trial.set_user_attr(key, summary[key])
    return summary["best_val_macro_f1"]


def storage(name: str) -> JournalStorage:
    path = STUDY_DIR / f"{name}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    # The open() lock works on Windows too (the default symlink lock needs extra rights there)
    return JournalStorage(JournalFileBackend(str(path), lock_obj=JournalFileOpenLock(str(path))))


def load_study(name: str) -> optuna.Study:
    """A finished study, for plots and the export (``name`` is the base run name)."""
    if not (STUDY_DIR / f"{name}.log").exists():
        raise FileNotFoundError(f"no study {name!r} in {STUDY_DIR}")
    return optuna.load_study(study_name=name, storage=storage(name))


def tune(config: str, n_trials: int = 30) -> optuna.Study:
    """Run trials of one study until it has ``n_trials`` (counting other workers' trials)."""
    base = load_config(config)
    if base["model"]["family"] != "encoder":
        raise ValueError(f"{config}: tune.py searches encoder settings only")
    name = base["run"]["name"]
    study = optuna.create_study(
        study_name=name,
        storage=storage(name),
        direction="maximize",
        # No sampler seed: workers with the same seed would all try the same first settings.
        sampler=optuna.samplers.TPESampler(multivariate=True, constant_liar=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
        load_if_exists=True,
    )
    study.set_user_attr("config", Path(config).as_posix())
    # Running trials count, so parallel workers stop at n_trials in total
    counted = (TrialState.COMPLETE, TrialState.PRUNED, TrialState.RUNNING)
    if len(study.get_trials(deepcopy=False, states=counted)) >= n_trials:
        print(f"{name}: already has {n_trials} trials")
        return study
    study.optimize(
        lambda trial: objective(trial, base),
        callbacks=[optuna.study.MaxTrialsCallback(n_trials, states=counted)],
        catch=(torch.cuda.OutOfMemoryError,),  # a too-large batch fails that trial only
    )
    return study


def export(name: str) -> dict[str, Path]:
    """Trials table, fANOVA importances and the tuned run config of one finished study."""
    study = load_study(name)
    paths = {
        "trials": STUDY_DIR / f"{name}_trials.csv",
        "importance": STUDY_DIR / f"{name}_importance.json",
        "config": CONFIG_DIR / "encoder" / f"{name}_tuned.yaml",
    }
    study.trials_dataframe(
        attrs=("number", "value", "state", "duration", "params", "user_attrs")
    ).to_csv(paths["trials"], index=False)
    try:
        importance = optuna.importance.get_param_importances(
            study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=42)
        )
    except RuntimeError as err:  # every completed trial has the same score
        print(f"  WARNING: no importances ({err})")
        importance = {}
    paths["importance"].write_text(json.dumps(importance, indent=2), encoding="utf-8")

    best = study.best_trial
    config = study.user_attrs["config"]
    tuned = apply_params(load_config(config), best.params)
    tuned["run"]["name"] = f"{name}_tuned"
    finished = len(study.get_trials(states=(TrialState.COMPLETE, TrialState.PRUNED)))
    header = (
        f"# T18 tuned settings for {name}: trial {best.number}, the best of {finished} Optuna\n"
        f"# trials (validation macro-F1 {best.value:.4f}). Written by\n"
        f"# python -m factcheck.tune --config {config} --export\n"
    )
    paths["config"].write_text(header + yaml.safe_dump(tuned, sort_keys=False), encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        required=True,
        help="base run config, e.g. configs/encoder/deberta_v3_base_liar.yaml",
    )
    parser.add_argument("--trials", type=int, default=30, help="trials in the study in total")
    parser.add_argument(
        "--export", action="store_true", help="write the trials, importances and tuned config"
    )
    args = parser.parse_args(argv)

    name = load_config(args.config)["run"]["name"]
    if not args.export:
        tune(args.config, args.trials)
        return
    paths = export(name)
    study = load_study(name)
    print(f"{name}: best validation macro-F1 {study.best_value:.4f} with {study.best_params}")
    for kind, path in paths.items():
        print(f"  {kind}: {path}")


if __name__ == "__main__":
    main()
