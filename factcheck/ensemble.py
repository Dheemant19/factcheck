"""Top-3 ensembles and significance tests from the saved predictions (owner M3, T31).

    python -m factcheck.ensemble                         # every dataset
    python -m factcheck.ensemble --datasets liar pubhealth
    python -m factcheck.ensemble --temperature           # temperature-scale members first
    python -m factcheck.ensemble --skip-ensemble         # tests only; runs.csv untouched

It reads ``results/predictions/<run>/seed<N>/{validation,test}.csv``, which ``log_run`` writes
for every run. That folder is not in git, so run this where the runs were trained (the
cluster) or after copying the folder. For each dataset:

1. Every model's seeds are averaged into one probability matrix per split.
2. The top 3 models by VALIDATION macro-F1 are picked; the test set never chooses. The best
   comes first; each further slot takes the best remaining model, unless one with a backbone
   not yet in the ensemble is within ``DIVERSITY_MARGIN`` of it, because ensembles gain most
   from models that make different errors.
3. Soft vote: the three matrices are averaged (with ``--temperature``, each is first
   temperature-scaled on validation, since LLM label probabilities can be over-confident).
4. The ensemble is logged to ``results/runs.csv`` as ``ensemble_top3_<dataset>`` (family
   ``ensemble``, seed 0, the members in ``model_name``), with its predictions saved like any run.
5. Planned comparisons (``COMPARISONS``) get a paired bootstrap and a McNemar test on test,
   Holm-corrected over all comparisons, in ``results/significance.csv``.

``results/ensemble_candidates.csv`` lists every candidate with its validation and test
macro-F1 and which were picked.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from factcheck.data import get_dataset, load_splits
from factcheck.data.registry import DatasetInfo
from factcheck.paths import PREDICTIONS_DIR, RESULTS_DIR
from factcheck.results import load_runs
from factcheck.stats import (
    fit_temperature,
    holm,
    macro_f1,
    mcnemar_p,
    paired_bootstrap,
    soft_vote,
    temperature_scale,
)

DATASETS = ["covid", "liar", "pubhealth", "isot"]
MAIN_VARIANT = {"isot": "clean"}  # the variant results are reported on (others: raw)
N_MEMBERS = 3
DIVERSITY_MARGIN = 0.01  # validation macro-F1 a different backbone may trail by and still win
SIGNIFICANCE_CSV = RESULTS_DIR / "significance.csv"
CANDIDATES_CSV = RESULTS_DIR / "ensemble_candidates.csv"
SPLITS = ("validation", "test")


@dataclass
class Model:
    """One run, its seeds averaged: probabilities per split in the dataset's row order."""

    run_name: str
    family: str
    backbone: str
    method: str
    seeds: list[int]
    probs: dict[str, np.ndarray]  # split -> (rows, classes)

    def pred(self, split: str) -> np.ndarray:
        return self.probs[split].argmax(axis=1)


def read_predictions(path: Path, info: DatasetInfo) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"id": str})
    missing = [f"prob_{label}" for label in info.labels if f"prob_{label}" not in df]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return df


def load_model(row: pd.Series, info: DatasetInfo, order: dict[str, pd.Index]) -> Model | None:
    """Average a run's seeds; None (with the reason printed) if it cannot be used."""
    probs = {}
    for split in SPLITS:
        per_seed = []
        for seed in row["seeds"]:
            path = PREDICTIONS_DIR / row["run_name"] / f"seed{seed}" / f"{split}.csv"
            if not path.exists():
                print(f"  skip {row['run_name']}: no {path.relative_to(PREDICTIONS_DIR)}")
                return None
            df = read_predictions(path, info).set_index("id")
            if set(df.index) != set(order[split]):
                print(f"  skip {row['run_name']}: its {split} rows are not the full {split} split")
                return None
            per_seed.append(df.loc[order[split], [f"prob_{c}" for c in info.labels]].to_numpy())
        probs[split] = soft_vote(per_seed)
    return Model(
        run_name=row["run_name"],
        family=row["family"],
        backbone=str(row["model_name"]),
        method=str(row["method"]),
        seeds=list(row["seeds"]),
        probs=probs,
    )


def candidate_runs(runs: pd.DataFrame, name: str) -> pd.DataFrame:
    """Runs of one dataset with validation and test scores for the same seeds."""
    variant = MAIN_VARIANT.get(name, "raw")
    keep = (
        (runs["dataset"] == name)
        & (runs["variant"] == variant)
        & runs["subset"].isna()
        & (runs["family"] != "ensemble")
        & runs["split"].isin(SPLITS)
    )
    rows = []
    for run_name, group in runs[keep].groupby("run_name"):
        seeds = set(group.loc[group["split"] == "validation", "seed"])
        seeds &= set(group.loc[group["split"] == "test", "seed"])
        if seeds:
            first = group.iloc[0]
            rows.append({"run_name": run_name, "family": first["family"],
                         "model_name": first["model_name"], "method": first["method"],
                         "seeds": sorted(int(s) for s in seeds)})  # fmt: skip
    return pd.DataFrame(rows)


def load_models(runs: pd.DataFrame, name: str) -> tuple[dict[str, Model], dict]:
    """Every usable model of a dataset, and its row order and labels per split."""
    info = get_dataset(name)
    splits = load_splits(name, MAIN_VARIANT.get(name, "raw"))
    order = {s: pd.Index(splits[s]["id"]) for s in SPLITS}
    labels = {s: splits[s]["label"].to_numpy() for s in SPLITS}
    models = {}
    for _, row in candidate_runs(runs, name).iterrows():
        model = load_model(row, info, order)
        if model is not None:
            models[model.run_name] = model
    return models, {"info": info, "splits": splits, "labels": labels}


def val_f1(model: Model, labels: dict, k: int) -> float:
    return macro_f1(labels["validation"], model.pred("validation"), k)


def select_members(
    models: dict[str, Model], labels: dict, k: int, n: int = N_MEMBERS
) -> list[Model]:
    """Top ``n`` by validation macro-F1, preferring a new backbone within the margin."""
    ranked = sorted(models.values(), key=lambda m: val_f1(m, labels, k), reverse=True)
    chosen: list[Model] = []
    while ranked and len(chosen) < n:
        best = ranked[0]
        used = {m.backbone for m in chosen}
        new = next((m for m in ranked if m.backbone not in used), None)
        if new is not None and val_f1(new, labels, k) >= val_f1(best, labels, k) - DIVERSITY_MARGIN:
            best = new
        chosen.append(best)
        ranked.remove(best)
    return chosen


def build_ensemble(name: str, members: list[Model], labels: dict, temperature: bool) -> Model:
    """Soft vote of the members; with ``temperature``, each is calibrated on validation first."""
    temps = [fit_temperature(labels["validation"], m.probs["validation"]) if temperature else 1.0
             for m in members]  # fmt: skip
    probs = {
        split: soft_vote(
            [temperature_scale(m.probs[split], t) for m, t in zip(members, temps, strict=True)]
        )
        for split in SPLITS
    }
    return Model(
        run_name=f"ensemble_top3_{name}",
        family="ensemble",
        backbone=" + ".join(m.run_name for m in members),
        method="soft_vote_temperature" if temperature else "soft_vote",
        seeds=[0],
        probs=probs,
    )


# --- planned comparisons -----------------------------------------------------------------


def best_of(models: dict[str, Model], labels: dict, k: int, keep) -> Model | None:
    found = [m for m in models.values() if keep(m)]
    return max(found, key=lambda m: val_f1(m, labels, k)) if found else None


def comparisons(
    name: str,
    models: dict[str, Model],
    labels: dict,
    k: int,
    ensemble: Model | None,
    best_member: Model | None,
) -> list[tuple]:
    """(question, model A, model B) for one dataset; pairs with a missing model are skipped."""
    suffix = "_clean" if MAIN_VARIANT.get(name) == "clean" else ""
    get = models.get
    encoder = best_of(models, labels, k, lambda m: m.family == "encoder")
    llm = best_of(models, labels, k, lambda m: m.family == "llm")
    tuned = get(f"deberta_v3_base_{name}_tuned")
    domain = {"covid": "covid_twitter_bert_covid", "pubhealth": "scibert_pubhealth"}.get(name)
    planned = [
        ("ensemble vs best single model", ensemble, best_member),
        ("best encoder vs BERT-base", encoder, get(f"bert_{name}{suffix}")),
        ("best encoder vs best Qwen3-8B mode", encoder, llm),
        ("Optuna-tuned vs default DeBERTa-v3-base", tuned, get(f"deberta_v3_base_{name}{suffix}")),
        ("+ metadata (tuned DeBERTa-v3-base)", get(f"deberta_v3_base_{name}_tuned_meta"), tuned),
        ("+ evidence (tuned DeBERTa-v3-base)", get(f"deberta_v3_base_{name}_tuned_evidence"), tuned),
        ("+ class weights (tuned DeBERTa-v3-base)", get(f"deberta_v3_base_{name}_tuned_cw"), tuned),
        ("domain model vs tuned DeBERTa-v3-base", get(domain) if domain else None, tuned),
    ]  # fmt: skip
    for backbone in ("roberta_large", "deberta_v3_large"):
        lora = best_of(models, labels, k, lambda m, b=backbone: m.run_name.startswith(f"{b}_lora_"))
        planned.append((f"LoRA vs full fine-tuning ({backbone})", lora, get(f"{backbone}_{name}")))
    return [(q, a, b) for q, a, b in planned if a is not None and b is not None and a is not b]


def test_pair(question: str, a: Model, b: Model, labels: dict, k: int, name: str) -> dict:
    y = labels["test"]
    boot = paired_bootstrap(y, a.pred("test"), b.pred("test"), n_classes=k)
    p_mcnemar, discordant = mcnemar_p(y, a.pred("test"), b.pred("test"))
    return {
        "dataset": name,
        "comparison": question,
        "model_a": a.run_name,
        "model_b": b.run_name,
        "seeds_a": len(a.seeds),
        "seeds_b": len(b.seeds),
        "test_macro_f1_a": round(macro_f1(y, a.pred("test"), k), 4),
        "test_macro_f1_b": round(macro_f1(y, b.pred("test"), k), 4),
        **{key: round(value, 4) for key, value in boot.items()},
        "p_mcnemar": round(p_mcnemar, 4),
        "discordant": discordant,
    }


# --- the job ------------------------------------------------------------------------------


def log_ensemble(ensemble: Model, context: dict, name: str) -> None:
    from factcheck.evaluate import log_run

    cfg = {
        "run": {"name": ensemble.run_name, "seed": 0, "config": "python -m factcheck.ensemble"},
        "data": {"dataset": name, "variant": MAIN_VARIANT.get(name, "raw")},
        "model": {"family": "ensemble", "name": ensemble.backbone, "method": ensemble.method},
    }
    scored = {s: (context["splits"][s], ensemble.probs[s]) for s in SPLITS}
    log_run(cfg, context["info"], scored, extra={"trainable_params": 0})


def run_dataset(
    name: str, runs: pd.DataFrame, temperature: bool, with_ensemble: bool = True
) -> tuple[list[dict], list]:
    """Build, log and compare one dataset's ensemble; returns candidate rows and comparisons.

    ``with_ensemble=False`` runs the significance tests only and leaves runs.csv untouched
    (use it after the results freeze, T43).
    """
    print(f"\n== {name}")
    models, context = load_models(runs, name)
    labels, k = context["labels"], context["info"].num_labels
    if len(models) < 2:
        print(f"  only {len(models)} model(s) with saved predictions; nothing to compare")
        return [], []
    members, ensemble = [], None
    if with_ensemble and len(models) >= N_MEMBERS:
        members = select_members(models, labels, k)
        ensemble = build_ensemble(name, members, labels, temperature)
        print("  members (by validation macro-F1):")
        for m in members:
            print(f"    {m.run_name:<45} val {val_f1(m, labels, k):.4f}  ({len(m.seeds)} seeds)")
        log_ensemble(ensemble, context, name)

    candidates = [
        {"dataset": name, "run_name": m.run_name, "family": m.family, "backbone": m.backbone,
         "seeds": len(m.seeds), "val_macro_f1": round(val_f1(m, labels, k), 4),
         "test_macro_f1": round(macro_f1(labels["test"], m.pred("test"), k), 4),
         "member_rank": next((i + 1 for i, c in enumerate(members) if c is m), None)}
        for m in models.values()
    ]  # fmt: skip
    best_member = members[0] if members else None
    pairs = comparisons(name, models, labels, k, ensemble, best_member=best_member)
    return candidates, [test_pair(q, a, b, labels, k, name) for q, a, b in pairs]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--temperature", action="store_true", help="temperature-scale members")
    parser.add_argument(
        "--skip-ensemble",
        action="store_true",
        help="significance tests only; runs.csv is not touched (use after the T43 freeze)",
    )
    args = parser.parse_args(argv)

    runs = load_runs()
    candidates, tests = [], []
    for name in args.datasets:
        c, t = run_dataset(name, runs, args.temperature, with_ensemble=not args.skip_ensemble)
        candidates += c
        tests += t
    if not tests:
        raise SystemExit("No comparisons: are the runs' predictions in results/predictions/?")

    table = pd.DataFrame(tests)
    table["p_bootstrap_holm"] = holm(table["p_bootstrap"]).round(4)
    table["p_mcnemar_holm"] = holm(table["p_mcnemar"]).round(4)
    # significant: the macro-F1 difference survives Holm over all comparisons of this run
    table["significant"] = table["p_bootstrap_holm"] < 0.05
    table.to_csv(SIGNIFICANCE_CSV, index=False)
    pd.DataFrame(candidates).sort_values(
        ["dataset", "val_macro_f1"], ascending=[True, False]
    ).to_csv(CANDIDATES_CSV, index=False)
    pd.set_option("display.width", 200)
    print("\n" + table[["dataset", "comparison", "model_a", "model_b", "diff", "ci_low",
                        "ci_high", "p_bootstrap_holm", "significant"]].to_string(index=False))  # fmt: skip
    print(f"\nwrote {SIGNIFICANCE_CSV} and {CANDIDATES_CSV}")


if __name__ == "__main__":
    main()
