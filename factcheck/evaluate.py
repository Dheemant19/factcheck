"""The ONE metrics function, also checks the specs S1-S10 (T04; M1, agreed with M4).

Every run is scored by ``compute_metrics`` and logged by ``log_run``, whatever the model
family, so every number in the report comes from the same code:

    from factcheck.evaluate import log_run
    log_run(cfg, info, {"validation": (val_df, val_probs), "test": (test_df, test_probs)},
            extra={"trainable_params": ..., "train_seconds": ...})

``log_run`` appends one row per split to ``results/runs.csv`` (see ``factcheck.results``) and
saves the per-row probabilities to ``results/predictions/<run>/seed<N>/<split>.csv``, which the
ensemble and significance tests (T31) and the calibration curves need.

The app's Results tab calls ``check_specs(runs)``, which returns one row per spec with columns
``id, spec, target, value, status, source`` (status: "PASS", "FAIL", "no runs yet" or
"check by hand"). From the command line:

    python -m factcheck.evaluate          # test-split runs, then the S1-S10 pass/fail table
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from factcheck.data.registry import DatasetInfo
from factcheck.paths import CONFIG_DIR, PREDICTIONS_DIR, PROJECT_ROOT
from factcheck.results import append_runs, load_runs

ECE_BINS = 15
SPECS_FILE = CONFIG_DIR / "specs.yaml"

# ---------------------------------------------------------------------------------------------
# Metrics


def expected_calibration_error(labels, probs, n_bins: int = ECE_BINS) -> float:
    """Top-label ECE: the gap between confidence and accuracy, averaged over equal-width bins."""
    labels, probs = np.asarray(labels), np.asarray(probs, dtype=float)
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    bins = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        in_bin = bins == b
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)


def compute_metrics(labels, probs, info: DatasetInfo) -> dict:
    """Every metric of plan Section 11 for one split.

    ``labels`` are class ids, ``probs`` is (rows, num_labels). The "fake" metrics use the
    dataset's binary mapping (``info.fake_labels`` vs ``info.real_labels``): P(fake) is the
    summed probability of the fake classes, a row is flagged when its top class is a fake
    class, and rows whose label is in neither group (e.g. PUBHEALTH "mixture") are left out.
    """
    labels, probs = np.asarray(labels), np.asarray(probs, dtype=float)
    if probs.shape != (len(labels), info.num_labels):
        raise ValueError(
            f"probs has shape {probs.shape}, expected ({len(labels)}, {info.num_labels})"
        )
    pred = probs.argmax(axis=1)
    classes = list(range(info.num_labels))
    metrics = {
        "n": len(labels),
        "macro_f1": f1_score(labels, pred, labels=classes, average="macro", zero_division=0),
        "accuracy": accuracy_score(labels, pred),
        "weighted_f1": f1_score(labels, pred, labels=classes, average="weighted", zero_division=0),
        "ece": expected_calibration_error(labels, probs),
        "confusion_matrix": json.dumps(confusion_matrix(labels, pred, labels=classes).tolist()),
    }

    fake_ids = info.fake_label_ids
    real_ids = [i for i, label in enumerate(info.labels) if label in info.real_labels]
    keep = np.isin(labels, fake_ids + real_ids)
    y_fake = np.isin(labels[keep], fake_ids).astype(int)
    flagged = np.isin(pred[keep], fake_ids).astype(int)
    p_fake = probs[keep][:, fake_ids].sum(axis=1)
    both_classes = len(np.unique(y_fake)) == 2
    metrics |= {
        "n_binary": int(keep.sum()),
        "precision_fake": precision_score(y_fake, flagged, zero_division=0),
        "recall_fake": recall_score(y_fake, flagged, zero_division=0),
        "roc_auc_fake": roc_auc_score(y_fake, p_fake) if both_classes else np.nan,
        "pr_auc_fake": average_precision_score(y_fake, p_fake) if both_classes else np.nan,
    }
    return {k: float(v) if isinstance(v, np.floating) else v for k, v in metrics.items()}


def measure_latency_ms(
    predict: Callable[[list[str]], object], texts: Sequence[str], warmup: int = 5
) -> float:
    """Median milliseconds to score one claim at a time (spec S6), after a few warm-up calls."""
    for text in texts[:warmup]:
        predict([text])
    times = []
    for text in texts:
        start = time.perf_counter()
        predict([text])  # returns numpy, so GPU work has finished when it returns
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


# ---------------------------------------------------------------------------------------------
# Logging


def git_commit() -> str:
    """Short commit hash, with "-dirty" if tracked files have uncommitted changes (spec S10).

    runs.csv itself is left out of the check: every run writes to it, so it is always changed.
    """
    try:
        run = lambda *args: subprocess.run(  # noqa: E731
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        commit = run("rev-parse", "--short", "HEAD")
        return commit + (
            "-dirty"
            if run("status", "--porcelain", "--untracked-files=no", "--", ".", ":!results/runs.csv")
            else ""
        )
    except (OSError, subprocess.CalledProcessError):
        return ""


def save_predictions(df: pd.DataFrame, probs: np.ndarray, info: DatasetInfo, path: Path) -> None:
    """Per-row probabilities: columns id, label, pred, prob_<label name>..."""
    out = pd.DataFrame({"id": df["id"], "label": df["label"], "pred": probs.argmax(axis=1)})
    for i, name in enumerate(info.labels):
        out[f"prob_{name}"] = probs[:, i]
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def log_run(
    cfg: dict,
    info: DatasetInfo,
    scored: dict[str, tuple[pd.DataFrame, np.ndarray]],
    extra: dict | None = None,
) -> list[dict]:
    """Score each split, save its predictions and log it to runs.csv; returns the rows.

    ``scored`` maps a split name to (its DataFrame in the standard schema, probabilities).
    ``extra`` holds run-level numbers such as trainable_params, train_seconds, peak_gpu_mem_mb,
    latency_ms and best_epoch. For ISOT, the test split is also scored on the fixed 2,000-article
    LLM subset, logged as split ``test_llm_subset``.
    """
    run, data, model = cfg["run"], cfg["data"], cfg["model"]
    seed = int(run["seed"])
    base = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_name": run["name"],
        "seed": seed,
        "dataset": info.name,
        "variant": data.get("variant", "raw"),
        "family": model["family"],
        "model_name": model.get("name"),
        "method": model.get("method"),
        "subset": data.get("subset"),
        "git_commit": git_commit(),
        "config": run.get("config"),
        **(extra or {}),
    }
    pred_dir = PREDICTIONS_DIR / run["name"] / f"seed{seed}"
    rows = []
    for split, (df, probs) in scored.items():
        probs = np.asarray(probs, dtype=float)
        save_predictions(df, probs, info, pred_dir / f"{split}.csv")
        rows.append(base | {"split": split} | compute_metrics(df["label"], probs, info))
        if split == "test" and df["in_llm_subset"].any():
            mask = df["in_llm_subset"].to_numpy(dtype=bool)
            metrics = compute_metrics(df["label"][mask], probs[mask], info)
            rows.append(base | {"split": "test_llm_subset"} | metrics)
    append_runs(rows)
    for row in rows:
        print(
            f"  {row['split']:<16} macro-F1 {row['macro_f1']:.4f}  accuracy {row['accuracy']:.4f}"
            f"  weighted F1 {row['weighted_f1']:.4f}  ECE {row['ece']:.4f}"
        )
    return rows


# ---------------------------------------------------------------------------------------------
# Specs S1-S10


def load_specs(path: Path = SPECS_FILE) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_target(target: str) -> tuple[str, float]:
    """ ">= 0.95" -> (">=", 0.95); "<= 50 ms" -> ("<=", 50.0)."""
    match = re.match(r"\s*(>=|<=)\s*([\d.]+)", target)
    if not match:
        raise ValueError(f"spec target {target!r} must start with >= or <= and a number")
    return match.group(1), float(match.group(2))


VALUE_FORMATS = {"latency_ms": "{:.1f} ms", "train_hours": "{:.2f} h"}


def check_metric(test: pd.DataFrame, spec: dict) -> tuple[str, str, str]:
    """(value, status, source) for one metric spec."""
    check = spec["check"]
    metric = check["metric"]
    op, threshold = parse_target(spec["target"])
    higher = op == ">="
    df = test
    for column in ("dataset", "variant", "family"):
        if column in check:
            df = df[df[column] == check[column]]
    df = df.assign(value=pd.to_numeric(df[metric], errors="coerce")).dropna(subset=["value"])
    if df.empty:
        return "", "no runs yet", ""

    if check.get("over") == "all":  # every run must pass: report the worst single run
        per_config = df.groupby("run_name")["value"].agg("min" if higher else "max")
        name = per_config.idxmin() if higher else per_config.idxmax()
        value, source = per_config[name], f"worst: {name}"
    else:  # best config, averaged over its seeds
        per_config = df.groupby("run_name")["value"].agg(["mean", "count"])
        name = per_config["mean"].idxmax() if higher else per_config["mean"].idxmin()
        value, seeds = per_config.loc[name, "mean"], int(per_config.loc[name, "count"])
        source = f"{name} ({seeds} seed{'s' if seeds > 1 else ''})"
    passed = value >= threshold if higher else value <= threshold
    return VALUE_FORMATS.get(metric, "{:.3f}").format(value), "PASS" if passed else "FAIL", source


def check_specs(runs: pd.DataFrame, specs: list[dict] | None = None) -> pd.DataFrame:
    """One row per spec: id, spec, target, value, status, source.

    Only final runs count: the test split, without debug runs on data subsets.
    """
    specs = specs if specs is not None else load_specs()
    test = runs[(runs["split"] == "test") & runs["subset"].isna()]
    test = test.assign(train_hours=pd.to_numeric(test["train_seconds"], errors="coerce") / 3600)
    rows = []
    for spec in specs:
        check = spec["check"]
        if check == "hand":
            value, status, source = "", "check by hand", ""
        elif check["metric"] == "families":
            families = sorted(test["family"].dropna().unique())
            _, needed = parse_target(spec["target"])
            value, source = str(len(families)), ", ".join(families)
            status = (
                "no runs yet" if not families else "PASS" if len(families) >= needed else "FAIL"
            )
        else:
            value, status, source = check_metric(test, spec)
        rows.append(
            {
                "id": spec["id"],
                "spec": spec["spec"],
                "target": spec["target"],
                "value": value,
                "status": status,
                "source": source,
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Print logged test runs and the S1-S10 table.")
    parser.parse_args(argv)
    runs = load_runs()
    test = runs[runs["split"] == "test"]
    if test.empty:
        print("No runs logged yet in results/runs.csv.")
    else:
        columns = ["run_name", "seed", "dataset", "variant", "macro_f1", "accuracy", "weighted_f1"]
        print(test[columns].to_string(index=False, float_format="{:.3f}".format))
    print()
    print(check_specs(runs).to_string(index=False))


if __name__ == "__main__":
    main()
