"""Table of the T55 recurrent-cell runs: scores next to the gradient sizes they logged (T55,
owner M2).

    python scripts/t55_gradients.py

For every T55 run (plain RNN with and without clipping, GRU, one-direction LSTM, Bi-LSTM) on
COVID, LIAR and PUBHEALTH, averaged over its seeds: test macro-F1 (mean and standard deviation,
from results/runs.csv), the total gradient norm before clipping over all training steps (mean
and max), the share of steps that clipping cut, and the gradient at the input words by distance
from the last word (``grad_by_distance`` in each run's train_summary.json), before training and
after the last epoch. ``d16 / d0`` is the gradient 16 words before the end as a share of the
gradient at the last word: near 0 means the early words get almost no learning signal
(vanishing gradients). Written to results/t55_gradients.csv.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from factcheck.paths import OUTPUT_DIR, RESULTS_DIR, RUNS_CSV

DATASETS = ("covid", "liar", "pubhealth")
MODELS = {  # run name prefix -> label
    "rnn_noclip": "plain RNN, no clipping",
    "rnn": "plain RNN",
    "gru": "GRU",
    "unilstm": "LSTM",
    "lstm": "Bi-LSTM",
}


def ratio(grads: dict[str, float], far: str = "16") -> float:
    return grads[far] / grads["0"] if far in grads and grads["0"] else np.nan


def summarise() -> pd.DataFrame:
    runs = pd.read_csv(RUNS_CSV)
    test = runs[(runs["split"] == "test") & runs["subset"].isna()]
    rows = []
    for dataset in DATASETS:
        for prefix, label in MODELS.items():
            name = f"{prefix}_{dataset}"
            summaries = [
                json.loads(path.read_text())
                for path in sorted((OUTPUT_DIR / name).glob("seed*/train_summary.json"))
            ]
            if not summaries:
                continue
            f1 = test.loc[test["run_name"] == name, "macro_f1"]
            norms = [
                (e["grad_norm_mean"], e["grad_norm_max"], e["clipped_fraction"])
                for s in summaries
                for e in s["history"]
            ]
            mean_norm, max_norm, clipped = np.asarray(norms).T
            rows.append(
                {
                    "dataset": dataset,
                    "model": label,
                    "run": name,
                    "seeds": len(f1),
                    "test_macro_f1": f1.mean(),
                    "test_macro_f1_sd": f1.std(),
                    "epochs_run": np.mean([s["epochs_run"] for s in summaries]),
                    "trainable_params": summaries[0]["trainable_params"],
                    "grad_norm_mean": mean_norm.mean(),
                    "grad_norm_max": max_norm.max(),
                    "clipped_fraction": clipped.mean(),
                    "d16_over_d0_init": np.nanmean(
                        [ratio(s["grad_by_distance_init"]) for s in summaries]
                    ),
                    "d16_over_d0_trained": np.nanmean(
                        [ratio(s["history"][-1]["grad_by_distance"]) for s in summaries]
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    table = summarise()
    table.to_csv(RESULTS_DIR / "t55_gradients.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(table.drop(columns="run").round(4).to_string(index=False))
    print(f"\nWritten to {RESULTS_DIR / 't55_gradients.csv'}")


if __name__ == "__main__":
    main()
