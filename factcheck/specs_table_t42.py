"""Task T42 Dynamic Figure & Specs Generator for EE6405 Project Report.

Dynamically parses evaluation logs from results/runs.csv to generate:
1. E1 Model Ladder Chart across architectures & datasets (results/figures/t42_model_ladder_dynamic.png).
2. E7 ISOT Shortcut Study Comparison (results/figures/t42_shortcut_analysis_dynamic.png).
3. T40/T42 Specifications Context Table (results/t42_specs_context_table.csv).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from factcheck.paths import FIGURES_DIR, RESULTS_DIR

sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams["font.sans-serif"] = "DejaVu Sans"


def load_runs_data() -> pd.DataFrame:
    """Loads evaluation runs dynamically from results/runs.csv.

    Raises FileNotFoundError if runs.csv is missing to strictly prevent hardcoded fallback numbers.
    """
    candidate_paths = [
        RESULTS_DIR / "final" / "runs_final.csv",  # the frozen results (T43)
        RESULTS_DIR / "runs.csv",
        Path("results/runs.csv"),
        Path("runs.csv"),
        Path("../results/runs.csv"),
    ]

    runs_file = next((p for p in candidate_paths if p.exists()), None)
    if not runs_file:
        raise FileNotFoundError(
            "Required results/runs.csv file not found. "
            "Please run evaluation benchmarks first to generate run outputs."
        )

    print(f"  [Data Source] Reading logged benchmark metrics from: {runs_file}", flush=True)
    return pd.read_csv(runs_file)


MAIN_VARIANT = {"isot": "clean"}  # the variant results are reported on (others: raw)


def run_scores(df_runs: pd.DataFrame, metric: str = "macro_f1") -> pd.DataFrame:
    """Test score per run: mean and standard deviation over its seeds (final runs only)."""
    test = df_runs[(df_runs["split"] == "test") & df_runs["subset"].isna()]
    return (
        test.groupby(["run_name", "dataset", "variant", "family"])[metric]
        .agg(mean="mean", std="std", seeds="count")
        .reset_index()
        .fillna({"std": 0.0})
    )


def generate_model_ladder_metrics(df_runs: pd.DataFrame) -> pd.DataFrame:
    """Test macro-F1 (mean over seeds, error bar = seed std) of the ladder's rungs per dataset."""
    scores = run_scores(df_runs)
    ladder_rows = []
    for ds in ["covid", "liar", "pubhealth", "isot"]:
        suffix = "_clean" if MAIN_VARIANT.get(ds) == "clean" else ""
        main = scores[
            (scores["dataset"] == ds) & (scores["variant"] == MAIN_VARIANT.get(ds, "raw"))
        ]
        rungs = {
            "1 TF-IDF + LR": f"tfidf_lr_{ds}{suffix}",
            "2 Bi-LSTM": f"lstm_{ds}",
            "3 DeBERTa-v3-base": f"deberta_v3_base_{ds}{suffix}",
        }
        for model, run in rungs.items():
            row = main[main["run_name"] == run]
            if not row.empty:  # no Bi-LSTM was trained on ISOT
                ladder_rows.append({"model": model, "dataset": ds.upper(),
                                    "macro_f1": float(row["mean"].iloc[0]),
                                    "std": float(row["std"].iloc[0])})  # fmt: skip
        best = main.loc[main["mean"].idxmax()]
        ladder_rows.append({"model": "4 Best run (any model)", "dataset": ds.upper(),
                            "macro_f1": float(best["mean"]), "std": float(best["std"])})  # fmt: skip
        print(f"  {ds}: best run {best['run_name']} ({best['mean']:.3f}, {best['seeds']} seeds)")
    return pd.DataFrame(ladder_rows)


def plot_dynamic_model_ladder(df_runs: pd.DataFrame) -> None:
    """Figure 1 (E1): Macro-F1 across model families with dynamic seed error bars."""
    df_metrics = generate_model_ladder_metrics(df_runs)
    if df_metrics.empty:
        raise ValueError("Could not assemble model ladder metrics from runs.csv")

    pivot_mean = df_metrics.pivot(index="dataset", columns="model", values="macro_f1")
    pivot_std = df_metrics.pivot(index="dataset", columns="model", values="std")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#95a5a6", "#e67e22", "#3498db", "#2ecc71"]

    pivot_mean.plot(
        kind="bar",
        yerr=pivot_std.fillna(0.0),
        ax=ax,
        color=colors[: len(pivot_mean.columns)],
        width=0.75,
        capsize=3,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_title(
        "E1 Main Comparison: Test Macro-F1 (Mean Over Seeds) Across the Ladder",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_ylabel("Macro-F1 Score", fontsize=11)
    ax.set_xlabel("Dataset Domain", fontsize=11)
    ax.set_ylim(0.0, 1.05)
    ax.legend(title="Model Architecture", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    plt.xticks(rotation=0)
    plt.tight_layout()

    out_path = FIGURES_DIR / "t42_model_ladder_dynamic.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [Saved] Model ladder chart -> {out_path}", flush=True)


def paired_raw_clean(df_runs: pd.DataFrame) -> pd.DataFrame:
    """Each run trained on cleaned text next to the same config on raw text (test, seeds averaged).

    Pairs by run name (``x_isot_clean`` with ``x_isot_raw``, ``x_covid_clean`` with ``x_covid``),
    so a model is only ever compared with itself.
    """
    acc, f1 = run_scores(df_runs, "accuracy"), run_scores(df_runs, "macro_f1")
    both = acc.merge(f1, on=["run_name", "dataset", "variant", "family"], suffixes=("_acc", "_f1"))
    by_name = both.set_index("run_name")
    rows = []
    for _, clean in both[both["variant"] == "clean"].iterrows():
        name = clean["run_name"]
        for raw_name in (name.replace("_clean", "_raw"), name.replace("_clean", "")):
            if raw_name == name or raw_name not in by_name.index:
                continue
            raw = by_name.loc[raw_name]
            if raw["variant"] != "raw":
                continue
            rows.append({
                "model": f"{name.replace('_clean', '')} ({clean['dataset'].upper()})",
                "dataset": clean["dataset"],
                "raw_run": raw_name, "clean_run": name,
                "raw_accuracy": raw["mean_acc"], "cleaned_accuracy": clean["mean_acc"],
                "raw_macro_f1": raw["mean_f1"], "cleaned_macro_f1": clean["mean_f1"],
                "accuracy_drop": raw["mean_acc"] - clean["mean_acc"],
                "macro_f1_drop": raw["mean_f1"] - clean["mean_f1"],
                "seeds_raw": int(raw["seeds_acc"]), "seeds_clean": int(clean["seeds_acc"]),
            })  # fmt: skip
            break
    return pd.DataFrame(rows)


def plot_dynamic_shortcut_study(df_runs: pd.DataFrame) -> None:
    """Figure 4 (E7): Dynamic Raw vs. Cleaned Shortcut Learning Comparison."""
    merged = paired_raw_clean(df_runs)
    if merged.empty:
        raise ValueError(
            "No paired raw vs clean evaluation runs found in runs.csv for shortcut study."
        )

    models = merged["model"].tolist()
    raw_acc = merged["raw_accuracy"].tolist()
    clean_acc = merged["cleaned_accuracy"].tolist()

    x = np.arange(len(models))
    width = 0.35

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    rects1 = ax.bar(
        x - width / 2,
        raw_acc,
        width,
        label="Raw Text (With Shortcuts)",
        color="#e74c3c",
        edgecolor="black",
    )
    rects2 = ax.bar(
        x + width / 2,
        clean_acc,
        width,
        label="Cleaned Text (No Shortcuts)",
        color="#2ecc71",
        edgecolor="black",
    )

    ax.set_title(
        "E7 Shortcut Study: Test Accuracy on Raw vs. Cleaned Text (Mean Over Seeds)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim(0.0, 1.10)
    ax.legend(loc="lower right", frameon=True)

    for rect in rects1 + rects2:
        h = rect.get_height()
        ax.annotate(
            f"{h:.3f}",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    out_path = FIGURES_DIR / "t42_shortcut_analysis_dynamic.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [Saved] Shortcut study chart -> {out_path}", flush=True)


# Specs measured by hand in the app on the laptop (T40, RTX 5060 8 GB, encoder + Qwen3-8B +
# retrieval loaded; contribution log 2026-11-06). runs.csv cannot give these: its latency and
# memory come from test scoring and training on the GPU servers.
HAND_MEASURED = {
    "S6": ("18 ms median (45 ms max)", "PASS", "app on the laptop GPU (T40)"),
    "S7": ("0.16 s median (0.27 s max)", "PASS", "app on the laptop GPU (T40)"),
    "S8": ("5.6 GB", "PASS", "app on the laptop GPU (T40)"),
    "S10": ("one YAML + one command per run", "PASS", "502 rows frozen as v1.0-results (T43)"),
}


def generate_dynamic_specs_table(df_runs: pd.DataFrame) -> pd.DataFrame:
    """Specs S1 to S10: S1-S5 and S9 from the test split by ``check_specs`` (each config's seeds
    averaged, then the best config); S6-S8 and S10 as measured by hand (``HAND_MEASURED``)."""
    from factcheck.evaluate import check_specs

    specs = check_specs(df_runs)
    for spec_id, (value, status, source) in HAND_MEASURED.items():
        specs.loc[specs["id"] == spec_id, ["value", "status", "source"]] = [value, status, source]
    df_specs = specs.rename(columns={"id": "Spec ID", "spec": "Requirement", "target": "Target",
                                     "value": "Achieved", "status": "Status", "source": "Source"})  # fmt: skip
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    specs_csv = RESULTS_DIR / "t42_specs_context_table.csv"
    df_specs.to_csv(specs_csv, index=False)
    print(f"  [Saved] Specifications table exported to: {specs_csv}", flush=True)
    return df_specs


def print_clean_specs_table(df: pd.DataFrame) -> None:
    """Formats the specifications table with fixed field widths to prevent terminal column collisions."""
    headers = ["Spec ID", "Requirement", "Target", "Achieved", "Status", "Source"]
    widths = [8, 80, 24, 32, 8, 50]

    header_str = "".join([f"{h:<{w}}" for h, w in zip(headers, widths, strict=False)])
    print(header_str, flush=True)
    print("-" * sum(widths), flush=True)

    for _, row in df.iterrows():
        print(
            "".join(f"{str(row[h]):<{w}}" for h, w in zip(headers, widths, strict=False)),
            flush=True,
        )


def main() -> None:
    print("==========================================", flush=True)
    print("Executing Dynamic T42 Visualizer & Specs Builder...", flush=True)
    print("==========================================", flush=True)

    df_runs = load_runs_data()
    plot_dynamic_model_ladder(df_runs)
    plot_dynamic_shortcut_study(df_runs)
    df_specs = generate_dynamic_specs_table(df_runs)

    print("\n--- Dynamically Generated Specifications Table ---", flush=True)
    print_clean_specs_table(df_specs)
    print("==========================================", flush=True)
    print("[T42 COMPLETE] Dynamic deliverables generated successfully!", flush=True)


if __name__ == "__main__":
    main()
