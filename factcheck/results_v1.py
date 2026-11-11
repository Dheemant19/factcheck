"""Generate Main Results Table v1 and primary figures for Task T21 (owner M4).

Outputs:
  - results/table_v1.md & results/table_v1.tex (Mean ± Std metrics across seeds)
  - outputs/figures/macro_f1_errorbars.png
  - outputs/figures/accuracy_vs_parameters.png
  - outputs/figures/optuna_importance_<dataset>.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    import optuna
    from optuna.visualization.matplotlib import plot_param_importances
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

from factcheck.paths import OUTPUT_DIR

RUNS_CSV = Path("results/runs.csv")
FIGURES_DIR = OUTPUT_DIR / "figures"


def find_metric_columns(df: pd.DataFrame) -> list[str]:
    """Dynamically locate evaluation metric columns present in runs.csv."""
    candidates = [
        "test_macro_f1", "val_macro_f1", "best_val_macro_f1", "macro_f1", 
        "test_f1", "val_f1", "test_accuracy", "val_accuracy", "accuracy", 
        "test_acc", "val_acc"
    ]
    found = [c for c in candidates if c in df.columns]
    if not found:
        # Fallback search for any column containing 'f1', 'acc', or 'score'
        found = [c for c in df.columns if any(k in c.lower() for k in ["f1", "acc", "score"])]
    return found


# -----------------------------------------------------------------------------
# 1. Main Results Table (Mean ± Std)
# -----------------------------------------------------------------------------

def generate_results_table(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Group runs by model family and dataset, returning formatted mean ± std."""
    available_metrics = find_metric_columns(df)
    
    if not available_metrics:
        print(f"[T21] Warning: No metric columns recognized in runs.csv.")
        print(f"      Available columns in file: {list(df.columns)}")
        return pd.DataFrame()

    # Identify model column
    if "family" in df.columns:
        model_col = "family"
    elif "run" in df.columns:
        model_col = "run"
    elif "model" in df.columns:
        model_col = "model"
    else:
        model_col = df.columns[0]
    
    dataset_col = "dataset" if "dataset" in df.columns else df.columns[1]

    grouped = df.groupby([model_col, dataset_col])[available_metrics].agg(["mean", "std"])
    
    formatted = pd.DataFrame(index=grouped.index)
    for metric in available_metrics:
        mean_series = grouped[(metric, "mean")]
        std_series = grouped[(metric, "std")].fillna(0.0)
        formatted[metric] = [
            f"{m:.4f} ± {s:.4f}" for m, s in zip(mean_series, std_series)
        ]
    
    # Pivot for clean multi-dataset view
    table_v1 = formatted.unstack(level=dataset_col)
    
    # Save Markdown and LaTeX versions
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = Path("results/table_v1.md")
    tex_path = Path("results/table_v1.tex")
    
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(table_v1.to_markdown(), encoding="utf-8")
    tex_path.write_text(table_v1.to_latex(), encoding="utf-8")
    
    print(f"[T21] Saved tables to {md_path} and {tex_path}")
    return table_v1


# -----------------------------------------------------------------------------
# 2. Macro-F1 by Model and Dataset with Error Bars
# -----------------------------------------------------------------------------

def plot_macro_f1_errorbars(df: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart of macro-F1 across models and datasets with std error bars."""
    available_metrics = find_metric_columns(df)
    if not available_metrics:
        return
        
    y_metric = available_metrics[0]  # Select primary metric found
    model_col = "family" if "family" in df.columns else ("run" if "run" in df.columns else df.columns[0])
    dataset_col = "dataset" if "dataset" in df.columns else "data"

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    ax = sns.barplot(
        data=df,
        x=model_col,
        y=y_metric,
        hue=dataset_col if dataset_col in df.columns else None,
        errorbar="sd",
        capsize=0.1,
        err_kws={"linewidth": 1.5},
        palette="Set2"
    )
    
    plt.title(f"Model Performance ({y_metric}) by Dataset (Mean ± Std)", fontsize=13, fontweight="bold")
    plt.xlabel("Model Architecture", fontsize=11)
    plt.ylabel(y_metric.replace("_", " ").title(), fontsize=11)
    plt.ylim(0, 1.0)
    if dataset_col in df.columns:
        plt.legend(title="Dataset", frameon=True)
    plt.tight_layout()
    
    fig_path = output_dir / "macro_f1_errorbars.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[T21] Saved macro-F1 error bar plot to {fig_path}")


# -----------------------------------------------------------------------------
# 3. Accuracy / Macro-F1 vs Trainable Parameters
# -----------------------------------------------------------------------------

def plot_accuracy_vs_params(df: pd.DataFrame, output_dir: Path) -> None:
    """Scatter plot comparing performance against trainable parameters (log scale)."""
    if "trainable_params" not in df.columns:
        print("[T21] Note: 'trainable_params' column not found in runs.csv; skipping parameter plot.")
        return

    available_metrics = find_metric_columns(df)
    if not available_metrics:
        return

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    model_col = "family" if "family" in df.columns else ("run" if "run" in df.columns else df.columns[0])
    dataset_col = "dataset" if "dataset" in df.columns else "data"
    y_metric = available_metrics[0]
    
    group_cols = [c for c in [model_col, dataset_col] if c in df.columns]
    avg_df = df.groupby(group_cols)[["trainable_params", y_metric]].mean().reset_index()
    
    sns.scatterplot(
        data=avg_df,
        x="trainable_params",
        y=y_metric,
        hue=model_col,
        style=dataset_col if dataset_col in df.columns else None,
        s=150,
        palette="tab10"
    )
    
    plt.xscale("log")
    plt.title("Performance vs. Trainable Parameters", fontsize=13, fontweight="bold")
    plt.xlabel("Trainable Parameters (Log Scale)", fontsize=11)
    plt.ylabel(y_metric.replace("_", " ").title(), fontsize=11)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=True)
    plt.tight_layout()
    
    fig_path = output_dir / "accuracy_vs_parameters.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[T21] Saved accuracy vs parameters plot to {fig_path}")


# -----------------------------------------------------------------------------
# 4. Optuna Importance Plots
# -----------------------------------------------------------------------------

def plot_optuna_importances(optuna_db_dir: Path, output_dir: Path) -> None:
    """Generate hyperparameter importance plots from Optuna studies."""
    if not HAS_OPTUNA:
        print("[T21] Optuna package not available; skipping Optuna importance plots.")
        return

    db_files = list(optuna_db_dir.glob("*.db")) if optuna_db_dir.exists() else []
    if not db_files:
        print(f"[T21] Note: No Optuna database files found in {optuna_db_dir}.")
        return

    for db_file in db_files:
        storage_url = f"sqlite:///{db_file.resolve()}"
        try:
            summaries = optuna.study.get_all_study_summaries(storage=storage_url)
            for summary in summaries:
                study = optuna.load_study(study_name=summary.study_name, storage=storage_url)
                
                ax = plot_param_importances(study)
                ax.set_title(f"Hyperparameter Importance: {summary.study_name}")
                plt.tight_layout()
                
                fig_path = output_dir / f"optuna_importance_{summary.study_name}.png"
                plt.savefig(fig_path, dpi=300)
                plt.close()
                print(f"[T21] Saved Optuna importance plot to {fig_path}")
        except Exception as e:
            print(f"[T21] Could not process Optuna file {db_file.name}: {e}")


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Task T21: Main Results Table v1 and Primary Charts")
    parser.add_argument("--runs_csv", type=Path, default=RUNS_CSV, help="Path to results/runs.csv")
    parser.add_argument("--output_dir", type=Path, default=FIGURES_DIR, help="Output directory for figures")
    parser.add_argument("--optuna_dir", type=Path, default=Path("outputs/optuna"), help="Directory with Optuna sqlite .db files")
    args = parser.parse_args()

    if not args.runs_csv.exists():
        raise FileNotFoundError(f"Results CSV not found at '{args.runs_csv}'.")

    df = pd.read_csv(args.runs_csv)
    print(f"[T21] Loaded {len(df)} run entries from {args.runs_csv}")

    generate_results_table(df, args.output_dir)
    plot_macro_f1_errorbars(df, args.output_dir)
    plot_accuracy_vs_params(df, args.output_dir)
    plot_optuna_importances(args.optuna_dir, args.output_dir)


if __name__ == "__main__":
    main()