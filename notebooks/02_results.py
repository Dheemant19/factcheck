import pandas as pd

from factcheck.evaluate import check_specs
from factcheck.paths import RESULTS_DIR

# Published baselines from literature (Section 12 of Project Plan), all on raw text
PUBLISHED_BASELINES = [
    {"dataset": "covid", "paper": "Patwa et al. (2021)", "metric": "weighted_f1", "published_score": 0.9300},
    {"dataset": "liar", "paper": "Wang (2017)", "metric": "accuracy", "published_score": 0.2740},
    # SciBERT, Table 4 of Kotonya & Toni (2020); 0.65 is our own S4 target, not the paper's figure
    {"dataset": "pubhealth", "paper": "Kotonya & Toni (2020)", "metric": "macro_f1", "published_score": 0.7052},
    {"dataset": "isot", "paper": "Ahmed et al. (2017)", "metric": "accuracy", "published_score": 0.9200},
]  # fmt: skip


def load_runs(csv_path=RESULTS_DIR / "runs.csv") -> pd.DataFrame:
    """Loads logged experiment runs from results/runs.csv."""
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found. Returning empty DataFrame.")
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def generate_baseline_summary(runs: pd.DataFrame) -> pd.DataFrame:
    """Creates a summary comparison table between our models and published baselines."""
    if runs.empty:
        print("No runs logged yet in results/runs.csv.")
        return pd.DataFrame()

    # Papers report full test-set scores on raw text: leave out validation rows, the ISOT
    # LLM-subset rows (split "test_llm_subset") and debug runs on data subsets
    runs = runs[(runs["split"] == "test") & runs["subset"].isna() & (runs["variant"] == "raw")]

    summary_rows = []

    for base in PUBLISHED_BASELINES:
        ds = base["dataset"]
        metric = base["metric"]
        pub_score = base["published_score"]

        # Filter runs for this dataset
        ds_runs = runs[runs["dataset"] == ds]

        if ds_runs.empty:
            summary_rows.append({
                "Dataset": ds.upper(),
                "Published Baseline": f"{base['paper']} ({pub_score:.4f})",
                "Best Our Model": "No runs yet",
                "Our Score": "N/A",
                "Metric": metric,
                "Beats Published?": "Pending",
            })  # fmt: skip
            continue

        # Find best model for this metric
        best_idx = ds_runs[metric].idxmax()
        best_run = ds_runs.loc[best_idx]
        our_score = best_run[metric]
        beats_pub = "YES" if our_score >= pub_score else "NO"

        summary_rows.append({
            "Dataset": ds.upper(),
            "Published Baseline": f"{base['paper']} ({pub_score:.4f})",
            "Best Our Model": f"{best_run['model_name']} ({best_run['family']})",
            "Our Score": f"{our_score:.4f}",
            "Metric": metric,
            "Beats Published?": beats_pub,
        })  # fmt: skip

    return pd.DataFrame(summary_rows)


if __name__ == "__main__":
    print("=" * 80)
    print(" TASK T10: SPRINT 1 BASELINE RESULTS & LITERATURE COMPARISON")
    print("=" * 80)

    runs = load_runs()

    # 1. Display Baseline Summary Table
    summary_df = generate_baseline_summary(runs)
    if not summary_df.empty:
        print("\n--- Model Performance vs. Published Baselines ---")
        print(summary_df.to_string(index=False))

    # 2. Run Specifications Board (S1 - S10)
    print("\n--- Project Specifications Board ---")
    specs_df = check_specs(runs)
    print(specs_df.to_string(index=False))
