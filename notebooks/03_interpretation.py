"""Task 03: Unified Interpretation & Analysis Pipeline (M4 Lead).

Combines:
  - T33: Calibration Diagram, UMAP Representation, Error Tagging, 3-Panel TF-IDF vs SHAP
  - T34: Shortcut Analysis (Raw vs. Cleaned ISOT/COVID using runs.csv and shortcuts/)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from sklearn.calibration import calibration_curve
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Define directories
EXPLANATIONS_DIR = Path("outputs/explanations")
FIGURES_DIR = Path("outputs/figures")
RESULTS_DIR = Path("results")
RUNS_CSV = Path("results/runs.csv")
SHORTCUTS_DIR = Path("results/shortcuts")

SHORTCUT_CUES = ["reuters", "washington", "twitter", "http", "https", "@", "pic.twitter"]

# =====================================================================
# TASK T33 FUNCTIONS
# =====================================================================


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 5) -> float:
    """Computes Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (y_prob >= bin_boundaries[i] if i == 0 else y_prob > bin_boundaries[i]) & (
            y_prob <= bin_boundaries[i + 1]
        )
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy = np.mean(y_true[in_bin])
            avg_confidence = np.mean(y_prob[in_bin])
            ece += np.abs(accuracy - avg_confidence) * prop_in_bin
    return float(ece)


def plot_calibration_curves(npz_files: list[Path], output_dir: Path, n_bins: int = 5) -> None:
    """Generate reliability diagrams with smooth binning and ECE scores."""
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")

    for npz_path in npz_files:
        data = np.load(npz_path, allow_pickle=True)
        probs = data["probs"]
        labels = data["labels"]
        fake_ids = data["fake_label_ids"]

        # Normalize raw logits to probabilities if outside [0, 1]
        if probs.min() < 0.0 or probs.max() > 1.0:
            exp_probs = np.exp(probs - np.max(probs, axis=-1, keepdims=True))
            probs = exp_probs / exp_probs.sum(axis=-1, keepdims=True)

        p_false = probs[:, fake_ids].sum(axis=1) if fake_ids.ndim > 0 else probs[:, 1]
        p_false = np.clip(p_false, 0.0, 1.0)
        y_true = np.isin(labels, fake_ids).astype(int)

        prob_true, prob_pred = calibration_curve(y_true, p_false, n_bins=n_bins)
        ece_score = compute_ece(y_true, p_false, n_bins=n_bins)
        name = npz_path.stem.replace("_shap", "")
        plt.plot(prob_pred, prob_true, marker="o", label=f"{name} (ECE={ece_score:.3f})")

    plt.xlabel("Mean Predicted Probability P(False)", fontsize=11)
    plt.ylabel("Fraction of Positives (True False Claims)", fontsize=11)
    plt.title("Model Calibration Diagrams & ECE Scores (T33)", fontsize=13, fontweight="bold")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    out_file = output_dir / "calibration_charts.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[T33] Saved calibration chart -> {out_file}")


def generate_umap_by_label(npz_files: list[Path], output_dir: Path) -> None:
    """Generates 2D UMAP scatter plots colored by true class label for key models."""
    for npz_path in npz_files:
        data = np.load(npz_path, allow_pickle=True)
        labels = data["labels"]

        if "cls_embeddings" in data:
            vectors = data["cls_embeddings"]
        else:
            values = data["values"]
            vectors = np.array(
                [v.mean(axis=0) if hasattr(v, "mean") else np.zeros(6) for v in values]
            )

        if len(vectors) > 5:
            reducer = umap.UMAP(
                n_neighbors=min(15, len(vectors) - 1),
                min_dist=0.1,
                metric="cosine",
                random_state=42,
            )
            coords = reducer.fit_transform(vectors)
            name = npz_path.stem.replace("_shap", "")

            plt.figure(figsize=(8, 6))
            sns.scatterplot(
                x=coords[:, 0], y=coords[:, 1], hue=labels, palette="tab10", alpha=0.8, s=60
            )
            plt.title(f"UMAP Representation Space: {name}", fontsize=13, fontweight="bold")
            plt.xlabel("UMAP Dimension 1")
            plt.ylabel("UMAP Dimension 2")
            plt.legend(title="Class Label", bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()

            out_file = output_dir / f"umap_{name}_by_label.png"
            plt.savefig(out_file, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"[T33] Saved UMAP plot -> {out_file}")


def export_error_analysis(npz_files: list[Path], output_dir: Path) -> None:
    """Identifies misclassifications and tags physical error categories."""
    error_rows = []

    for npz_path in npz_files:
        data = np.load(npz_path, allow_pickle=True)
        ids, texts, labels = data["ids"], data["texts"], data["labels"]
        probs = data["probs"]
        preds = probs.argmax(axis=1)

        for doc_id, text, true_lbl, pred_lbl, prob_dist in zip(
            ids, texts, labels, preds, probs, strict=False
        ):
            if true_lbl != pred_lbl:
                t_str = str(text)
                upper_ratio = sum(1 for c in t_str if c.isupper()) / max(1, len(t_str))

                if any(cue in t_str.lower() for cue in SHORTCUT_CUES):
                    error_type = "Source Cue Shortcut"
                elif any(kw in t_str for kw in ["!", "URGENT", "BREAKING"]) or upper_ratio > 0.25:
                    error_type = "Stylistic Bias"
                else:
                    error_type = "Content / Semantic Ambiguity"

                error_rows.append(
                    {
                        "model": npz_path.stem.replace("_shap", ""),
                        "doc_id": doc_id,
                        "text": t_str,
                        "true_label": true_lbl,
                        "pred_label": pred_lbl,
                        "confidence": float(prob_dist[pred_lbl]),
                        "error_category": error_type,
                    }
                )

    df_err = pd.DataFrame(error_rows).sort_values(by="confidence", ascending=False)
    out_file = output_dir / "error_analysis_report.csv"
    df_err.to_csv(out_file, index=False)
    print(f"[T33] Exported {len(df_err)} misclassified samples -> {out_file}")


def ensure_tfidf_top_words(
    output_dir: Path,
    dataset_name: str = "isot",
    variant: str = "raw",
    npz_files: list[Path] | None = None,
) -> Path:
    """Loads pre-computed TF-IDF top words from results/shortcuts/."""
    target_csv = output_dir / f"tfidf_top_words_{dataset_name}_{variant}.csv"

    search_paths = [
        SHORTCUTS_DIR,
        Path("results/shortcuts"),
        Path("shortcuts"),
        RESULTS_DIR / "shortcuts",
    ]

    for base_dir in search_paths:
        if not base_dir.exists():
            continue

        f = base_dir / f"top_words_{dataset_name}_{variant}.csv"
        if f.exists():
            print(f"[T33] Loading TF-IDF top words from {f}...")
            df = pd.read_csv(f)
            df.to_csv(target_csv, index=False)
            return target_csv

    if target_csv.exists() and target_csv.stat().st_size > 10:
        return target_csv

    if npz_files:
        print(
            f"[T33 Fallback] Generating temporary TF-IDF top words for {dataset_name}_{variant}..."
        )
        all_texts, all_labels = [], []
        for p in npz_files:
            data = np.load(p, allow_pickle=True)
            all_texts.extend(list(data["texts"]))
            all_labels.extend(list(data["labels"]))

        if all_texts:
            vec = TfidfVectorizer(max_features=5000, stop_words="english")
            X = vec.fit_transform(all_texts)
            clf = LogisticRegression(max_iter=500)
            clf.fit(X, (np.array(all_labels) > 0).astype(int))

            df_words = pd.DataFrame(
                {"word": vec.get_feature_names_out(), "importance_weight": clf.coef_[0]}
            ).sort_values(by="importance_weight", ascending=False)

            df_words.to_csv(target_csv, index=False)
            return target_csv

    return target_csv


def parse_top_words_df(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts words and numerical feature weights from top words CSV files."""
    if df.empty:
        return pd.DataFrame(columns=["word", "weight"])

    df_clean = df.copy()
    df_clean.columns = [str(c).lower().strip() for c in df_clean.columns]

    word_candidates = ["word", "feature", "token", "term", "ngram"]
    word_col = next((c for c in word_candidates if c in df_clean.columns), None)
    if not word_col:
        non_num = df_clean.select_dtypes(include=["object", "string"]).columns
        word_col = non_num[0] if len(non_num) > 0 else df_clean.columns[0]

    weight_candidates = [
        "importance_weight",
        "weight",
        "tfidf",
        "score",
        "importance",
        "coef",
        "coefficient",
    ]
    weight_col = next((c for c in weight_candidates if c in df_clean.columns), None)
    if not weight_col:
        num_cols = [
            c
            for c in df_clean.select_dtypes(include=[np.number]).columns
            if "rank" not in c and "index" not in c and "unnamed" not in c
        ]
        weight_col = (
            num_cols[0]
            if len(num_cols) > 0
            else (df_clean.columns[1] if len(df_clean.columns) > 1 else df_clean.columns[0])
        )

    res = df_clean[[word_col, weight_col]].rename(columns={word_col: "word", weight_col: "weight"})
    res["weight"] = pd.to_numeric(res["weight"], errors="coerce")
    return res.dropna().sort_values(by="weight", ascending=False).head(10)


def plot_shap_vs_tfidf(npz_files: list[Path], output_dir: Path, dataset_name: str = "isot") -> None:
    """Plots a 3-panel horizontal comparison: Raw TF-IDF vs Clean TF-IDF vs Model SHAP attributions."""
    raw_tfidf_csv = ensure_tfidf_top_words(
        output_dir, dataset_name=dataset_name, variant="raw", npz_files=npz_files
    )
    clean_tfidf_csv = ensure_tfidf_top_words(
        output_dir, dataset_name=dataset_name, variant="clean", npz_files=npz_files
    )

    def load_top_10(csv_path: Path) -> pd.DataFrame:
        if not csv_path.exists():
            return pd.DataFrame(columns=["word", "weight"])
        return parse_top_words_df(pd.read_csv(csv_path))

    top_raw_tfidf = load_top_10(raw_tfidf_csv)
    top_clean_tfidf = load_top_10(clean_tfidf_csv)

    # Filter NPZ files specifically for dataset matching
    dataset_npz_files = [p for p in npz_files if dataset_name.lower() in p.stem.lower()]
    if not dataset_npz_files:
        dataset_npz_files = npz_files

    word_shap_scores: dict[str, float] = {}
    for p in dataset_npz_files:
        data = np.load(p, allow_pickle=True)
        if "texts" in data and "values" in data:
            texts, values = data["texts"], data["values"]
            for text, val_matrix in zip(texts, values, strict=False):
                tokens = str(text).split()
                scores = (
                    np.abs(val_matrix).mean(axis=-1) if val_matrix.ndim > 1 else np.abs(val_matrix)
                )
                for tok, sc in zip(tokens, scores, strict=False):
                    tok_clean = tok.lower().strip(".,!?:;\"'")
                    if len(tok_clean) > 2 and tok_clean not in ENGLISH_STOP_WORDS:
                        word_shap_scores[tok_clean] = word_shap_scores.get(tok_clean, 0.0) + float(
                            sc
                        )

    df_shap = (
        pd.DataFrame(list(word_shap_scores.items()), columns=["word", "weight"])
        .sort_values(by="weight", ascending=False)
        .head(10)
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)

    if not top_raw_tfidf.empty:
        sns.barplot(
            data=top_raw_tfidf, x="weight", y="word", ax=axes[0], palette="Reds_r", hue="word"
        )
        if axes[0].legend_:
            axes[0].legend_.remove()
    axes[0].set_title(
        f"1. Raw TF-IDF Top Words ({dataset_name.upper()})", fontsize=11, fontweight="bold"
    )
    axes[0].set_xlabel("TF-IDF / Model Weight")
    axes[0].set_ylabel("Word")

    if not top_clean_tfidf.empty:
        sns.barplot(
            data=top_clean_tfidf, x="weight", y="word", ax=axes[1], palette="Blues_r", hue="word"
        )
        if axes[1].legend_:
            axes[1].legend_.remove()
    axes[1].set_title(
        f"2. Cleaned TF-IDF Top Words ({dataset_name.upper()})", fontsize=11, fontweight="bold"
    )
    axes[1].set_xlabel("TF-IDF / Model Weight")
    axes[1].set_ylabel("Word")

    if not df_shap.empty:
        sns.barplot(data=df_shap, x="weight", y="word", ax=axes[2], palette="Oranges_r", hue="word")
        if axes[2].legend_:
            axes[2].legend_.remove()
    axes[2].set_title(
        f"3. Model SHAP Top Words ({dataset_name.upper()})", fontsize=11, fontweight="bold"
    )
    axes[2].set_xlabel("Mean Absolute SHAP Value")
    axes[2].set_ylabel("Word")

    plt.suptitle(
        f"Interpretation Analysis: Raw vs. Cleaned TF-IDF vs. SHAP Word Importances ({dataset_name.upper()}) (T33)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()

    out_file = output_dir / f"shap_vs_tfidf_raw_clean_{dataset_name}.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[T33] Saved 3-panel comparison plot -> {out_file}")


# =====================================================================
# TASK T34 FUNCTIONS
# =====================================================================


def load_shortcut_scores_from_runs_csv(runs_csv_path: Path = RUNS_CSV) -> pd.DataFrame:
    """Reads raw vs. cleaned macro-F1 and accuracy scores directly from runs.csv."""
    candidate_paths = [
        runs_csv_path,
        RESULTS_DIR / "runs.csv",
        Path("runs.csv"),
        Path("../results/runs.csv"),
    ]

    runs_file = next((p for p in candidate_paths if p.exists()), None)
    if not runs_file:
        print("[T34 Warning] runs.csv not found in candidate search paths.")
        return pd.DataFrame()

    print(f"[T34] Reading benchmark scores from {runs_file}...")
    df_runs = pd.read_csv(runs_file)

    # Test split only, each run's seeds averaged, each cleaned run paired with the same config
    # on raw text (shared with the T42 shortcut chart, so both show the same numbers).
    from factcheck.specs_table_t42 import paired_raw_clean

    return paired_raw_clean(df_runs)


def run_shortcut_analysis(output_dir: Path = FIGURES_DIR) -> pd.DataFrame:
    """Analyzes shortcut reliance using runs.csv score comparisons.

    Falls back to paired .npz explanation files if runs.csv is unavailable.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Primary path: Parse runs.csv scores
    df = load_shortcut_scores_from_runs_csv(RUNS_CSV)

    # 2. Fallback: Parse paired NPZ files if runs.csv is missing
    if df.empty:
        print("[T34] runs.csv scores missing. Falling back to NPZ explanation files...")
        raw_files = [
            f
            for f in list(EXPLANATIONS_DIR.glob("*raw*_shap.npz"))
            + list(EXPLANATIONS_DIR.glob("*seed42_shap.npz"))
            if "_clean" not in f.stem
        ]

        results = []
        for raw_path in raw_files:
            possible_clean_names = [
                EXPLANATIONS_DIR / f"{raw_path.stem}_on_clean_shap.npz",
                EXPLANATIONS_DIR / raw_path.name.replace("raw", "clean"),
                EXPLANATIONS_DIR / f"{raw_path.stem.replace('_shap', '')}_clean_shap.npz",
            ]
            clean_path = next((p for p in possible_clean_names if p.exists()), None)
            if clean_path is None:
                continue

            raw_data = np.load(raw_path, allow_pickle=True)
            clean_data = np.load(clean_path, allow_pickle=True)

            raw_preds = raw_data["probs"].argmax(axis=1)
            clean_preds = clean_data["probs"].argmax(axis=1)
            labels = raw_data["labels"]

            raw_acc = float(accuracy_score(labels, raw_preds))
            clean_acc = float(accuracy_score(labels, clean_preds))
            raw_f1 = float(f1_score(labels, raw_preds, average="macro"))
            clean_f1 = float(f1_score(labels, clean_preds, average="macro"))
            prob_shift = float(np.abs(raw_data["probs"] - clean_data["probs"]).mean())

            model_name = raw_path.stem.replace("_seed42_shap", "").replace("_shap", "")

            results.append(
                {
                    "model": model_name,
                    "raw_accuracy": raw_acc,
                    "cleaned_accuracy": clean_acc,
                    "accuracy_drop": raw_acc - clean_acc,
                    "raw_macro_f1": raw_f1,
                    "cleaned_macro_f1": clean_f1,
                    "macro_f1_drop": raw_f1 - clean_f1,
                    "mean_prob_shift": prob_shift,
                }
            )
        df = pd.DataFrame(results)

    if df.empty:
        print(
            "[T34 Warning] No paired raw vs cleaned results found in runs.csv or outputs/explanations."
        )
        return df

    # Export CSV with Excel file lock handling
    csv_out = RESULTS_DIR / "shortcut_study_t34.csv"
    try:
        df.to_csv(csv_out, index=False)
        print(f"[T34] Saved shortcut analysis CSV -> {csv_out}")
    except PermissionError:
        fallback_csv = RESULTS_DIR / "shortcut_study_t34_updated.csv"
        df.to_csv(fallback_csv, index=False)
        print(
            f"[T34 Warning] {csv_out} locked by another process (e.g. Excel). Saved to {fallback_csv}."
        )

    plot_shortcut_results(df, output_dir)
    return df


def plot_shortcut_results(df: pd.DataFrame, output_dir: Path) -> None:
    """Generates a side-by-side bar chart of Raw vs. Cleaned performance."""
    plot_data = []
    for _, row in df.iterrows():
        plot_data.append(
            {
                "Model": row["model"],
                "Variant": "Raw Text (With Shortcuts)",
                "Macro-F1": row["raw_macro_f1"],
            }
        )
        plot_data.append(
            {
                "Model": row["model"],
                "Variant": "Cleaned Text (No Shortcuts)",
                "Macro-F1": row["cleaned_macro_f1"],
            }
        )

    df_plot = pd.DataFrame(plot_data)

    plt.figure(figsize=(9, 5))
    ax = sns.barplot(
        data=df_plot, x="Model", y="Macro-F1", hue="Variant", palette=["#a3302a", "#2b5fb4"]
    )
    plt.title(
        "Shortcut Study: Model Macro-F1 on Raw vs. Cleaned Text (T34)",
        fontsize=13,
        fontweight="bold",
    )
    plt.ylim(0, 1.05)
    plt.ylabel("Macro-F1 Score")
    plt.xlabel("Model Configuration")
    plt.grid(axis="y", linestyle=":", alpha=0.6)

    for idx, row in df.iterrows():
        drop_pct = row["macro_f1_drop"] * 100
        ax.text(
            idx,
            max(row["raw_macro_f1"], row["cleaned_macro_f1"]) + 0.02,
            f"Drop: {drop_pct:+.1f}%",
            ha="center",
            fontsize=9,
            fontweight="bold",
            color="#a3302a" if drop_pct > 0 else "#2b5fb4",
        )

    plt.legend(loc="lower right")
    plt.tight_layout()

    out_file = output_dir / "shortcut_comparison.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[T34] Saved shortcut comparison plot -> {out_file}")


# =====================================================================
# MAIN PIPELINE EXECUTION
# =====================================================================


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    npz_files = list(EXPLANATIONS_DIR.glob("*_shap.npz"))
    if not npz_files:
        npz_files = list(Path(".").glob("*_shap.npz"))

    if npz_files:
        plot_calibration_curves(npz_files, FIGURES_DIR, n_bins=5)
        generate_umap_by_label(npz_files, FIGURES_DIR)
        export_error_analysis(npz_files, FIGURES_DIR)

    ensure_tfidf_top_words(
        output_dir=FIGURES_DIR, dataset_name="isot", variant="raw", npz_files=npz_files
    )

    if npz_files:
        plot_shap_vs_tfidf(npz_files, FIGURES_DIR, dataset_name="isot")

    # Execute T34 Shortcut Analysis dynamically from runs.csv
    run_shortcut_analysis(FIGURES_DIR)


if __name__ == "__main__":
    main()
