"""Task 03: Unified Interpretation & Analysis Pipeline (M4 Lead).

Combines:
  - T33: Calibration, UMAP, Error Tagging, TF-IDF vs SHAP
  - T34: Shortcut Analysis (Raw vs. Cleaned ISOT/COVID)
  - T36 / T41: Cross-Domain Heatmap & Climate-FEVER Extrapolation
  - T42: Model Ladder Chart & Specs Pass/Fail Table
  - T57: Consolidated Interpretation Write-Up Exporter
"""

from __future__ import annotations
import zipfile
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import umap

# Define directories
EXPLANATIONS_DIR = Path("outputs/explanations")
FIGURES_DIR = Path("outputs/figures")
RESULTS_DIR = Path("results")
RUNS_CSV = Path("results/runs.csv")

SHORTCUT_CUES = ["reuters", "washington", "twitter", "http", "https", "@", "pic.twitter"]

# =====================================================================
# TASK T33 FUNCTIONS
# =====================================================================

def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 5) -> float:
    """Computes Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (y_prob >= bin_boundaries[i] if i == 0 else y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            # Fix: Use empirical positive rate (y_true mean in bin), NOT threshold match
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

        p_false = probs[:, fake_ids].sum(axis=1) if fake_ids.ndim > 0 else probs[:, 1]
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
            vectors = np.array([v.mean(axis=0) if hasattr(v, 'mean') else np.zeros(6) for v in values])

        if len(vectors) > 5:
            reducer = umap.UMAP(n_neighbors=min(15, len(vectors) - 1), min_dist=0.1, metric="cosine", random_state=42)
            coords = reducer.fit_transform(vectors)
            name = npz_path.stem.replace("_shap", "")

            plt.figure(figsize=(8, 6))
            sns.scatterplot(
                x=coords[:, 0], y=coords[:, 1],
                hue=labels, palette="tab10", alpha=0.8, s=60
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
    source_cues = ["reuters", "washington", "twitter", "http", "https", "@", "pic.twitter"]

    for npz_path in npz_files:
        data = np.load(npz_path, allow_pickle=True)
        ids, texts, labels = data["ids"], data["texts"], data["labels"]
        probs = data["probs"]
        preds = probs.argmax(axis=1)

        for doc_id, text, true_lbl, pred_lbl, prob_dist in zip(ids, texts, labels, preds, probs):
            if true_lbl != pred_lbl:
                t_str = str(text)
                upper_ratio = sum(1 for c in t_str if c.isupper()) / max(1, len(t_str))

                if any(cue in t_str.lower() for cue in source_cues):
                    error_type = "Source Cue Shortcut"
                elif any(kw in t_str for kw in ["!", "URGENT", "BREAKING"]) or upper_ratio > 0.25:
                    error_type = "Stylistic Bias"
                else:
                    error_type = "Content / Semantic Ambiguity"

                error_rows.append({
                    "model": npz_path.stem.replace("_shap", ""),
                    "doc_id": doc_id,
                    "text": t_str,
                    "true_label": true_lbl,
                    "pred_label": pred_lbl,
                    "confidence": float(prob_dist[pred_lbl]),
                    "error_category": error_type
                })

    df_err = pd.DataFrame(error_rows).sort_values(by="confidence", ascending=False)
    out_file = output_dir / "error_analysis_report.csv"
    df_err.to_csv(out_file, index=False)
    print(f"[T33] Exported {len(df_err)} misclassified samples -> {out_file}")


def ensure_tfidf_top_words(
    npz_files: list[Path],
    output_dir: Path,
    dataset_name: str = "covid",  # Options: 'covid', 'isot_clean', 'liar', 'pubhealth'
    tfidf_base_dir: Optional[Path] = None,
) -> Path:
    """Loads pre-computed TF-IDF top words from the friend's tfidf_for_M4 directory.

    Falls back to on-the-fly vectorization if the directory/file is not found.
    """
    target_csv = output_dir / "tfidf_top_words.csv"
    if target_csv.exists():
        return target_csv

    # 1. Look for pre-computed tfidf_for_M4 directory
    search_paths = [
        tfidf_base_dir,
        Path("results/tfidf_for_M4"),
        Path("tfidf_for_M4"),
    ]

    tfidf_dir = None
    for p in search_paths:
        if p and p.exists():
            tfidf_dir = p
            break

    if tfidf_dir:
        # Standard model run for T33 SHAP comparison is Logistic Regression (tfidf_lr_*)
        run_folder = tfidf_dir / "top_words" / f"tfidf_lr_{dataset_name}"
        top_words_file = run_folder / "top_words.csv"

        if top_words_file.exists():
            print(f"[T33] Loading pre-computed TF-IDF top words from {top_words_file}...")
            df = pd.read_csv(top_words_file)

            # Filter for word n-grams (ignoring character n-grams)
            if "kind" in df.columns:
                df = df[df["kind"] == "word"]

            # Map columns to match expected output schema
            df_out = df.rename(
                columns={"feature": "word", "weight": "importance_weight"}
            )[["word", "importance_weight", "label", "rank"]]

            df_out.to_csv(target_csv, index=False)
            print(f"[T33] Exported converted TF-IDF top words -> {target_csv}")
            return target_csv

    # 2. Fallback: Fit TfidfVectorizer on-the-fly if tfidf_for_M4 is missing
    print("[T33 Fallback] tfidf_for_M4 not found. Generating temporary TF-IDF top words from SHAP texts...")
    all_texts, all_labels = [], []
    for p in npz_files:
        data = np.load(p, allow_pickle=True)
        all_texts.extend(list(data["texts"]))
        all_labels.extend(list(data["labels"]))

    if not all_texts:
        return target_csv

    vec = TfidfVectorizer(max_features=5000, stop_words="english")
    X = vec.fit_transform(all_texts)
    clf = LogisticRegression(max_iter=500)
    clf.fit(X, (np.array(all_labels) > 0).astype(int))

    df_words = pd.DataFrame({
        "word": vec.get_feature_names_out(),
        "importance_weight": clf.coef_[0]
    }).sort_values(by="importance_weight", ascending=False)

    df_words.to_csv(target_csv, index=False)
    print(f"[T33] Saved temporary TF-IDF top words -> {target_csv}")
    return target_csv

def plot_shap_vs_tfidf(npz_files: list[Path], output_dir: Path) -> None:
    """Plots a side-by-side comparison of TF-IDF top words vs SHAP word attributions."""
    tfidf_csv = output_dir / "tfidf_top_words.csv"
    if not tfidf_csv.exists():
        print("[T33 Warning] tfidf_top_words.csv missing, skipping comparative plot.")
        return

    # Load TF-IDF Top Words
    df_tfidf = pd.read_csv(tfidf_csv)
    top_tfidf = df_tfidf.head(10)[["word", "importance_weight"]]

    # Extract Top SHAP Words across provided NPZ models
    word_shap_scores: dict[str, float] = {}
    for p in npz_files:
        data = np.load(p, allow_pickle=True)
        if "texts" in data and "values" in data:
            texts = data["texts"]
            values = data["values"]
            for text, val_matrix in zip(texts, values):
                tokens = str(text).split()
                # Aggregate absolute SHAP importance per word token
                scores = np.abs(val_matrix).mean(axis=-1) if val_matrix.ndim > 1 else np.abs(val_matrix)
                for tok, sc in zip(tokens, scores):
                    tok_clean = tok.lower().strip(".,!?:;\"'")
                    if len(tok_clean) > 2 and tok_clean not in ENGLISH_STOP_WORDS:
                        word_shap_scores[tok_clean] = word_shap_scores.get(tok_clean, 0.0) + float(sc)
                        
    df_shap = (
        pd.DataFrame(list(word_shap_scores.items()), columns=["word", "importance_weight"])
        .sort_values(by="importance_weight", ascending=False)
        .head(10)
    )

    # Plot Side-by-Side Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    # TF-IDF Barplot
    sns.barplot(
        data=top_tfidf,
        x="importance_weight",
        y="word",
        ax=axes[0],
        palette="Blues_r",
        hue="word",
        legend=False,
    )
    axes[0].set_title("Top 10 Baseline TF-IDF Top Words (Weight)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Logistic Regression Coefficient Weight")

    # SHAP Barplot
    sns.barplot(
        data=df_shap,
        x="importance_weight",
        y="word",
        ax=axes[1],
        palette="Oranges_r",
        hue="word",
        legend=False,
    )
    axes[1].set_title("Top 10 Model SHAP Attributed Words (|SHAP|)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Mean Absolute SHAP Value")

    plt.suptitle("Interpretation Analysis: TF-IDF vs SHAP Word Importances (T33)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_file = output_dir / "shap_vs_tfidf_comparison.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[T33] Saved SHAP vs TF-IDF comparison plot -> {out_file}")


# =====================================================================
# TASK T34 FUNCTIONS
# =====================================================================

def run_shortcut_analysis(output_dir: Path = FIGURES_DIR) -> pd.DataFrame:
    """Analyzes shortcut reliance between raw and cleaned dataset variants.
    
    Calculates accuracy drop, Macro-F1 drop, mean probability shift,
    and SHAP attribution weight shifts on source tokens.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    raw_files = list(EXPLANATIONS_DIR.glob("*raw*_shap.npz")) + list(EXPLANATIONS_DIR.glob("*seed42_shap.npz"))
    # Filter out files that are already clean targets
    raw_files = [f for f in raw_files if "_clean" not in f.stem]

    results = []

    for raw_path in raw_files:
        # Check potential clean file naming patterns
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

        # Metrics: Accuracy & Macro-F1
        raw_acc = float(accuracy_score(labels, raw_preds))
        clean_acc = float(accuracy_score(labels, clean_preds))
        raw_f1 = float(f1_score(labels, raw_preds, average="macro"))
        clean_f1 = float(f1_score(labels, clean_preds, average="macro"))

        # Mean probability distribution shift
        prob_shift = float(np.abs(raw_data["probs"] - clean_data["probs"]).mean())

        # SHAP attribution shift on shortcut tokens (if token attributions exist)
        shortcut_shap_weight_raw = 0.0
        if "values" in raw_data and "texts" in raw_data:
            shap_vals = raw_data["values"]
            texts = raw_data["texts"]
            cue_weights = []
            for text, val in zip(texts, shap_vals):
                tokens = str(text).lower().split()
                # Aggregate absolute attribution on shortcut cues
                cue_indices = [idx for idx, tok in enumerate(tokens) if any(cue in tok for cue in SHORTCUT_CUES)]
                if cue_indices and hasattr(val, "__len__") and len(val) > max(cue_indices):
                    cue_weights.append(np.abs(val[cue_indices]).sum())
            if cue_weights:
                shortcut_shap_weight_raw = float(np.mean(cue_weights))

        model_name = raw_path.stem.replace("_seed42_shap", "").replace("_shap", "")

        results.append({
            "model": model_name,
            "raw_accuracy": raw_acc,
            "cleaned_accuracy": clean_acc,
            "accuracy_drop": raw_acc - clean_acc,
            "raw_macro_f1": raw_f1,
            "cleaned_macro_f1": clean_f1,
            "macro_f1_drop": raw_f1 - clean_f1,
            "mean_prob_shift": prob_shift,
            "shortcut_shap_impact": shortcut_shap_weight_raw,
        })

    if not results:
        print("[T34] Warning: No paired raw/clean .npz files found in outputs/explanations.")
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # 1. Save detailed CSV
    csv_out = RESULTS_DIR / "shortcut_study_t34.csv"
    df.to_csv(csv_out, index=False)
    print(f"[T34] Saved shortcut analysis CSV -> {csv_out}")

    # 2. Plot Raw vs. Cleaned Performance Comparison
    plot_shortcut_results(df, output_dir)

    return df


def plot_shortcut_results(df: pd.DataFrame, output_dir: Path) -> None:
    """Generates a side-by-side bar chart of Raw vs. Cleaned performance."""
    plot_data = []
    for _, row in df.iterrows():
        plot_data.append({"Model": row["model"], "Variant": "Raw Text (With Shortcuts)", "Macro-F1": row["raw_macro_f1"]})
        plot_data.append({"Model": row["model"], "Variant": "Cleaned Text (No Shortcuts)", "Macro-F1": row["cleaned_macro_f1"]})

    df_plot = pd.DataFrame(plot_data)

    plt.figure(figsize=(9, 5))
    ax = sns.barplot(
        data=df_plot,
        x="Model",
        y="Macro-F1",
        hue="Variant",
        palette=["#a3302a", "#2b5fb4"]
    )
    plt.title("Shortcut Study: Model Macro-F1 on Raw vs. Cleaned Text (T34)", fontsize=13, fontweight="bold")
    plt.ylim(0, 1.05)
    plt.ylabel("Macro-F1 Score")
    plt.xlabel("Model Configuration")
    plt.grid(axis="y", linestyle=":", alpha=0.6)

    # Annotate percentage drop on bars
    for idx, row in df.iterrows():
        drop_pct = row["macro_f1_drop"] * 100
        ax.text(
            idx,
            row["raw_macro_f1"] + 0.02,
            f"Drop: -{drop_pct:.1f}%",
            ha="center",
            fontsize=9,
            fontweight="bold",
            color="#a3302a"
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

    if not npz_files:
        print("No .npz files found.")
        return

    plot_calibration_curves(npz_files, FIGURES_DIR, n_bins=5)
    generate_umap_by_label(npz_files, FIGURES_DIR)
    export_error_analysis(npz_files, FIGURES_DIR)
    ensure_tfidf_top_words(npz_files, FIGURES_DIR)
    
    # Generate side-by-side comparison visualization
    plot_shap_vs_tfidf(npz_files, FIGURES_DIR)

    # Execute T34 (Shortcut Analysis)
    run_shortcut_analysis(FIGURES_DIR)


if __name__ == "__main__":
    main()