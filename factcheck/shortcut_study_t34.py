"""Task T34: Analyze raw vs. cleaned text shortcut reliance."""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

EXPLANATIONS_DIR = Path("outputs/explanations")


def analyze_shortcuts() -> None:
    raw_files = list(EXPLANATIONS_DIR.glob("*_seed42_shap.npz"))
    results = []

    for raw_path in raw_files:
        clean_path = EXPLANATIONS_DIR / f"{raw_path.stem}_on_clean_shap.npz"
        if not clean_path.exists():
            continue

        raw_data = np.load(raw_path, allow_pickle=True)
        clean_data = np.load(clean_path, allow_pickle=True)

        raw_acc = (raw_data["probs"].argmax(axis=1) == raw_data["labels"]).mean()
        clean_acc = (clean_data["probs"].argmax(axis=1) == clean_data["labels"]).mean()
        
        # Mean probability distribution shift
        prob_shift = np.abs(raw_data["probs"] - clean_data["probs"]).mean()

        results.append({
            "model": raw_path.stem.replace("_seed42_shap", ""),
            "raw_accuracy": raw_acc,
            "cleaned_accuracy": clean_acc,
            "accuracy_drop": raw_acc - clean_acc,
            "mean_prob_shift": prob_shift,
        })

    df = pd.DataFrame(results)
    print("=== T34 Shortcut Analysis Summary ===")
    print(df.to_string(index=False))
    df.to_csv("results/shortcut_study_t34.csv", index=False)

if __name__ == "__main__":
    analyze_shortcuts()