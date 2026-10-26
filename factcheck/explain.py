from __future__ import annotations

"""
SHAP Explainability Module (Owner: M4, Task: T16, T33)

Calculates token-level SHAP attributions for transformer classifiers and outputs:
1. Token attribution dictionaries (for feature importance & Streamlit UI).
2. HTML highlights & saved numpy arrays (for analysis in notebook 03).
"""

import os
# Force transformers to load model weights on the main thread (prevents Windows threadpool C-crash)
os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pathlib
import numpy as np
import torch
import shap
from typing import Dict, Any, List, Tuple, Optional, Union
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


def get_shap_explainer(
    model_name_or_path: str,
    device: int = -1
) -> Tuple[shap.Explainer, Any, Any]:
    """
    Loads a Hugging Face classification pipeline and creates a SHAP Explainer.
    
    Args:
        model_name_or_path: Local path to saved checkpoint or Hugging Face model ID.
        device: GPU device ID (0 for cuda:0, -1 for CPU).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        low_cpu_mem_usage=False
    )

    pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        top_k=None,
        device=device
    )

    explainer = shap.Explainer(pipe)
    return explainer, model, tokenizer


def explain_claim(
    claim_text: str,
    explainer: shap.Explainer,
    target_class_idx: int = 1,
    max_evals: int = 500
) -> Dict[str, Any]:
    """
    Computes SHAP attributions for a single claim text input.
    
    Args:
        claim_text: Text claim to explain.
        explainer: Initialized SHAP Explainer object.
        target_class_idx: Class label index to explain (e.g., 1 for 'fake', or 0-5 for LIAR).
        max_evals: Max function evaluations for SHAP partition explainer.
    """
    shap_values = explainer([claim_text], max_evals=max_evals)
    tokens = list(shap_values.data[0])
    
    raw_vals = shap_values.values[0]
    if raw_vals.ndim == 2:
        scores = raw_vals[:, target_class_idx].tolist()
        base_val = float(shap_values.base_values[0][target_class_idx])
    else:
        scores = raw_vals.tolist()
        base_val = float(shap_values.base_values[0])

    token_attributions = [
        {"token": str(tok), "score": float(sc)}
        for tok, sc in zip(tokens, scores)
    ]

    return {
        "text": claim_text,
        "target_class_idx": target_class_idx,
        "base_value": base_val,
        "token_attributions": token_attributions,
        "shap_values_obj": shap_values
    }


def explain_dataset_batch(
    claims: List[str],
    explainer: shap.Explainer,
    output_dir: str = "results/explanations",
    filename_prefix: str = "distilbert_covid",
    max_evals: int = 500
) -> str:
    """
    Computes SHAP attributions across a fixed batch of evaluation claims and saves artifacts.
    """
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    shap_values = explainer(claims, max_evals=max_evals)
    
    npz_path = os.path.join(output_dir, f"{filename_prefix}_shap_values.npz")
    np.savez_compressed(
        npz_path,
        values=np.array(shap_values.values, dtype=object),
        base_values=np.array(shap_values.base_values, dtype=object),
        data=np.array(claims, dtype=object)
    )

    html_content = shap.plots.text(shap_values, display=False)
    html_path = os.path.join(output_dir, f"{filename_prefix}_text_plots.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(str(html_content))

    return html_path


if __name__ == "__main__":
    import gc
    import os
    import pathlib
    import pandas as pd

    def find_path_upwards(relative_path: str) -> pathlib.Path:
        """Searches parent directories to resolve a relative path."""
        search_roots = [
            pathlib.Path(__file__).resolve().parent,
            pathlib.Path.cwd().resolve()
        ]
        for root in search_roots:
            for p in [root] + list(root.parents):
                candidate = p / relative_path
                if candidate.exists():
                    return candidate
        return None

    # 1. Locate model checkpoints folder
    FOLDER_NAME = "t06_distilbert_checkpoints_seed42"
    BASE_DIR = find_path_upwards(f"models/{FOLDER_NAME}") or find_path_upwards(FOLDER_NAME)

    if BASE_DIR is None:
        raise FileNotFoundError(f"Checkpoints directory '{FOLDER_NAME}' not found.")

    print(f"Successfully located checkpoints directory at: {BASE_DIR}")

    # 2. Configure dataset runs pointing to data/splits/
    runs_config = [
        {
            "name": "COVID-19",
            "checkpoint_path": str(BASE_DIR / "distilbert_covid" / "seed42" / "model"),
            "csv_filename": "covid.csv",
            "prefix": "t06_distilbert_covid_50"
        },
        {
            "name": "LIAR",
            "checkpoint_path": str(BASE_DIR / "distilbert_liar" / "seed42" / "model"),
            "csv_filename": "liar.csv",
            "prefix": "t06_distilbert_liar_50"
        },
        {
            "name": "Public Health",
            "checkpoint_path": str(BASE_DIR / "distilbert_pubhealth" / "seed42" / "model"),
            "csv_filename": "pubhealth.csv",
            "prefix": "t06_distilbert_pubhealth_50"
        }
    ]

    for run in runs_config:
        ckpt_path = run["checkpoint_path"]
        print(f"\n==========================================")
        print(f"Processing Dataset: {run['name']}")
        print(f"Loading checkpoint: {ckpt_path}")
        print(f"==========================================")

        if not os.path.exists(ckpt_path):
            print(f"Skipping {run['name']}: Directory '{ckpt_path}' not found.")
            continue

        # Locate dataset CSV in data/splits/
        csv_path = find_path_upwards(f"data/splits/{run['csv_filename']}")

        if csv_path and csv_path.exists():
            print(f"Loading data from: {csv_path}")
            df_test = pd.read_csv(csv_path)

            # Filter for test set if 'split' column exists
            if "split" in df_test.columns:
                df_test = df_test[df_test["split"] == "test"]

            # Identify target text column
            text_col = "claim" if "claim" in df_test.columns else ("text" if "text" in df_test.columns else df_test.columns[0])

            # Clean and sanitize strings
            df_test = df_test.dropna(subset=[text_col])
            df_test[text_col] = df_test[text_col].astype(str).str.strip()
            df_test = df_test[df_test[text_col] != ""]

            claims = df_test[text_col].head(50).tolist()
        else:
            print(f"Warning: CSV '{run['csv_filename']}' not found in 'data/splits/'. Using dummy claims.")
            claims = [f"Sample claim evaluation for {run['name']} dataset."] * 50

        # Load Explainer & Model
        explainer, model, tokenizer = get_shap_explainer(ckpt_path, device=-1)

        # Generate output artifacts (.npz and .html)
        html_path = explain_dataset_batch(
            claims=claims,
            explainer=explainer,
            output_dir="results/explanations",
            filename_prefix=run["prefix"]
        )

        print(f"Saved SHAP explanations for {run['name']}:")
        print(f" - HTML report: {html_path}")
        print(f" - NPZ array: results/explanations/{run['prefix']}_shap_values.npz")

        # Cleanup GPU/RAM memory
        del explainer, model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()