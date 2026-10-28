from __future__ import annotations

"""
SHAP Explainability Module (Owner: M4, Tasks: T16, T33)

Calculates token-level SHAP attributions for transformer classifiers and outputs:
1. Token attribution dictionaries (for feature importance & Streamlit UI).
2. HTML highlights & saved compressed numpy arrays (.npz) for evaluation analysis.
"""

import os
os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import argparse
import gc
import pathlib
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import shap
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Ensure project root directory is in sys.path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def find_path_upwards(relative_path: str) -> Optional[pathlib.Path]:
    """Searches parent directories to resolve a relative project path."""
    search_roots = [
        pathlib.Path(__file__).resolve().parent,
        pathlib.Path.cwd().resolve(),
        PROJECT_ROOT
    ]
    for root in search_roots:
        for p in [root] + list(root.parents):
            candidate = p / relative_path
            if candidate.exists():
                return candidate
    return None


def load_dataset_claims(
    dataset_name: str,
    num_samples: int = 50,
    only_llm_subset: bool = True
) -> Tuple[List[str], List[Any]]:
    """
    Loads dataset claims using the project's native `load_splits` utility or raw CSV fallback.
    Ensures non-empty claim retrieval for evaluation.
    """
    df_test = None

    # 1. Primary: Load via factcheck.data.load_splits
    try:
        from factcheck.data import load_splits
        splits = load_splits(dataset_name)
        if isinstance(splits, dict) and "test" in splits:
            df_test = splits["test"]
        elif isinstance(splits, pd.DataFrame):
            df_test = splits
    except Exception as e:
        print(f"Notice: load_splits import/load failed ({e}). Searching local data directory...")

    # 2. Fallback: Direct CSV loading from data/ directory
    if df_test is None:
        clean_name = dataset_name.lower().replace("-", "").replace("_", "")
        data_dir = find_path_upwards("data")
        if data_dir:
            for csv_path in data_dir.glob("**/*.csv"):
                if clean_name in csv_path.stem.lower().replace("-", "").replace("_", ""):
                    try:
                        temp_df = pd.read_csv(csv_path)
                        candidate_cols = ["claim", "statement", "text", "tweet", "headline", "content"]
                        if any(c in temp_df.columns for c in candidate_cols):
                            df_test = temp_df
                            break
                    except Exception:
                        continue

    if df_test is None or df_test.empty:
        raise RuntimeError(f"Could not load any data rows for dataset '{dataset_name}'. Check data/ directory.")

    # 3. Filter for test split
    if "split" in df_test.columns:
        df_test = df_test[df_test["split"] == "test"]

    # 4. Filter by in_llm_subset if available
    filtered_df = df_test.copy()
    if only_llm_subset and "in_llm_subset" in filtered_df.columns:
        mask = filtered_df["in_llm_subset"].astype(str).str.lower().isin(["true", "1", "t"])
        sub_df = filtered_df[mask]
        if not sub_df.empty:
            filtered_df = sub_df
        else:
            print(f"Warning: 'in_llm_subset' filtering produced 0 rows for '{dataset_name}'. Using standard test set.")

    # 5. Extract claim text column
    candidate_cols = ["text", "claim", "statement", "tweet", "headline", "content", "article"]
    text_col = next((col for col in candidate_cols if col in filtered_df.columns), None)

    if text_col is None:
        non_meta = [c for c in filtered_df.columns if c not in ["id", "split", "in_llm_subset", "label"]]
        text_col = non_meta[0] if non_meta else filtered_df.columns[0]

    # Clean text values
    filtered_df = filtered_df.dropna(subset=[text_col])
    filtered_df[text_col] = filtered_df[text_col].astype(str).str.strip()
    filtered_df = filtered_df[filtered_df[text_col] != ""]

    df_subset = filtered_df.head(num_samples)
    claims = df_subset[text_col].tolist()
    ids = df_subset["id"].tolist() if "id" in df_subset.columns else list(range(len(claims)))

    if not claims:
        raise ValueError(f"Extracted 0 valid claim strings for dataset '{dataset_name}'. Check dataset files.")

    print(f"Loaded {len(claims)} evaluation claims for dataset '{dataset_name}'.")
    return claims, ids


def get_shap_explainer(
    model_name_or_path: str,
    device: int = -1,
    max_length: int = 512
) -> Tuple[shap.Explainer, Any, Any]:
    """
    Loads Hugging Face model/tokenizer and creates a SHAP Explainer using a custom
    probability prediction function and Text masker.
    """
    path_obj = pathlib.Path(model_name_or_path)
    resolved_path = find_path_upwards(model_name_or_path) or (path_obj.resolve() if path_obj.exists() else None)

    if resolved_path and resolved_path.exists():
        model_name_or_path = str(resolved_path)
    else:
        local_prefixes = ("outputs", "models", "factcheck", ".", "..")
        is_local_intent = any(model_name_or_path.startswith(p) for p in local_prefixes) or "\\" in model_name_or_path
        if is_local_intent:
            raise FileNotFoundError(f"Local checkpoint directory '{model_name_or_path}' not found on disk.")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, low_cpu_mem_usage=False)

    target_device = torch.device(f"cuda:{device}" if device >= 0 and torch.cuda.is_available() else "cpu")
    model.to(target_device)
    model.eval()

    def predict_proba(texts: Union[List[str], np.ndarray]) -> np.ndarray:
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        text_list = [str(t) for t in texts]
        
        inputs = tokenizer(
            text_list,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(target_device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
        return probs

    masker = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(predict_proba, masker)
    return explainer, model, tokenizer


def explain_dataset_batch(
    claims: List[str],
    explainer: shap.Explainer,
    output_dir: str = "outputs/explanations",
    filename_prefix: str = "distilbert_covid",
    max_evals: int = 500,
    sample_ids: Optional[List[Any]] = None
) -> Tuple[str, str]:
    """Computes SHAP attributions across evaluation claims and saves outputs."""
    if not claims:
        raise ValueError("Cannot run SHAP explanation on an empty claims list.")

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Computing SHAP values for {len(claims)} claims (max_evals={max_evals})...")
    shap_values = explainer(claims, max_evals=max_evals)

    npz_path = os.path.join(output_dir, f"{filename_prefix}_shap_values.npz")
    np.savez_compressed(
        npz_path,
        values=np.array(shap_values.values, dtype=object),
        base_values=np.array(shap_values.base_values, dtype=object),
        data=np.array(claims, dtype=object),
        ids=np.array(sample_ids if sample_ids is not None else [], dtype=object)
    )

    html_content = shap.plots.text(shap_values, display=False)
    html_path = os.path.join(output_dir, f"{filename_prefix}_text_plots.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(str(html_content))

    return html_path, npz_path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate SHAP token attributions for sequence classification checkpoints.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to local checkpoint directory or HF model ID.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., covid, liar, pubhealth, isot).")
    parser.add_argument("--output_dir", type=str, default="outputs/explanations", help="Directory where artifacts are saved.")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of claims from evaluation set to explain.")
    parser.add_argument("--max_evals", type=int, default=500, help="Max evaluations for SHAP partition explainer.")
    parser.add_argument("--device", type=int, default=-1, help="Device ID (-1 for CPU, 0 for CUDA:0).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"Running SHAP explanation for checkpoint: {args.checkpoint} on dataset: {args.dataset}")
    claims, sample_ids = load_dataset_claims(args.dataset, num_samples=args.num_samples, only_llm_subset=True)
    
    explainer, model, tokenizer = get_shap_explainer(args.checkpoint, device=args.device)
    prefix = f"t16_{args.dataset}_{len(claims)}"

    html_path, npz_path = explain_dataset_batch(
        claims=claims,
        explainer=explainer,
        output_dir=args.output_dir,
        filename_prefix=prefix,
        max_evals=args.max_evals,
        sample_ids=sample_ids
    )

    print(f"\nSuccessfully generated SHAP attributions:")
    print(f"  HTML Report: {html_path}")
    print(f"  NPZ Array:   {npz_path}")