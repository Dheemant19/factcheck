"""Task T36 (Stretch): Evaluate each dataset's best checkpoint across all datasets."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from factcheck.data import get_dataset, load_splits
from factcheck.paths import OUTPUT_DIR, RESULTS_DIR

DATASETS = ["covid", "liar", "pubhealth"]
TEMP_STAGING_DIR = Path(tempfile.gettempdir()) / "factcheck_models"


def resolve_checkpoint_path(train_ds: str) -> Path:
    """Find best checkpoint path dynamically from runs.csv or directory search."""
    runs_csv = RESULTS_DIR / "runs.csv"
    if not runs_csv.exists():
        runs_csv = Path("runs.csv")

    if runs_csv.exists():
        try:
            df_runs = pd.read_csv(runs_csv)
            mask = (df_runs["dataset"].str.lower() == train_ds.lower()) & (df_runs["split"].str.lower() == "validation")
            df_ds = df_runs[mask]
            if not df_ds.empty:
                best_run = df_ds.sort_values(by="macro_f1", ascending=False).iloc[0]
                run_name = str(best_run["run_name"])
                seed = best_run["seed"]

                candidates = [
                    OUTPUT_DIR / run_name / f"seed{seed}" / "model",
                    OUTPUT_DIR / "factcheck_checkpoints_M4_2026-10-28" / "outputs" / run_name / f"seed{seed}" / "model",
                    OUTPUT_DIR / run_name / "model",
                ]
                for c in candidates:
                    if c.exists():
                        return c
        except Exception:
            pass

    # Directory glob fallback search
    glob_patterns = [
        f"*{train_ds}*/seed*/model",
        f"**/*{train_ds}*/seed*/model",
        f"**/*{train_ds}*/model",
    ]
    for pat in glob_patterns:
        matches = list(OUTPUT_DIR.glob(pat))
        if matches:
            return matches[0]

    # Explicit fallback paths
    fallbacks = {
        "covid": OUTPUT_DIR / "factcheck_checkpoints_M4_2026-10-28/outputs/covid_twitter_bert_covid/seed42/model",
        "liar": OUTPUT_DIR / "factcheck_checkpoints_M4_2026-10-28/outputs/deberta_v3_large_liar/seed42/model",
        "pubhealth": OUTPUT_DIR / "factcheck_checkpoints_M4_2026-10-28/outputs/roberta_pubhealth/seed42/model",
    }
    return fallbacks.get(train_ds, OUTPUT_DIR / train_ds / "model")


def stage_checkpoint(model_path: Path, dataset_name: str) -> Path:
    """Copies model checkpoint to local temp directory cleanly to prevent lock issues."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    target_dir = TEMP_STAGING_DIR / dataset_name
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"  [1/4] Staging model to local path: {target_dir}...", flush=True)
    shutil.copytree(model_path, target_dir, dirs_exist_ok=True)
    return target_dir


def load_model_and_tokenizer_safely(local_model_path: Path):
    """Direct state_dict loader that bypasses transformers ThreadPoolExecutor."""
    str_path = str(local_model_path)
    print("  [2/4] Loading tokenizer...", flush=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(str_path, local_files_only=True, use_fast=False)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(str_path, local_files_only=True)

    print("  [3/4] Instantiating architecture from config & loading weights...", flush=True)
    config = AutoConfig.from_pretrained(str_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_config(config)

    bin_file = local_model_path / "pytorch_model.bin"
    safe_file = local_model_path / "model.safetensors"

    if safe_file.exists():
        from safetensors.torch import load_file
        state_dict = load_file(str(safe_file), device="cpu")
    elif bin_file.exists():
        state_dict = torch.load(str(bin_file), map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(f"No pytorch_model.bin or model.safetensors found in {local_model_path}")

    model.load_state_dict(state_dict)
    model.eval()

    print("  [4/4] Model successfully loaded!", flush=True)
    return tokenizer, model


def map_logits_to_binary_preds(logits: torch.Tensor, fake_label_ids: list | int) -> np.ndarray:
    """Map multi-class or binary logits to binary predictions (1 = False/Fake, 0 = True/Real)."""
    probs = torch.softmax(logits, dim=-1)
    if isinstance(fake_label_ids, (int, np.integer)):
        fake_ids = [fake_label_ids]
    else:
        fake_ids = list(fake_label_ids)

    p_fake = probs[:, fake_ids].sum(dim=-1)
    return (p_fake >= 0.5).int().cpu().numpy()


def run_cross_eval(batch_size: int = 64) -> None:
    device = "cpu"
    print(f"Running cross-domain evaluation on device: {device}", flush=True)
    results = []

    for train_ds in DATASETS:
        model_path = resolve_checkpoint_path(train_ds)
        print("\n==========================================", flush=True)
        print(f"Loading [{train_ds.upper()}] model from {model_path}...", flush=True)

        try:
            local_model_path = stage_checkpoint(model_path, train_ds)
            tokenizer, model = load_model_and_tokenizer_safely(local_model_path)
        except Exception as e:
            print(f"  [ERROR] Could not load model {train_ds}: {e}", flush=True)
            continue

        train_info = get_dataset(train_ds)
        fake_label_ids = train_info.fake_label_ids
        max_len = getattr(train_info, "max_length", 128)

        for test_ds in DATASETS:
            print(f"  -> Testing on [{test_ds}]...", flush=True)
            try:
                splits = load_splits(test_ds, "raw")
                test_df = splits["test"]

                # 1. Standardize binary labels and drop binary_label == -1 (left out classes)
                test_df = test_df[test_df["binary_label"] != -1].copy()
                texts = test_df["text"].tolist()
                y_true_binary = test_df["binary_label"].to_numpy().astype(int)

                # 2. Compute predictions using model's fake probability sum
                y_pred_binary = []
                with torch.no_grad():
                    for i in range(0, len(texts), batch_size):
                        batch_texts = texts[i : i + batch_size]
                        inputs = tokenizer(
                            batch_texts,
                            padding=True,
                            truncation=True,
                            max_length=max_len,
                            return_tensors="pt"
                        ).to(device)
                        logits = model(**inputs).logits
                        preds = map_logits_to_binary_preds(logits, fake_label_ids)
                        y_pred_binary.extend(preds)

                y_pred_binary = np.array(y_pred_binary)
                f1 = float(f1_score(y_true_binary, y_pred_binary, average="macro"))
                acc = float(accuracy_score(y_true_binary, y_pred_binary))

                print(f"     [RESULT] [{train_ds.upper()} -> {test_ds}] Macro-F1: {f1:.4f} | Accuracy: {acc:.4f}", flush=True)

                results.append({
                    "train_dataset": train_ds,
                    "test_dataset": test_ds,
                    "macro_f1": round(f1, 4),
                    "accuracy": round(acc, 4)
                })
            except Exception as e:
                print(f"     [ERROR] Inference failed on {test_ds}: {e}", flush=True)

        del model, tokenizer
        shutil.rmtree(local_model_path, ignore_errors=True)

    if results:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_csv = RESULTS_DIR / "cross_dataset.csv"
        df_res = pd.DataFrame(results)
        df_res.to_csv(out_csv, index=False)
        print("\n==========================================", flush=True)
        print(f"[T36 COMPLETE] Saved results to {out_csv}", flush=True)

        pivot_df = df_res.pivot(index="train_dataset", columns="test_dataset", values="macro_f1")
        print("\n--- Macro-F1 Transfer Matrix ---")
        print(pivot_df.to_string())

    shutil.rmtree(TEMP_STAGING_DIR, ignore_errors=True)


if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)

    run_cross_eval()