"""Task T36 (Stretch): Evaluate each dataset's best checkpoint across all datasets."""

from __future__ import annotations
import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

# Disable Hugging Face & PyTorch multithreading to stop Windows C++ thread crashes
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.set_num_threads(1)
if hasattr(torch, "set_num_interop_threads"):
    torch.set_num_interop_threads(1)

from factcheck.data import load_splits

DATASETS = ["covid", "liar", "pubhealth"]
CHECKPOINTS = {
    "covid": "outputs/factcheck_checkpoints_M4_2026-10-28/outputs/covid_twitter_bert_covid/seed42/model",
    "liar": "outputs/factcheck_checkpoints_M4_2026-10-28/outputs/deberta_v3_large_liar/seed42/model",
    "pubhealth": "outputs/factcheck_checkpoints_M4_2026-10-28/outputs/roberta_pubhealth/seed42/model",
}

TEMP_STAGING_DIR = r"C:\factcheck_models"


def stage_checkpoint_outside_onedrive(model_path: str, dataset_name: str) -> str:
    """Copies model checkpoint to C:\\factcheck_models\\<dataset_name> cleanly."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    target_dir = os.path.join(TEMP_STAGING_DIR, dataset_name)
    os.makedirs(target_dir, exist_ok=True)

    print(f"  [1/4] Staging model to local path: {target_dir}...", flush=True)
    shutil.copytree(model_path, target_dir, dirs_exist_ok=True)
    return target_dir


def load_model_and_tokenizer_safely(local_model_path: str):
    """
    Direct state_dict loader that bypasses transformers' ThreadPoolExecutor
    to prevent Windows C++ memory access violations.
    """
    print("  [2/4] Loading tokenizer...", flush=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True, use_fast=False)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)

    print("  [3/4] Instantiating architecture from config & loading weights...", flush=True)
    config = AutoConfig.from_pretrained(local_model_path, local_files_only=True)
    
    # 1. Instantiate empty model structure on CPU (bypasses ThreadPoolExecutor)
    model = AutoModelForSequenceClassification.from_config(config)

    # 2. Directly load state dictionary single-threaded
    bin_file = os.path.join(local_model_path, "pytorch_model.bin")
    safe_file = os.path.join(local_model_path, "model.safetensors")

    if os.path.exists(safe_file):
        from safetensors.torch import load_file
        state_dict = load_file(safe_file, device="cpu")
    elif os.path.exists(bin_file):
        state_dict = torch.load(bin_file, map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(f"No pytorch_model.bin or model.safetensors found in {local_model_path}")

    # 3. Populate parameters safely
    model.load_state_dict(state_dict)
    model.eval()

    print("  [4/4] Model successfully loaded!", flush=True)
    return tokenizer, model


def map_ground_truth_to_binary(labels: list | pd.Series, dataset_name: str) -> np.ndarray:
    """Map ground-truth labels to binary: 0 = False/Fake, 1 = True/Real."""
    binary_labels = []
    for y in labels:
        y_str = str(y).lower().strip()
        if dataset_name == "covid":
            binary_labels.append(int(y))
        elif dataset_name == "liar":
            binary_labels.append(0 if y_str in ["pants-fire", "false", "barely-true", "0", "1", "2"] else 1)
        elif dataset_name == "pubhealth":
            binary_labels.append(0 if y_str in ["false", "mixture", "unproven", "0", "1", "3"] else 1)
        else:
            binary_labels.append(0 if int(y) < 1 else 1)
    return np.array(binary_labels)


def map_logits_to_binary_preds(logits: torch.Tensor, train_dataset: str) -> np.ndarray:
    """Map multi-class or binary logits to binary predictions (0 or 1)."""
    probs = torch.softmax(logits, dim=-1)
    num_classes = logits.shape[-1]

    if num_classes == 2:
        return torch.argmax(probs, dim=-1).cpu().numpy()
    elif train_dataset == "liar" and num_classes == 6:
        prob_false = probs[:, :3].sum(dim=-1)
        prob_true = probs[:, 3:].sum(dim=-1)
        return (prob_true > prob_false).int().cpu().numpy()
    elif train_dataset == "pubhealth" and num_classes == 4:
        prob_true = probs[:, 2]
        return (prob_true > (1.0 - prob_true)).int().cpu().numpy()
    else:
        mid = num_classes // 2
        return (probs[:, mid:].sum(dim=-1) > probs[:, :mid].sum(dim=-1)).int().cpu().numpy()


def run_cross_eval(batch_size: int = 64) -> None:
    device = "cpu"
    print(f"Running cross-domain evaluation on device: {device}", flush=True)
    results = []

    for train_ds, model_path in CHECKPOINTS.items():
        print(f"\n==========================================", flush=True)
        print(f"Loading [{train_ds.upper()}] model...", flush=True)

        try:
            local_model_path = stage_checkpoint_outside_onedrive(model_path, train_ds)
            tokenizer, model = load_model_and_tokenizer_safely(local_model_path)
        except Exception as e:
            print(f" [ERROR] Could not load model {train_ds}: {e}", flush=True)
            continue

        for test_ds in DATASETS:
            print(f"  -> Testing on [{test_ds}]...", flush=True)
            try:
                splits = load_splits(test_ds, "raw")
                test_df = splits["test"]
                texts = test_df["text"].tolist()
                y_true_binary = map_ground_truth_to_binary(test_df["label"], test_ds)

                y_pred_binary = []
                with torch.no_grad():
                    for i in range(0, len(texts), batch_size):
                        batch_texts = texts[i : i + batch_size]
                        inputs = tokenizer(
                            batch_texts,
                            padding=True,
                            truncation=True,
                            max_length=128,
                            return_tensors="pt"
                        ).to(device)
                        logits = model(**inputs).logits
                        y_pred_binary.extend(map_logits_to_binary_preds(logits, train_ds))

                y_pred_binary = np.array(y_pred_binary)
                f1 = f1_score(y_true_binary, y_pred_binary, average="macro")
                acc = accuracy_score(y_true_binary, y_pred_binary)

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
        os.makedirs("results", exist_ok=True)
        df_res = pd.DataFrame(results)
        df_res.to_csv("results/cross_dataset.csv", index=False)
        print("\n==========================================", flush=True)
        print("[T36 COMPLETE] Saved results to results/cross_dataset.csv", flush=True)

        pivot_df = df_res.pivot(index="train_dataset", columns="test_dataset", values="macro_f1")
        print("\n--- Macro-F1 Transfer Matrix ---")
        print(pivot_df.to_string())

    shutil.rmtree(TEMP_STAGING_DIR, ignore_errors=True)


if __name__ == "__main__":
    run_cross_eval()