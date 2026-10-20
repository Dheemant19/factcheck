"""Prove a machine can fine-tune a transformer: a tiny DistilBERT run on COVID (task T02).

    python scripts/download_data.py covid
    python scripts/smoke_test.py                 # about a minute on a GPU
    python scripts/smoke_test.py --cpu           # CPU only: a smaller model and sample

Trains on a stratified sample of the raw COVID training posts for one epoch and scores the
official validation set. A working setup reaches well above 0.80 accuracy; about 0.5 means
training is broken (it is a balanced two-class set). Nothing is saved or logged to runs.csv.
This reads the raw CSVs directly, so it does not depend on the project's loaders.
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from factcheck.paths import RAW_DIR  # noqa: E402

LABELS = {"real": 0, "fake": 1}


def driver_version() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        )  # fmt: skip
        return out.stdout.strip().splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        return "n/a"


def load_covid(split_file: str, n: int | None, seed: int) -> pd.DataFrame:
    path = RAW_DIR / "covid" / split_file
    if not path.exists():
        raise SystemExit(f"{path} not found. Run: python scripts/download_data.py covid")
    df = pd.read_csv(path)
    df = df[df["label"].isin(LABELS)].assign(y=lambda d: d["label"].map(LABELS))
    if n and n < len(df):  # stratified sample, same class balance as the full split
        df = df.groupby("y").sample(frac=n / len(df), random_state=seed)
    return df.reset_index(drop=True)


def batches(df: pd.DataFrame, tokenizer, batch_size: int, max_length: int, shuffle: bool):
    def collate(rows: list[tuple[str, int]]) -> dict[str, torch.Tensor]:
        texts, ys = zip(*rows, strict=True)
        enc = tokenizer(
            list(texts), truncation=True, max_length=max_length, padding=True,
            return_tensors="pt",
        )  # fmt: skip
        enc["labels"] = torch.tensor(ys)
        return enc

    rows = list(zip(df["tweet"].astype(str), df["y"], strict=True))
    return DataLoader(rows, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--model", default=None, help="default: distilbert-base-uncased")
    parser.add_argument("--train-rows", type=int, default=None, help="default: 2000 (CPU 600)")
    parser.add_argument("--cpu", action="store_true", help="force CPU, smaller model and sample")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    on_gpu = device.type == "cuda"
    model_name = args.model or (
        "distilbert-base-uncased" if on_gpu else "google/bert_uncased_L-4_H-256_A-4"
    )
    train_rows = args.train_rows or (2000 if on_gpu else 600)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train = load_covid("Constraint_Train.csv", train_rows, args.seed)
    val = load_covid("Constraint_Val.csv", None if on_gpu else 400, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    train_loader = batches(train, tokenizer, 32, 128, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    total = len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / (0.1 * total)) * max(0.0, 1 - step / total)
    )
    use_bf16 = on_gpu and torch.cuda.is_bf16_supported()
    if on_gpu:
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    model.train()
    for step, batch in enumerate(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
            loss = model(**batch).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        if step % 20 == 0:
            print(f"step {step:3d}/{total}  loss {loss.item():.3f}")
    train_s = time.perf_counter() - start

    model.eval()
    preds = []
    with torch.no_grad():
        for batch in batches(val, tokenizer, 64, 128, shuffle=False):
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                preds.append(model(**batch).logits.argmax(-1).cpu())
    pred = torch.cat(preds).numpy()
    acc = accuracy_score(val["y"], pred)
    macro_f1 = f1_score(val["y"], pred, average="macro")

    gpu = torch.cuda.get_device_name(0) if on_gpu else "none (CPU)"
    peak = f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GB" if on_gpu else "n/a"
    print()
    print(f"machine       {platform.node()}  ({platform.system()} {platform.release()})")
    print(f"gpu           {gpu}")
    print(f"driver        {driver_version()}")
    print(f"torch         {torch.__version__}  (CUDA {torch.version.cuda})")
    print(f"model         {model_name}  ({len(train)} train rows, 1 epoch, bf16 {use_bf16})")
    print(f"train time    {train_s:.1f} s   peak GPU memory {peak}")
    print(f"validation    accuracy {acc:.3f}   macro-F1 {macro_f1:.3f}   ({len(val)} posts)")
    if acc < 0.75:
        raise SystemExit(
            "FAIL: accuracy is near chance, so training is not working on this machine."
        )
    print("PASS: this machine can fine-tune transformers.")


if __name__ == "__main__":
    main()
