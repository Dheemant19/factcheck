"""Fine-tuned transformer encoders (owners M1 + M2, T06; LoRA follows in T22).

Any Hugging Face encoder with a sequence-classification head: DistilBERT, BERT, RoBERTa,
DeBERTa-v3, ModernBERT and the domain models. Settings follow plan Section 10: AdamW with
weight decay 0.01, learning rate 2e-5, linear decay after 6% warmup, batch 32 (16 at 384 to
512 tokens), up to 4 epochs, grad clip 1.0, bf16 where the GPU supports it, early stopping
on validation macro-F1 with the best epoch kept. Efficiency (plan Section 16): dynamic padding,
length-grouped batches and tokenised splits cached to disk.

    python -m factcheck.train --config configs/encoder/distilbert_covid.yaml

Each run writes, under ``outputs/<name>/seed<N>/``:

    config.yaml              the full config used (defaults filled in); the app reads it
    model/                   the best epoch, loadable with AutoModelForSequenceClassification
    train_summary.json       per-epoch validation scores, trainable parameters, time, memory

and logs the validation and test scores to ``results/runs.csv`` through
``factcheck.evaluate.log_run``, which also saves the per-row probabilities.

Long texts (ISOT) can keep the head and the tail of the article instead of only the head:
``data.head_tokens: 128`` with ``max_length: 512`` keeps the first 128 and the last 382 tokens.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sys
import time
from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from transformers import set_seed as hf_set_seed

from factcheck.data import get_dataset, load_splits
from factcheck.data.registry import DatasetInfo
from factcheck.evaluate import compute_metrics, log_run, measure_latency_ms
from factcheck.paths import CACHE_DIR, OUTPUT_DIR

# Plan Section 10, base-encoder column. A run config only needs to list what it changes.
DEFAULTS = {
    "data": {
        "max_length": None,  # tokens; None = the dataset's max_length from the registry
        "head_tokens": None,  # keep this many head tokens and fill the rest from the tail
        "subset": None,  # train on a stratified sample of this many rows (debug runs only)
    },
    "model": {"method": "full"},
    "train": {
        "learning_rate": 2.0e-5,
        "batch_size": None,  # None = 32, or 16 at 384 tokens and above
        "eval_batch_size": 64,
        "grad_accum": 1,
        "epochs": 4,
        "warmup_ratio": 0.06,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "precision": "bf16",  # bf16 or fp32; bf16 falls back to fp32 on GPUs without it
        "early_stopping_patience": 2,
        "num_workers": 0,
    },
}


def with_defaults(cfg: dict) -> dict:
    """The run config with every encoder setting filled in (YAML numbers coerced to float)."""
    cfg = copy.deepcopy(cfg)
    for section, defaults in DEFAULTS.items():
        cfg[section] = {**defaults, **(cfg.get(section) or {})}
    for key in ("learning_rate", "warmup_ratio", "weight_decay", "grad_clip"):
        cfg["train"][key] = float(cfg["train"][key])
    return cfg


# ---------------------------------------------------------------------------------------------
# Tokenising


def encode_texts(
    texts: Sequence[str], tokenizer, max_length: int, head_tokens: int | None = None
) -> list[list[int]]:
    """Token ids with special tokens, cut to ``max_length``.

    Without ``head_tokens`` the text is cut at the end. With it, a long text keeps its first
    ``head_tokens`` tokens and its last ``max_length - head_tokens - specials`` tokens.
    """
    if head_tokens is None:
        return tokenizer(list(texts), truncation=True, max_length=max_length)["input_ids"]
    # The special tokens around a single text, e.g. [CLS] ... [SEP] or <s> ... </s>
    specials = tokenizer("")["input_ids"]
    prefix, suffix = specials[:1], specials[1:]
    tail_tokens = max_length - head_tokens - len(specials)
    out = []
    for ids in tokenizer(list(texts), add_special_tokens=False, verbose=False)["input_ids"]:
        if len(ids) > head_tokens + tail_tokens:
            ids = ids[:head_tokens] + ids[-tail_tokens:]
        out.append(prefix + ids + suffix)
    return out


def tokenized_split(
    df: pd.DataFrame, tokenizer, model_name: str, max_length: int, head_tokens: int | None
) -> Dataset:
    """One split as a ``datasets.Dataset`` of input_ids and labels, cached in ``CACHE_DIR``.

    The cache key covers the rows (ids and texts), the tokenizer and the truncation settings,
    so a change to any of them re-tokenises.
    """
    key = hashlib.sha1(
        "\n".join([model_name, str(max_length), str(head_tokens), *df["id"], *df["text"]]).encode()
    ).hexdigest()[:16]
    path = CACHE_DIR / "tokenized" / key
    if path.exists():
        return Dataset.load_from_disk(str(path))
    ds = Dataset.from_dict(
        {
            "input_ids": encode_texts(df["text"].tolist(), tokenizer, max_length, head_tokens),
            "labels": df["label"].tolist(),
        }
    )
    ds.save_to_disk(str(path))
    return ds


def stratified_subset(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """About ``n`` rows with the same label balance (for quick debug runs)."""
    if n >= len(df):
        return df
    return df.groupby("label").sample(frac=n / len(df), random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------------------------------------
# Training


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def count_trainable(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def predict_proba(model, dataset: Dataset, collator, batch_size: int, bf16: bool) -> np.ndarray:
    """Probabilities in the dataset's row order.

    Not ``Trainer.predict``: with ``train_sampling_strategy="group_by_length"``, transformers 5
    also length-groups evaluation batches, so its predictions come back shuffled.
    """
    model.eval()
    loader = DataLoader(
        dataset.remove_columns("labels"), batch_size=batch_size, shuffle=False, collate_fn=collator
    )
    out = []
    for batch in loader:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        with torch.autocast(model.device.type, dtype=torch.bfloat16, enabled=bf16):
            logits = model(**batch).logits
        out.append(torch.softmax(logits.float(), dim=-1).cpu())
    return torch.cat(out).numpy()


def train_encoder(cfg: dict) -> dict:
    """Fine-tune one encoder run from a config dict (see ``configs/encoder/``)."""
    cfg = with_defaults(cfg)
    run, data, mcfg, tcfg = cfg["run"], cfg["data"], cfg["model"], cfg["train"]
    if mcfg["method"] != "full":
        raise NotImplementedError(f"model.method {mcfg['method']!r}: LoRA is T22")
    seed = int(run["seed"])
    hf_set_seed(seed)
    info: DatasetInfo = get_dataset(data["dataset"])
    splits = load_splits(info.name, variant=data.get("variant", "raw"))
    if data["subset"]:
        splits["train"] = stratified_subset(splits["train"], int(data["subset"]), seed)
    max_length = int(data["max_length"] or info.max_length)
    batch_size = int(tcfg["batch_size"] or (32 if max_length < 384 else 16))
    cuda = torch.cuda.is_available()
    bf16 = tcfg["precision"] == "bf16" and cuda and torch.cuda.is_bf16_supported()

    name = mcfg["name"]
    tokenizer = AutoTokenizer.from_pretrained(name)
    datasets = {
        split: tokenized_split(df, tokenizer, name, max_length, data["head_tokens"])
        for split, df in splits.items()
    }
    model = AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=info.num_labels, id2label=info.id2label, label2id=info.label2id
    )
    trainable = count_trainable(model)
    print(
        f"{run['name']} seed {seed}: {name} on {info.name} ({info.num_labels} labels), "
        f"{len(splits['train']):,} train rows, max {max_length} tokens, batch {batch_size}, "
        f"{trainable:,} trainable parameters, {'bf16' if bf16 else 'fp32'}"
    )

    run_dir = OUTPUT_DIR / run["name"] / f"seed{seed}"
    args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        seed=seed,
        learning_rate=tcfg["learning_rate"],
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=int(tcfg["eval_batch_size"]),
        gradient_accumulation_steps=int(tcfg["grad_accum"]),
        num_train_epochs=int(tcfg["epochs"]),
        warmup_steps=tcfg["warmup_ratio"],  # transformers 5: a float below 1 is a ratio
        lr_scheduler_type="linear",
        weight_decay=tcfg["weight_decay"],
        max_grad_norm=tcfg["grad_clip"],
        bf16=bf16,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,  # with load_best_model_at_end the best checkpoint is also kept
        save_only_model=True,  # no optimizer state: the run is never resumed
        train_sampling_strategy="group_by_length",  # batches of similar length, less padding
        dataloader_num_workers=int(tcfg["num_workers"]),
        report_to="none",
        disable_tqdm=not sys.stderr.isatty(),  # keep log files on the cluster readable
    )

    def val_metrics(pred) -> dict:
        m = compute_metrics(pred.label_ids, softmax(pred.predictions), info)
        return {k: m[k] for k in ("macro_f1", "accuracy", "weighted_f1", "ece")}

    collator = DataCollatorWithPadding(tokenizer)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=val_metrics,
        callbacks=[EarlyStoppingCallback(int(tcfg["early_stopping_patience"]))],
    )
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    trainer.train()
    train_seconds = time.perf_counter() - start
    peak_mb = torch.cuda.max_memory_allocated() / 2**20 if cuda else None

    history = [
        {
            "epoch": round(h["epoch"]),
            "val_macro_f1": h["eval_macro_f1"],
            "val_accuracy": h["eval_accuracy"],
            "val_loss": h["eval_loss"],
        }
        for h in trainer.state.log_history
        if "eval_macro_f1" in h
    ]
    best = max(history, key=lambda h: h["val_macro_f1"])

    probs = {
        split: predict_proba(
            trainer.model, datasets[split], collator, int(tcfg["eval_batch_size"]), bf16
        )
        for split in ("validation", "test")
    }
    # Guard against rows and probabilities getting out of step: re-scoring the validation set
    # must reproduce the best epoch's score (up to bf16 noise from different padding).
    rescored = compute_metrics(splits["validation"]["label"], probs["validation"], info)
    if abs(rescored["macro_f1"] - best["val_macro_f1"]) > 0.01:
        print(
            f"  WARNING: validation macro-F1 is {rescored['macro_f1']:.4f} after training but "
            f"{best['val_macro_f1']:.4f} during it; check the prediction order"
        )
    (run_dir / "model").mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(run_dir / "model"))  # the best epoch, with its tokenizer
    shutil.rmtree(run_dir / "checkpoints", ignore_errors=True)  # model/ holds the best one
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    model = trainer.model.float().eval()  # timed in fp32, the way the app runs it (spec S6)

    @torch.no_grad()
    def predict_texts(texts: list[str]) -> np.ndarray:
        ids = encode_texts(texts, tokenizer, max_length, data["head_tokens"])
        inputs = {"input_ids": torch.tensor(ids, device=model.device)}
        return torch.softmax(model(**inputs).logits, dim=-1).cpu().numpy()

    latency = measure_latency_ms(predict_texts, splits["test"]["text"].tolist()[:200])
    summary = {
        "run": run["name"],
        "seed": seed,
        "dataset": info.name,
        "variant": data.get("variant", "raw"),
        "family": "encoder",
        "model_name": name,
        "best_epoch": best["epoch"],
        "best_val_macro_f1": best["val_macro_f1"],
        "epochs_run": len(history),
        "trainable_params": trainable,
        "train_seconds": train_seconds,
        "peak_gpu_mem_mb": peak_mb,
        "latency_ms": latency,
        "device": torch.cuda.get_device_name() if cuda else "cpu",
        "history": history,
    }
    with open(run_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    for h in history:
        print(f"  epoch {h['epoch']}  val macro-F1 {h['val_macro_f1']:.4f}")
    print(
        f"  done in {train_seconds:.0f} s; best val macro-F1 {best['val_macro_f1']:.4f} "
        f"(epoch {best['epoch']}); {latency:.1f} ms per claim"
    )
    log_run(
        cfg,
        info,
        {split: (splits[split], probs[split]) for split in probs},
        extra={
            k: summary[k]
            for k in (
                "trainable_params",
                "train_seconds",
                "peak_gpu_mem_mb",
                "latency_ms",
                "best_epoch",
                "device",
            )
        },
    )
    return summary
