"""Fine-tuned transformer encoders (owners M1 + M2, T06; LoRA T26).

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

Two settings the Optuna search (T18) varies: ``model.dropout`` sets every dropout rate of the
encoder and its head, and ``model.freeze_layers: k`` freezes the embeddings and the bottom k
layers.

``model.method: lora`` (T26) trains low-rank adapters instead of every weight: rank
``lora_r``, scale ``lora_alpha`` (default 2 x rank), on the query/key/value projections
(``lora_targets: attention``) or on every linear layer of the encoder (``all_linear``). The
classification head (and DeBERTa's pooler) still trains in full. The adapters are merged into
the weights before ``model/`` is saved, so a LoRA model loads and runs like any other.

``model.method: qlora`` (T27, configs in ``configs/llm/``) is the same LoRA on a causal LM with
a classification head (Qwen3-8B as ``Qwen3ForSequenceClassification``) whose weights are
loaded in 4-bit NF4 and frozen, with gradient checkpointing. A 4-bit model cannot be merged
cleanly, so ``model/`` holds the adapters and the head only; load them with
``load_qlora(model_dir)``. ``data.train_rows: n`` trains on a stratified sample of n rows (ISOT's
15k subset); unlike ``subset``, it is a real run, not a debug run.

``train.class_weights: balanced`` (T29 ablation) weights the cross-entropy loss by inverse
class frequency in the training split, n / (classes x count), so rare labels (LIAR's
pants-fire, PUBHEALTH's unproven) count as much as common ones.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from transformers import set_seed as hf_set_seed

from factcheck.data import get_dataset
from factcheck.data.registry import DatasetInfo
from factcheck.evaluate import compute_metrics, log_run, measure_latency_ms
from factcheck.inputs import load_inputs
from factcheck.paths import CACHE_DIR, OUTPUT_DIR

# Plan Section 10, base-encoder column. A run config only needs to list what it changes.
DEFAULTS = {
    "data": {
        "max_length": None,  # tokens; None = the dataset's max_length from the registry
        "head_tokens": None,  # keep this many head tokens and fill the rest from the tail
        "subset": None,  # train on a stratified sample of this many rows (debug runs only)
        "train_rows": None,  # the same, for a real run (T27: ISOT's 15k QLoRA subset)
    },
    "model": {
        "method": "full",
        "dropout": None,  # None = the pretrained model's own rates (0.1 for most encoders)
        "freeze_layers": 0,  # freeze the embeddings and this many bottom layers
        "lora_r": 8,  # method lora only (T26)
        "lora_alpha": None,  # None = 2 x lora_r
        "lora_dropout": 0.1,
        "lora_targets": "attention",  # attention (query/key/value) or all_linear
    },
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
        "class_weights": None,  # None or "balanced": inverse-frequency loss weights (T29)
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
    texts: Sequence[str],
    tokenizer,
    max_length: int,
    head_tokens: int | None = None,
    pairs: Sequence[str] | None = None,
) -> list[list[int]]:
    """Token ids with special tokens, cut to ``max_length``.

    Without ``head_tokens`` the text is cut at the end. With it, a long text keeps its first
    ``head_tokens`` tokens and its last ``max_length - head_tokens - specials`` tokens.
    With ``pairs`` (evidence, T15) each text is encoded with its pair, separated the model's own
    way, and only the pair is cut, from its end (the least similar evidence goes first).
    """
    if pairs is not None:
        if head_tokens is not None:
            raise ValueError("head_tokens is for long single texts (ISOT), not text pairs")
        encoded = tokenizer(
            list(texts), list(pairs), truncation="only_second", max_length=max_length
        )
        return encoded["input_ids"]
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

    The cache key covers the rows (ids, texts and evidence pairs), the tokenizer and the
    truncation settings, so a change to any of them re-tokenises.
    """
    pairs = df["text_pair"].tolist() if "text_pair" in df else None
    key = hashlib.sha1(
        "\n".join(
            [model_name, str(max_length), str(head_tokens), *df["id"], *df["text"], *(pairs or [])]
        ).encode()
    ).hexdigest()[:16]
    path = CACHE_DIR / "tokenized" / key
    if path.exists():
        return Dataset.load_from_disk(str(path))
    ds = Dataset.from_dict(
        {
            "input_ids": encode_texts(
                df["text"].tolist(), tokenizer, max_length, head_tokens, pairs
            ),
            "labels": df["label"].tolist(),
        }
    )
    # Written to a temporary folder and renamed, so parallel workers (several seeds, or the T18
    # Optuna workers) never read a half-written cache.
    tmp = path.with_name(f"{key}.tmp{os.getpid()}")
    ds.save_to_disk(str(tmp))
    try:
        tmp.rename(path)
    except OSError:  # another worker saved the same split first
        shutil.rmtree(tmp, ignore_errors=True)
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


# Dropout rates in the configs of the encoders we use: BERT, RoBERTa and DeBERTa-v3 (hidden,
# attention, head), DistilBERT, and ModernBERT.
DROPOUT_KEYS = (
    "hidden_dropout_prob",
    "attention_probs_dropout_prob",
    "pooler_dropout",
    "classifier_dropout",
    "dropout",
    "attention_dropout",
    "seq_classif_dropout",
    "embedding_dropout",
    "mlp_dropout",
)


def load_model(name: str, info: DatasetInfo, dropout: float | None = None):  # noqa: ANN201
    """The pretrained encoder with a fresh classification head, always with fp32 weights.

    transformers 5 loads weights in the dtype they were saved in, and DeBERTa-v3 is saved in
    fp16: AdamW on fp16 weights turns them into inf/NaN on the first step (eps 1e-8 is 0 in
    fp16). bf16 is applied by autocast during training, so the weights stay fp32 here.
    ``dropout`` replaces every dropout rate the model's config has.
    """
    config = AutoConfig.from_pretrained(
        name, num_labels=info.num_labels, id2label=info.id2label, label2id=info.label2id
    )
    if dropout is not None:
        for key in DROPOUT_KEYS:
            if hasattr(config, key):
                setattr(config, key, float(dropout))
    return AutoModelForSequenceClassification.from_pretrained(
        name, config=config, dtype=torch.float32
    )


def load_4bit_model(name: str, info: DatasetInfo, pad_token_id: int):  # noqa: ANN201
    """A causal LM with a fresh classification head, weights in 4-bit NF4 (T27, QLoRA).

    Computation runs in bf16. The head scores the last real token, so the model has to know
    the pad token.
    """
    from peft import prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        name,
        num_labels=info.num_labels,
        id2label=info.id2label,
        label2id=info.label2id,
        pad_token_id=pad_token_id,
        quantization_config=quant,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    return prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )


def load_qlora(model_dir: str | Path):  # noqa: ANN201
    """A saved QLoRA run (``outputs/<run>/seed<N>/model``): the 4-bit base model with its
    trained adapters and head, ready for inference, and its tokenizer."""
    from peft import PeftConfig, PeftModel

    from factcheck.data import get_dataset

    model_dir = Path(model_dir)
    with open(model_dir.parent / "config.yaml", encoding="utf-8") as f:
        info = get_dataset(yaml.safe_load(f)["data"]["dataset"])
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    base_name = PeftConfig.from_pretrained(model_dir).base_model_name_or_path
    base = load_4bit_model(base_name, info, tokenizer.pad_token_id)
    return PeftModel.from_pretrained(base, model_dir).eval(), tokenizer


def freeze_bottom_layers(model: torch.nn.Module, k: int) -> None:
    """Freeze the embeddings and the bottom ``k`` transformer layers (``k = 0`` freezes nothing)."""
    if not k:
        return
    base = model.base_model
    # BERT, RoBERTa and DeBERTa keep them in encoder.layer, DistilBERT in transformer.layer,
    # ModernBERT in layers.
    for path in ("encoder.layer", "transformer.layer", "layers"):
        layers = base
        for attr in path.split("."):
            layers = getattr(layers, attr, None)
        if isinstance(layers, torch.nn.ModuleList):
            break
    else:
        raise ValueError(f"freeze_layers: no transformer layers found in {type(base).__name__}")
    if k > len(layers):
        raise ValueError(f"freeze_layers is {k} but the model has only {len(layers)} layers")
    for module in (base.embeddings, *layers[:k]):
        for p in module.parameters():
            p.requires_grad = False


# Query, key and value projections: BERT and RoBERTa, DeBERTa-v3, DistilBERT, ModernBERT (fused).
ATTENTION_MODULES = (
    "query",
    "key",
    "value",
    "query_proj",
    "key_proj",
    "value_proj",
    "q_lin",
    "k_lin",
    "v_lin",
    "Wqkv",
    "q_proj",  # Qwen3 (T27)
    "k_proj",
    "v_proj",
)
# Layers on top of the encoder that start untrained, so they train in full next to LoRA.
HEAD_MODULES = ("pooler", "pre_classifier", "head", "classifier", "score")


def add_lora(model: torch.nn.Module, mcfg: dict) -> torch.nn.Module:
    """Wrap the encoder in LoRA adapters (T26); everything but the adapters and the head is
    frozen. Targets are full module names inside the encoder, so a head layer that happens to
    share a name (RoBERTa's ``classifier.dense``) is never given an adapter."""
    from peft import LoraConfig, TaskType, get_peft_model

    targets = mcfg["lora_targets"]
    if targets not in ("attention", "all_linear"):
        raise ValueError(f"lora_targets must be attention or all_linear, not {targets!r}")
    prefix = model.base_model_prefix
    names = [
        f"{prefix}.{name}"
        for name, module in getattr(model, prefix).named_modules()
        if isinstance(module, torch.nn.Linear)
        and (targets == "all_linear" or name.rsplit(".", 1)[-1] in ATTENTION_MODULES)
    ]
    if not names:
        raise ValueError(f"lora_targets {targets}: no such layers in {type(model).__name__}")
    rank = int(mcfg["lora_r"])
    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=rank,
        lora_alpha=int(mcfg["lora_alpha"] or 2 * rank),
        lora_dropout=float(mcfg["lora_dropout"]),
        target_modules=names,
        modules_to_save=[m for m in HEAD_MODULES if m != prefix and hasattr(model, m)],
    )
    return get_peft_model(model, config)


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


def balanced_weights(labels: Sequence[int], num_labels: int) -> torch.Tensor:
    """Inverse-frequency class weights, n / (classes x count); 1.0 for every class if balanced."""
    counts = np.bincount(np.asarray(labels), minlength=num_labels).astype(np.float64)
    if (counts == 0).any():
        raise ValueError(f"class_weights: a class has no training rows ({counts.tolist()})")
    return torch.tensor(counts.sum() / (num_labels * counts), dtype=torch.float32)


class WeightedLossTrainer(Trainer):
    """``Trainer`` with a class-weighted cross-entropy loss (T29)."""

    def __init__(self, *args, class_weights: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = torch.nn.functional.cross_entropy(
            outputs.logits.float(), labels, weight=self.class_weights.to(labels.device)
        )
        return (loss, outputs) if return_outputs else loss


def train_encoder(cfg: dict, callbacks: Sequence = (), save: bool = True) -> dict:
    """Fine-tune one encoder run from a config dict (see ``configs/encoder/``).

    ``callbacks`` are extra ``transformers`` trainer callbacks. With ``save=False`` (Optuna
    trials, T18) the run only trains and returns its validation scores: no checkpoint, no test
    predictions and nothing in results/runs.csv.
    """
    cfg = with_defaults(cfg)
    run, data, mcfg, tcfg = cfg["run"], cfg["data"], cfg["model"], cfg["train"]
    if mcfg["method"] not in ("full", "lora", "qlora"):
        raise ValueError(f"model.method must be full, lora or qlora, not {mcfg['method']!r}")
    qlora = mcfg["method"] == "qlora"
    seed = int(run["seed"])
    hf_set_seed(seed)
    info: DatasetInfo = get_dataset(data["dataset"])
    splits = load_inputs(cfg)  # load_splits + the features switches (metadata, evidence; T15)
    if data["subset"]:
        splits["train"] = stratified_subset(splits["train"], int(data["subset"]), seed)
    if data["train_rows"]:
        splits["train"] = stratified_subset(splits["train"], int(data["train_rows"]), seed)
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
    if qlora:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = load_4bit_model(name, info, tokenizer.pad_token_id)
    else:
        model = load_model(name, info, mcfg["dropout"])
    freeze_bottom_layers(model, int(mcfg["freeze_layers"]))
    if mcfg["method"] in ("lora", "qlora"):
        model = add_lora(model, mcfg)
    trainable = count_trainable(model)
    print(
        f"{run['name']} seed {seed}: {name} on {info.name} ({info.num_labels} labels), "
        f"{len(splits['train']):,} train rows, max {max_length} tokens, batch {batch_size}, "
        f"{trainable:,} trainable parameters, {'bf16' if bf16 else 'fp32'}"
    )

    run_dir = OUTPUT_DIR / run["name"] / f"seed{seed}"
    # Trials write nothing under outputs/, so they can never touch a real run of the same name
    checkpoints = run_dir / "checkpoints" if save else Path(tempfile.mkdtemp(prefix="trial_"))
    args = TrainingArguments(
        output_dir=str(checkpoints),
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
        save_strategy="epoch" if save else "no",
        logging_strategy="epoch",
        load_best_model_at_end=save,
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
    weighting = tcfg["class_weights"]
    if weighting not in (None, "balanced"):
        raise ValueError(f"train.class_weights must be balanced or left out, not {weighting!r}")
    trainer_class, trainer_kwargs = Trainer, {}
    if weighting:
        weights = balanced_weights(splits["train"]["label"], info.num_labels)
        shown = dict(zip(info.labels, weights.round(decimals=3).tolist(), strict=True))
        print(f"  class weights {shown}")
        trainer_class, trainer_kwargs = WeightedLossTrainer, {"class_weights": weights}
    trainer = trainer_class(
        **trainer_kwargs,
        model=model,
        args=args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=val_metrics,
        callbacks=[EarlyStoppingCallback(int(tcfg["early_stopping_patience"])), *callbacks],
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
    summary = {
        "run": run["name"],
        "seed": seed,
        "dataset": info.name,
        "variant": data.get("variant", "raw"),
        "family": mcfg.get("family", "encoder"),
        "model_name": name,
        "best_epoch": best["epoch"],
        "best_val_macro_f1": best["val_macro_f1"],
        "epochs_run": len(history),
        "trainable_params": trainable,
        "train_seconds": train_seconds,
        "peak_gpu_mem_mb": peak_mb,
    }
    if not save:
        shutil.rmtree(checkpoints, ignore_errors=True)
        return {**summary, "history": history}

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
    if mcfg["method"] == "lora":  # fold the adapters into the weights: one plain model to save
        trainer.model = trainer.model.merge_and_unload()
    (run_dir / "model").mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(run_dir / "model"))  # the best epoch, with its tokenizer
    shutil.rmtree(run_dir / "checkpoints", ignore_errors=True)  # model/ holds the best one
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # Timed in fp32, the way the app runs it (spec S6); a 4-bit QLoRA model runs as it is.
    model = trainer.model.eval() if qlora else trainer.model.float().eval()

    @torch.no_grad()
    def predict_texts(items: list) -> np.ndarray:
        """Items are texts, or (text, evidence) tuples when the run uses evidence."""
        if isinstance(items[0], str):
            texts, pairs = items, None
        else:
            texts, pairs = [t for t, _ in items], [e for _, e in items]
        ids = encode_texts(texts, tokenizer, max_length, data["head_tokens"], pairs)
        inputs = {"input_ids": torch.tensor(ids, device=model.device)}
        return torch.softmax(model(**inputs).logits, dim=-1).cpu().numpy()

    test = splits["test"].head(200)
    if "text_pair" in test:
        items = list(zip(test["text"], test["text_pair"], strict=True))
    else:
        items = test["text"].tolist()
    latency = measure_latency_ms(predict_texts, items)
    summary.update(
        latency_ms=latency,
        device=torch.cuda.get_device_name() if cuda else "cpu",
        history=history,
    )
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
