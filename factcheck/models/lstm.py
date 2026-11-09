"""Bi-LSTM baseline with frozen GloVe 300-d (owners M1 + M2, T05).

The sequence-model floor of plan Section 8.1: lowercase word tokens, a vocabulary built from
the training split, frozen 300-d GloVe vectors, a 2-layer bidirectional LSTM and a linear
classifier on the final forward and backward states. Settings follow plan Section 10: Adam at
1e-3 (constant), batch 64, up to 15 epochs, dropout 0.3, grad clip 1.0, fp32, early stopping on
validation macro-F1 with the best epoch kept.

    python -m factcheck.train --config configs/lstm/lstm_liar.yaml

T55 adds the other recurrent cells next to it: ``model.cell`` is ``lstm`` (default), ``gru``
or ``rnn`` (plain tanh RNN), ``model.bidirectional: false`` reads left to right only, and
``train.grad_clip: null`` turns clipping off. Every run logs gradient sizes in
``train_summary.json``: per epoch, the total gradient norm before clipping (mean, median, max)
and the share of steps that clipping cut, and, on a fixed batch of training claims before
training and after each epoch, the mean gradient of the loss at each input word by its
distance from the last word (``grad_by_distance``), which shows vanishing gradients directly.

Each run writes, under ``outputs/<name>/seed<N>/``:

    config.yaml              the full config used (defaults filled in)
    model/model.pt           weights, vocabulary and labels (load with ``LSTMPredictor``)
    train_summary.json       per-epoch validation macro-F1, trainable parameters, time, memory

and logs the validation and test scores to ``results/runs.csv`` through
``factcheck.evaluate.log_run``, which also saves the per-row probabilities.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import re
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader

from factcheck.data import get_dataset
from factcheck.evaluate import log_run, measure_latency_ms
from factcheck.inputs import load_inputs, single_text
from factcheck.paths import CACHE_DIR, OUTPUT_DIR, RAW_DIR

PAD, UNK = "<pad>", "<unk>"  # ids 0 and 1; both have zero vectors
CELLS = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn": nn.RNN}
DISTANCES = (0, 1, 2, 4, 8, 16, 32, 64)  # words from the last one, for grad_by_distance

# Plan Section 10, Bi-LSTM column. A run config only needs to list what it changes.
DEFAULTS = {
    "model": {
        "embedding_path": str(RAW_DIR / "glove" / "glove.6B.300d.txt"),
        "embedding_dim": 300,
        "hidden_size": 256,  # about 2.7M trainable parameters with 2 layers
        "num_layers": 2,
        "dropout": 0.3,
        "cell": "lstm",  # T55: lstm, gru or rnn
        "bidirectional": True,
    },
    "train": {
        "learning_rate": 1.0e-3,
        "batch_size": 64,
        "eval_batch_size": 256,
        "epochs": 15,
        "early_stopping_patience": 3,
        "grad_clip": 1.0,  # null: no clipping (T55)
        "grad_probe_size": 256,  # training claims in the fixed batch for grad_by_distance
        "min_freq": 1,  # a train word needs this many occurrences (and a GloVe vector) to get an id
    },
}


# ---------------------------------------------------------------------------------------------
# Text to ids


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens split like GloVe's training text ("don't" -> "do", "n't")."""
    text = re.sub(r"n't\b", " n't", text.lower())
    return re.findall(r"n't|'\w+|\w+|[^\w\s]", text)


def build_vocab(texts: Iterable[str], glove_words: set[str], min_freq: int = 1) -> list[str]:
    """Train words that have a GloVe vector, most frequent first, after PAD and UNK."""
    counts = Counter(tok for text in texts for tok in tokenize(text))
    words = [w for w, c in counts.most_common() if c >= min_freq and w in glove_words]
    return [PAD, UNK, *words]


def encode(text: str, word2id: dict[str, int], max_length: int) -> list[int]:
    """Token ids, cut to the first ``max_length`` words; never empty (UNK for blank text)."""
    ids = [word2id.get(tok, 1) for tok in tokenize(text)[:max_length]]
    return ids or [1]


def glove_words_and_vectors(
    path: Path, wanted: set[str] | None = None
) -> tuple[set[str], dict[str, np.ndarray]]:
    """Every word in the GloVe file, and the vectors of the ``wanted`` ones (one pass)."""
    words: set[str] = set()
    vectors: dict[str, np.ndarray] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            word, rest = line.rstrip("\n").split(" ", 1)
            words.add(word)
            if wanted is not None and word in wanted:
                vectors[word] = np.asarray(rest.split(" "), dtype=np.float32)
    return words, vectors


def vocab_and_embeddings(
    train_texts: Sequence[str], glove_path: Path, dim: int, min_freq: int
) -> tuple[list[str], torch.Tensor]:
    """Vocabulary and its frozen GloVe matrix, cached in ``CACHE_DIR/glove/``.

    Reading the 1 GB GloVe file takes about a minute, so the result is cached under a key
    made from the training tokens, ``min_freq`` and the GloVe file; any change rebuilds it.
    """
    candidates = sorted({tok for text in train_texts for tok in tokenize(text)})
    key = hashlib.sha1(
        "\n".join([*candidates, f"min_freq={min_freq}", glove_path.name]).encode()
    ).hexdigest()[:16]
    cache = CACHE_DIR / "glove" / f"{key}.pt"
    if cache.exists():
        saved = torch.load(cache)
        return saved["vocab"], saved["embeddings"]

    print(f"Reading {glove_path} (about a minute; cached afterwards)")
    glove_words, vectors = glove_words_and_vectors(glove_path, set(candidates))
    vocab = build_vocab(train_texts, glove_words, min_freq)
    matrix = np.zeros((len(vocab), dim), dtype=np.float32)
    for i, word in enumerate(vocab[2:], start=2):
        matrix[i] = vectors[word]
    embeddings = torch.from_numpy(matrix)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"vocab": vocab, "embeddings": embeddings}, cache)
    return vocab, embeddings


# ---------------------------------------------------------------------------------------------
# Model


class RecurrentClassifier(nn.Module):
    """Frozen embeddings -> 2-layer recurrent net -> final state(s) of the top layer -> linear.

    The defaults are the T05 Bi-LSTM ([last forward state; last backward state]); T55 swaps
    in a GRU or a plain RNN and can read one direction only (the last forward state).
    """

    def __init__(
        self,
        embeddings: torch.Tensor,
        num_labels: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        cell: str = "lstm",
        bidirectional: bool = True,
    ):
        super().__init__()
        if cell not in CELLS:
            raise ValueError(f"model.cell must be one of {sorted(CELLS)}, not {cell!r}")
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=True, padding_idx=0)
        self.lstm = CELLS[cell](  # named lstm so T05 checkpoints keep loading
            embeddings.shape[1],
            hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear((2 if bidirectional else 1) * hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.classify(self.embedding(input_ids), lengths)

    def classify(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Logits from embedded inputs (batch, time, dim); split out for ``gradient_by_distance``."""
        packed = pack_padded_sequence(
            self.dropout(x), lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.lstm(packed)
        h = (
            h[0] if isinstance(h, tuple) else h
        )  # LSTM returns (h, c); h: (layers * dirs, batch, hidden)
        final = torch.cat([h[-2], h[-1]], dim=1) if self.bidirectional else h[-1]
        return self.classifier(self.dropout(final))


BiLSTMClassifier = RecurrentClassifier  # the T05 name, still used by the tests and SHAP


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def gradient_by_distance(
    model: RecurrentClassifier, x: torch.Tensor, lengths: torch.Tensor, y: torch.Tensor
) -> dict[str, float]:
    """Mean size of the loss gradient at each input word, by its distance from the last word.

    The classifier reads the final state, so in a plain RNN the gradient reaching early words
    shrinks fast with distance (vanishing gradients); the GRU and LSTM gates keep it larger.
    Per-claim loss (summed, not averaged), no dropout, and no cuDNN, whose RNN backward pass
    only runs in training mode. Parameter gradients are not touched.
    """
    model.eval()
    with torch.backends.cudnn.flags(enabled=False):
        emb = model.embedding(x).detach().requires_grad_()
        loss = nn.functional.cross_entropy(model.classify(emb, lengths), y, reduction="sum")
        (grad,) = torch.autograd.grad(loss, emb)
    norms = grad.norm(dim=2)  # (batch, time)
    pos_last = lengths.to(norms.device) - 1
    out = {}
    for d in DISTANCES:
        rows = (pos_last >= d).nonzero().squeeze(1)
        if len(rows):
            out[str(d)] = norms[rows, pos_last[rows] - d].mean().item()
    return out


def collate(batch: Sequence[tuple[list[int], int]]):
    """(ids, label) pairs -> padded ids, lengths, labels."""
    seqs, ys = zip(*batch, strict=True)
    return (
        pad_sequence([torch.tensor(s) for s in seqs], batch_first=True, padding_value=0),
        torch.tensor([len(s) for s in seqs]),
        torch.tensor(ys),
    )


def make_loader(ids: list[list[int]], labels: Sequence[int], batch_size: int, shuffle: bool):
    return DataLoader(
        list(zip(ids, labels, strict=True)),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
    )


@torch.no_grad()
def predict_proba(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    probs = [torch.softmax(model(x.to(device), lengths), dim=1).cpu() for x, lengths, _ in loader]
    return torch.cat(probs).numpy()


# ---------------------------------------------------------------------------------------------
# Training


def with_defaults(cfg: dict) -> dict:
    """The run config with every Bi-LSTM setting filled in (YAML numbers coerced to float/int)."""
    cfg = copy.deepcopy(cfg)
    for section, defaults in DEFAULTS.items():
        cfg[section] = {**defaults, **(cfg.get(section) or {})}
    cfg["train"]["learning_rate"] = float(cfg["train"]["learning_rate"])
    if cfg["train"]["grad_clip"] is not None:
        cfg["train"]["grad_clip"] = float(cfg["train"]["grad_clip"])
    cfg["model"]["dropout"] = float(cfg["model"]["dropout"])
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_lstm(cfg: dict) -> dict:
    """Train one Bi-LSTM run from a config dict (see ``configs/lstm/``); returns the summary."""
    cfg = with_defaults(cfg)
    run, data, mcfg, tcfg = cfg["run"], cfg["data"], cfg["model"], cfg["train"]
    seed = int(run["seed"])
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = get_dataset(data["dataset"])
    splits = {  # load_splits + the features switches (T15), as one sequence per row
        split: df.assign(text=single_text(df)) for split, df in load_inputs(cfg).items()
    }
    max_length = int(data.get("max_length") or info.max_length)

    vocab, embeddings = vocab_and_embeddings(
        splits["train"]["text"].tolist(),
        Path(mcfg["embedding_path"]),
        mcfg["embedding_dim"],
        tcfg["min_freq"],
    )
    word2id = {w: i for i, w in enumerate(vocab)}
    loaders = {
        split: make_loader(
            [encode(t, word2id, max_length) for t in df["text"]],
            df["label"].tolist(),
            tcfg["batch_size"] if split == "train" else tcfg["eval_batch_size"],
            shuffle=split == "train",
        )
        for split, df in splits.items()
    }
    train_tokens = [tok for t in splits["train"]["text"] for tok in tokenize(t)]
    oov = sum(tok not in word2id for tok in train_tokens) / max(len(train_tokens), 1)

    model = RecurrentClassifier(
        embeddings,
        info.num_labels,
        mcfg["hidden_size"],
        mcfg["num_layers"],
        mcfg["dropout"],
        mcfg["cell"],
        mcfg["bidirectional"],
    ).to(device)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=tcfg["learning_rate"]
    )
    loss_fn = nn.CrossEntropyLoss()
    clip = tcfg["grad_clip"]
    # Fixed batch: the first training claims. Not through a DataLoader, whose iterator draws a
    # seed from the global generator and would change the shuffle order of every later epoch.
    first = splits["train"].iloc[: tcfg["grad_probe_size"]]
    probe_x, probe_lengths, probe_y = collate(
        [
            (encode(t, word2id, max_length), y)
            for t, y in zip(first["text"], first["label"], strict=True)
        ]
    )
    probe = (probe_x.to(device), probe_lengths, probe_y.to(device))
    grad_by_distance_init = gradient_by_distance(model, *probe)
    print(
        f"{run['name']} seed {seed}: {mcfg['cell']}"
        f"{' bidirectional' if mcfg['bidirectional'] else ''}, grad clip {clip}; "
        f"{info.name} ({info.num_labels} labels), vocab {len(vocab):,} "
        f"({oov:.1%} of train tokens unknown), {count_trainable(model):,} trainable parameters, {device}"
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    history, best_f1, best_epoch, best_state, waited = [], -1.0, 0, None, 0
    val_labels = splits["validation"]["label"].to_numpy()
    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        total_loss, n, grad_norms = 0.0, 0, []
        for x, lengths, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x, lengths), y)
            loss.backward()
            # the norm before clipping; with clipping off, an infinite limit only measures it
            norm = nn.utils.clip_grad_norm_(model.parameters(), clip or float("inf"))
            grad_norms.append(norm.item())
            optimizer.step()
            total_loss += loss.item() * len(y)
            n += len(y)
        val_pred = predict_proba(model, loaders["validation"], device).argmax(axis=1)
        val_f1 = f1_score(val_labels, val_pred, average="macro")
        grad_norms = np.asarray(grad_norms)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / n,
                "val_macro_f1": val_f1,
                "grad_norm_mean": float(grad_norms.mean()),
                "grad_norm_median": float(np.median(grad_norms)),
                "grad_norm_max": float(grad_norms.max()),
                "clipped_fraction": float((grad_norms > clip).mean()) if clip else 0.0,
                "grad_by_distance": gradient_by_distance(model, *probe),
            }
        )
        print(
            f"  epoch {epoch:2d}  train loss {total_loss / n:.4f}  val macro-F1 {val_f1:.4f}  "
            f"grad norm {grad_norms.mean():.3g} (max {grad_norms.max():.3g})"
        )
        if val_f1 > best_f1:
            best_f1, best_epoch, waited = val_f1, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            waited += 1
            if waited >= tcfg["early_stopping_patience"]:
                print(f"  early stop: no gain for {waited} epochs; best epoch {best_epoch}")
                break
    train_seconds = time.perf_counter() - start
    model.load_state_dict(best_state)

    run_dir = OUTPUT_DIR / run["name"] / f"seed{seed}"
    (run_dir / "model").mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocab": vocab,
            "labels": list(info.labels),
            "max_length": max_length,
            "hidden_size": mcfg["hidden_size"],
            "num_layers": mcfg["num_layers"],
            "dropout": mcfg["dropout"],
            "cell": mcfg["cell"],
            "bidirectional": mcfg["bidirectional"],
        },
        run_dir / "model" / "model.pt",
    )

    def predict_texts(texts: list[str]) -> np.ndarray:
        ids = [encode(t, word2id, max_length) for t in texts]
        return predict_proba(model, make_loader(ids, [0] * len(ids), len(ids), False), device)

    latency = measure_latency_ms(predict_texts, splits["test"]["text"].tolist()[:200])
    summary = {
        "run": run["name"],
        "seed": seed,
        "dataset": info.name,
        "variant": data.get("variant", "raw"),
        "family": "lstm",
        "cell": mcfg["cell"],
        "bidirectional": mcfg["bidirectional"],
        "grad_clip": clip,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "epochs_run": len(history),
        "trainable_params": count_trainable(model),
        "vocab_size": len(vocab),
        "train_oov_rate": oov,
        "train_seconds": train_seconds,
        "peak_gpu_mem_mb": (
            torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
        ),
        "latency_ms": latency,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "grad_by_distance_init": grad_by_distance_init,
        "history": history,
    }
    with open(run_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(
        f"  done in {train_seconds:.0f} s; best val macro-F1 {best_f1:.4f} (epoch {best_epoch}); "
        f"{latency:.1f} ms per claim"
    )
    log_run(
        cfg,
        info,
        {
            split: (splits[split], predict_proba(model, loaders[split], device))
            for split in ("validation", "test")
        },
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


# ---------------------------------------------------------------------------------------------
# Inference (app, SHAP)


class LSTMPredictor:
    """A saved Bi-LSTM as ``texts -> probabilities``, for the app and the SHAP script (T16).

    p = LSTMPredictor("outputs/lstm_liar/seed42/model/model.pt").predict_proba(["..."])
    """

    def __init__(self, path: str | Path, device: str | None = None):
        saved = torch.load(path, map_location="cpu")
        self.labels: list[str] = saved["labels"]
        self.max_length: int = saved["max_length"]
        self.word2id = {w: i for i, w in enumerate(saved["vocab"])}
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = RecurrentClassifier(
            saved["state_dict"]["embedding.weight"],
            len(self.labels),
            saved["hidden_size"],
            saved["num_layers"],
            saved["dropout"],
            saved.get("cell", "lstm"),  # T05 checkpoints predate T55
            saved.get("bidirectional", True),
        )
        self.model.load_state_dict(saved["state_dict"])
        self.model.to(self.device).eval()

    def predict_proba(self, texts: Sequence[str], batch_size: int = 256) -> np.ndarray:
        ids = [encode(t, self.word2id, self.max_length) for t in texts]
        loader = make_loader(ids, [0] * len(ids), batch_size, shuffle=False)
        return predict_proba(self.model, loader, self.device)
