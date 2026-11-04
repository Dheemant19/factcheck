"""Prompted LLM classification with label-probability scoring (owners M1 + M2, T07, T20).

Plan Section 8.4: Qwen3-8B with "thinking" off reads an instruction, the label definitions
and the claim, and we score each allowed label word instead of parsing free text. One forward
pass per claim: the probability of each label is the model's next-token probability of the
label's first token, renormalised over the labels. No invalid outputs, and real probabilities
for ROC-AUC and calibration. QLoRA (T27, ``method: qlora``) trains instead; see
``factcheck.models.encoder``.

    python -m factcheck.train --config configs/llm/qwen3_8b_zeroshot_liar.yaml

``model.method`` picks what the prompt adds to the claim (``factcheck.prompts``, T14):

    zeroshot  nothing (T07)
    knn       the ``model.k`` most similar TRAINING claims with their verdicts (T20)
    rag       the claim's ``features.evidence_k`` best evidence sentences; needs
              ``features.use_evidence: true`` and ``evidence_source`` own or pool (T20)

These runs train nothing; they score the validation and test splits and log them to
``results/runs.csv`` like any other run. ``data.llm_subset: true`` scores only ISOT's fixed
2,000-article test subset (long articles; no validation split is scored). Climate-FEVER has
only a test split. ``outputs/<name>/seed<N>/`` gets the config and a ``score_summary.json``
with the scoring time, memory and the label coverage: how much of the model's next-token
probability fell on the label tokens at all (low coverage means the model wanted to say
something else, so the prompt needs work).
"""

from __future__ import annotations

import copy
import json
import time
from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import set_seed as hf_set_seed

from factcheck.data import get_dataset
from factcheck.data.registry import DatasetInfo
from factcheck.evaluate import log_run, measure_latency_ms
from factcheck.inputs import features, load_inputs
from factcheck.paths import OUTPUT_DIR
from factcheck.prompts import (  # noqa: F401  (T14 moved PROMPTS and zero_shot_message)
    PROMPTS,
    knn_messages,
    rag_messages,
    zero_shot_message,
    zero_shot_messages,
)

DEFAULTS = {
    # max_length: claim tokens kept; None = the dataset's registry max_length
    # llm_subset: score only ISOT's fixed 2,000-article test subset
    "data": {"max_length": None, "llm_subset": False},
    "model": {"method": "zeroshot", "k": 8},  # k: retrieved examples for method knn
    "train": {"eval_batch_size": 16},
}
METHODS = ("zeroshot", "knn", "rag")

# Per-claim prompt context: kNN examples as (text, label name) pairs, or evidence sentences.
Context = Sequence[tuple[str, str]] | Sequence[str]


def with_defaults(cfg: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    for section, defaults in DEFAULTS.items():
        cfg[section] = {**defaults, **(cfg.get(section) or {})}
    return cfg


class LLMScorer:
    """A causal LM as ``texts -> label probabilities`` for one dataset.

    scorer = LLMScorer("Qwen/Qwen3-8B", get_dataset("liar"))
    probs = scorer.predict_proba(["Says the Annies List political group supports ..."])

    With ``method="knn"`` or ``"rag"``, pass one context per text (kNN examples or evidence
    sentences, most similar first) as ``predict_proba(texts, contexts=...)``.
    """

    def __init__(
        self,
        model_name: str,
        info: DatasetInfo,
        max_length: int | None = None,
        device: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        model=None,
        tokenizer=None,
        method: str = "zeroshot",
    ):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, not {method!r}")
        self.info = info
        self.method = method
        self.max_length = max_length or info.max_length
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"  # so the last position is the answer slot in a batch
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model or AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, device_map={"": device}
        )
        self.model.eval()
        self.answer_ids = self.label_token_ids()
        self.coverage: list[float] = []  # label-token probability mass, one value per text

    def label_token_ids(self) -> list[int]:
        """First token of each label word, in class-id order; they must all differ."""
        ids = [
            self.tokenizer.encode(label, add_special_tokens=False)[0] for label in self.info.labels
        ]
        if len(set(ids)) != len(ids):
            raise ValueError(
                f"labels {self.info.labels} share a first token; rename the answer words"
            )
        return ids

    def messages(self, text: str, context: Context | None = None) -> list[dict]:
        if self.method == "zeroshot":
            return zero_shot_messages(text, self.info.name)
        if context is None:
            raise ValueError(f"method {self.method!r} needs a context for every text")
        if self.method == "knn":
            return knn_messages(text, context, self.info.name)
        return rag_messages(text, context, self.info.name)

    def prompt(self, text: str, context: Context | None = None) -> str:
        """The chat-templated prompt; only the claim is cut to ``max_length`` tokens."""
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > self.max_length:
            text = self.tokenizer.decode(ids[: self.max_length])
        return self.tokenizer.apply_chat_template(
            self.messages(text, context),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    @torch.no_grad()
    def predict_proba(
        self,
        texts: Sequence[str],
        batch_size: int = 16,
        contexts: Sequence[Context] | None = None,
    ) -> np.ndarray:
        """Label probabilities in the order of ``texts`` (batches are built longest first)."""
        if contexts is not None and len(contexts) != len(texts):
            raise ValueError("contexts needs one entry per text")
        contexts = contexts if contexts is not None else [None] * len(texts)
        prompts = [self.prompt(t, c) for t, c in zip(texts, contexts, strict=True)]
        order = np.argsort([-len(p) for p in prompts], kind="stable")
        probs = np.zeros((len(texts), self.info.num_labels), dtype=np.float32)
        coverage = np.zeros(len(texts), dtype=np.float32)
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            batch = self.tokenizer(
                [prompts[i] for i in idx],
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to(self.model.device)
            # use_cache=False: one pass per prompt, so the KV cache (GBs for 2k-token ISOT
            # kNN prompts) is never needed
            logits = self.model(**batch, logits_to_keep=1, use_cache=False).logits[:, -1].float()
            label_logits = logits[:, self.answer_ids]
            probs[idx] = torch.softmax(label_logits, dim=-1).cpu().numpy()
            coverage[idx] = torch.softmax(logits, dim=-1)[:, self.answer_ids].sum(-1).cpu().numpy()
        self.coverage.extend(coverage.tolist())
        return probs


def scored_splits(cfg: dict, splits: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """The splits a run scores: validation and test, leaving out empty ones (Climate-FEVER has
    only a test split); with ``data.llm_subset``, ISOT's fixed 2,000-article test subset only."""
    if cfg["data"]["llm_subset"]:
        test = splits["test"]
        if not test["in_llm_subset"].any():
            raise ValueError(f"{cfg['data']['dataset']} has no LLM test subset (ISOT only)")
        return {"test": test[test["in_llm_subset"]].reset_index(drop=True)}
    return {split: splits[split] for split in ("validation", "test") if len(splits[split])}


def prompt_contexts(cfg: dict, df: pd.DataFrame, split: str) -> list[Context] | None:
    """Each row's kNN examples or evidence sentences for the run's method; None for zero-shot.

    Retrieval always sees the claim alone (``claim``), never a metadata prefix.
    """
    method, dataset = cfg["model"]["method"], cfg["data"]["dataset"]
    claims = df.assign(text=df["claim"])
    if method == "knn":
        from factcheck.retrieval import knn_examples

        hits = knn_examples(dataset, claims, split, int(cfg["model"]["k"]))
        return [[(hit.text, hit.label_name) for hit in row] for row in hits]
    if method == "rag":
        from factcheck.retrieval import evidence_for

        feats = features(cfg)
        return evidence_for(dataset, claims, int(feats["evidence_k"]), feats["evidence_source"])
    return None


def score_llm(cfg: dict) -> dict:
    """Score one prompted-LLM run (no training) from a config dict (see ``configs/llm/``).

    ``model.method: qlora`` (T27) trains instead: a 4-bit LLM with LoRA and a classification
    head, through the encoder trainer (``factcheck.models.encoder``).
    """
    if (cfg.get("model") or {}).get("method") == "qlora":
        from factcheck.models.encoder import train_encoder

        return train_encoder(cfg)
    cfg = with_defaults(cfg)
    run, data, mcfg, tcfg = cfg["run"], cfg["data"], cfg["model"], cfg["train"]
    method = mcfg["method"]
    if method not in METHODS:
        raise ValueError(f"model.method must be zeroshot, knn, rag or qlora, not {method!r}")
    seed = int(run["seed"])
    hf_set_seed(seed)  # scoring is deterministic; the seed only names the run
    info = get_dataset(data["dataset"])
    feats = features(cfg)
    if feats["use_evidence"] != (method == "rag"):
        raise ValueError(
            "the LLM reads evidence only in the RAG mode: use model.method: rag together with "
            "features.use_evidence: true"
        )
    if method == "knn" and data.get("variant", "raw") != "raw":
        raise ValueError("the claim index is built on the raw variant")
    splits = scored_splits(cfg, load_inputs(cfg))  # load_splits + the features switches (T15)
    cuda = torch.cuda.is_available()

    start = time.perf_counter()
    contexts = {split: prompt_contexts(cfg, df, split) for split, df in splits.items()}
    retrieval_seconds = time.perf_counter() - start
    scorer = LLMScorer(mcfg["name"], info, data["max_length"], method=method)
    print(
        f"{run['name']}: {mcfg['name']} {method} on {info.name} ({info.num_labels} labels), "
        f"answer tokens {scorer.tokenizer.convert_ids_to_tokens(scorer.answer_ids)}"
    )
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    probs, coverage = {}, {}
    for split, df in splits.items():
        scorer.coverage = []
        probs[split] = scorer.predict_proba(
            df["text"].tolist(), int(tcfg["eval_batch_size"]), contexts[split]
        )
        coverage[split] = float(np.mean(scorer.coverage))
        print(
            f"  {split}: {len(probs[split]):,} texts scored, label coverage {coverage[split]:.3f}"
        )
    score_seconds = time.perf_counter() - start
    peak_mb = torch.cuda.max_memory_allocated() / 2**20 if cuda else None
    test_texts = splits["test"]["text"].tolist()
    test_contexts = contexts["test"] or [None] * len(test_texts)
    context_of = dict(zip(test_texts, test_contexts, strict=True))
    latency = measure_latency_ms(  # the LLM pass only; retrieval is timed separately
        lambda texts: scorer.predict_proba(texts, 1, [context_of[t] for t in texts]),
        test_texts[:50],
    )

    run_dir = OUTPUT_DIR / run["name"] / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    summary = {
        "run": run["name"],
        "seed": seed,
        "dataset": info.name,
        "family": "llm",
        "model_name": mcfg["name"],
        "method": method,
        "k": int(mcfg["k"]) if method == "knn" else None,
        "evidence": (
            {"k": int(feats["evidence_k"]), "source": feats["evidence_source"]}
            if method == "rag"
            else None
        ),
        "n": {split: len(df) for split, df in splits.items()},
        "label_coverage": coverage,
        "retrieval_seconds": retrieval_seconds,
        "score_seconds": score_seconds,
        "peak_gpu_mem_mb": peak_mb,
        "latency_ms": latency,
        "device": torch.cuda.get_device_name() if cuda else "cpu",
        "prompt_example": scorer.prompt(test_texts[0], test_contexts[0]),
    }
    with open(run_dir / "score_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  done in {score_seconds:.0f} s; {latency:.0f} ms per claim")
    log_run(
        cfg,
        info,
        {split: (splits[split], probs[split]) for split in probs},
        extra={
            "trainable_params": 0,
            "peak_gpu_mem_mb": peak_mb,
            "latency_ms": latency,
            "device": summary["device"],
        },
    )
    return summary
