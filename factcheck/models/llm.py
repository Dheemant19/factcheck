"""Prompted LLM classification with label-probability scoring (owners M1 + M2, T07).

Plan Section 8.4: Qwen3-8B with "thinking" off reads an instruction, the label definitions
and the claim, and we score each allowed label word instead of parsing free text. One forward
pass per claim: the probability of each label is the model's next-token probability of the
label's first token, renormalised over the labels. No invalid outputs, and real probabilities
for ROC-AUC and calibration. Retrieved fact-checks and evidence (T20) and QLoRA (T27) build on
``LLMScorer``.

    python -m factcheck.train --config configs/llm/qwen3_8b_zeroshot_liar.yaml

Zero-shot runs train nothing; they score the validation and test splits and log them to
``results/runs.csv`` like any other run. ``outputs/<name>/seed<N>/`` gets the config and a
``score_summary.json`` with the scoring time, memory and the label coverage: how much of the
model's next-token probability fell on the label tokens at all (low coverage means the model
wanted to say something else, so the prompt needs work).
"""

from __future__ import annotations

import copy
import json
import time
from collections.abc import Sequence

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import set_seed as hf_set_seed

from factcheck.data import get_dataset
from factcheck.data.registry import DatasetInfo
from factcheck.evaluate import log_run, measure_latency_ms
from factcheck.inputs import features, load_inputs
from factcheck.paths import OUTPUT_DIR
from factcheck.prompts import PROMPTS, zero_shot_message  # noqa: F401  (T14 moved them)

DEFAULTS = {
    "data": {"max_length": None},  # claim tokens kept; None = the dataset's registry max_length
    "model": {"method": "zeroshot"},
    "train": {"eval_batch_size": 16},
}


def with_defaults(cfg: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    for section, defaults in DEFAULTS.items():
        cfg[section] = {**defaults, **(cfg.get(section) or {})}
    return cfg


class LLMScorer:
    """A causal LM as ``texts -> label probabilities`` for one dataset.

    scorer = LLMScorer("Qwen/Qwen3-8B", get_dataset("liar"))
    probs = scorer.predict_proba(["Says the Annies List political group supports ..."])
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
    ):
        self.info = info
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

    def prompt(self, text: str) -> str:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > self.max_length:
            text = self.tokenizer.decode(ids[: self.max_length])
        messages = [{"role": "user", "content": zero_shot_message(text, self.info.name)}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    @torch.no_grad()
    def predict_proba(self, texts: Sequence[str], batch_size: int = 16) -> np.ndarray:
        """Label probabilities in the order of ``texts`` (batches are built longest first)."""
        prompts = [self.prompt(t) for t in texts]
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
            logits = self.model(**batch, logits_to_keep=1).logits[:, -1].float()
            label_logits = logits[:, self.answer_ids]
            probs[idx] = torch.softmax(label_logits, dim=-1).cpu().numpy()
            coverage[idx] = torch.softmax(logits, dim=-1)[:, self.answer_ids].sum(-1).cpu().numpy()
        self.coverage.extend(coverage.tolist())
        return probs


def score_llm(cfg: dict) -> dict:
    """Score one prompted-LLM run (no training) from a config dict (see ``configs/llm/``)."""
    cfg = with_defaults(cfg)
    run, data, mcfg, tcfg = cfg["run"], cfg["data"], cfg["model"], cfg["train"]
    if mcfg["method"] != "zeroshot":
        raise NotImplementedError(
            f"model.method {mcfg['method']!r}: knn and rag are T20, qlora T27"
        )
    seed = int(run["seed"])
    hf_set_seed(seed)  # scoring is deterministic; the seed only names the run
    info = get_dataset(data["dataset"])
    if features(cfg)["use_evidence"]:
        raise NotImplementedError(
            "use_evidence for the LLM is the RAG mode (T20): build the prompt with "
            "factcheck.prompts.rag_messages and factcheck.retrieval.evidence_for"
        )
    splits = load_inputs(cfg)  # load_splits + the features switches (T15)
    cuda = torch.cuda.is_available()

    scorer = LLMScorer(mcfg["name"], info, data["max_length"])
    print(
        f"{run['name']}: {mcfg['name']} zero-shot on {info.name} ({info.num_labels} labels), "
        f"answer tokens {scorer.tokenizer.convert_ids_to_tokens(scorer.answer_ids)}"
    )
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    probs, coverage = {}, {}
    for split in ("validation", "test"):
        scorer.coverage = []
        probs[split] = scorer.predict_proba(
            splits[split]["text"].tolist(), int(tcfg["eval_batch_size"])
        )
        coverage[split] = float(np.mean(scorer.coverage))
        print(
            f"  {split}: {len(probs[split]):,} texts scored, label coverage {coverage[split]:.3f}"
        )
    score_seconds = time.perf_counter() - start
    peak_mb = torch.cuda.max_memory_allocated() / 2**20 if cuda else None
    latency = measure_latency_ms(
        lambda texts: scorer.predict_proba(texts, 1), splits["test"]["text"].tolist()[:50]
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
        "method": mcfg["method"],
        "label_coverage": coverage,
        "score_seconds": score_seconds,
        "peak_gpu_mem_mb": peak_mb,
        "latency_ms": latency,
        "device": torch.cuda.get_device_name() if cuda else "cpu",
        "prompt_example": scorer.prompt(splits["test"]["text"].iloc[0]),
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
