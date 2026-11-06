"""One claim in; verdict, evidence and explanation out: the engine behind the app (T22, T35).

    from factcheck.serve import Classifier, EvidenceSearch, explain

    clf = Classifier("outputs/covid_twitter_bert_covid/seed13/model")   # or a QLoRA adapter folder
    search = EvidenceSearch()
    found = search.find("Vitamin D cures COVID-19 in 24 hours.", dataset=clf.info.name)
    pred = clf.predict("Vitamin D cures COVID-19 in 24 hours.", found.sentence_texts())
    pred.label_probs, pred.fake_prob, pred.seconds
    why = explain(clf, "Vitamin D cures COVID-19 in 24 hours.")         # SHAP, per token

A **classifier** is any saved run: a fine-tuned encoder (``model/`` with ``config.json``) or a
QLoRA Qwen3-8B adapter (``model/`` with ``adapter_config.json``, T27). Either way it scores a
claim exactly as in training, from the run's ``config.yaml``: the same truncation, and for runs
trained with evidence (``features.use_evidence``, T29) the claim paired with evidence sentences.
In the app those sentences come from open retrieval, not from the claim's own article.

**Evidence** (plan Section 15) is the claim's most similar already fact-checked TRAINING claims
of the model's dataset, with their verdicts (the T08 claim index), and the most similar
sentences from the evidence pools: PUBHEALTH's fact-check articles and Climate-FEVER's evidence.
The claim is embedded once for both. Building the caches the first time takes a few minutes:
``python -m factcheck.retrieval build``.

**Why** is a SHAP partition explainer (the masker of M4's ``factcheck.explain``) over the
classifier that is already loaded, so the app holds one copy of each model's weights.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import torch
import yaml

from factcheck.data import get_dataset
from factcheck.data.registry import DatasetInfo

EVIDENCE_POOLS = ("pubhealth", "climate_fever")
POOL_NAMES = {
    "pubhealth": "PUBHEALTH fact-check article",
    "climate_fever": "Climate-FEVER evidence",
}
SIMILAR_K = 5
SENTENCE_K = 5  # the T29 evidence runs were trained on the top 5 sentences
EXPLAIN_EVALS = 300  # model calls per explained claim; 500 in the offline T16 runs


def run_config(model_dir: str | Path) -> dict:
    """The run's config.yaml (it sits next to ``model/``)."""
    with open(Path(model_dir).parent / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def is_qlora(model_dir: str | Path) -> bool:
    return (Path(model_dir) / "adapter_config.json").exists()


def is_servable(model_dir: str | Path) -> bool:
    """A model the app can run on a pasted claim: an encoder or a QLoRA adapter with its config.

    Runs trained with LIAR speaker metadata are left out: a pasted claim has no speaker, party
    or context, so the model would see inputs unlike its training data.
    """
    model_dir = Path(model_dir)
    if not (model_dir.parent / "config.yaml").exists():
        return False
    if not ((model_dir / "config.json").exists() or is_qlora(model_dir)):
        return False
    return not (run_config(model_dir).get("features") or {}).get("use_metadata")


# --------------------------------------------------------------------------- classifiers


@dataclass
class Prediction:
    label_probs: dict[str, float]  # label name -> probability, in the dataset's label order
    fake_prob: float  # summed over the labels that count as false
    seconds: float  # the model's forward pass for this claim (loading excluded)


class Classifier:
    """A saved run loaded once, scoring one claim at a time (on the GPU when there is one)."""

    def __init__(self, model_dir: str | Path, device: str | None = None, half: bool | None = None):
        """``half`` (default: on a GPU) runs an encoder in fp16. Verdicts move by about 1e-5, and
        it halves the memory, so an encoder and the 4-bit Qwen3-8B fit in 8 GB without spilling into
        shared memory (with fp32 weights the encoder slowed from 10 ms to 120 ms next to Qwen)."""
        # Load weights on the main thread (transformers 5's threaded loading crashed on Windows)
        os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
        self.model_dir = Path(model_dir)
        self.cfg = run_config(self.model_dir)
        data = self.cfg["data"]
        self.info: DatasetInfo = get_dataset(data["dataset"])
        self.variant = data.get("variant", "raw")
        self.max_length = int(data.get("max_length") or self.info.max_length)
        self.head_tokens = data.get("head_tokens")
        self.uses_evidence = bool((self.cfg.get("features") or {}).get("use_evidence"))
        self.qlora = is_qlora(self.model_dir)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.qlora:
            from factcheck.models.encoder import load_qlora

            self.model, self.tokenizer = load_qlora(self.model_dir)  # 4-bit, on GPU 0
        else:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            half = (device == "cuda") if half is None else half
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_dir, dtype=torch.float16 if half else torch.float32
            )
            self.model.to(device)
        self.model.eval()
        self.device = next(self.model.parameters()).device
        # The first CUDA passes for a new input length are slow; warm up on short and longer
        # inputs so the timed check measures the model, not the GPU's start-up.
        warm = "Health officials said the claim spread widely online before it was checked."
        for text in ("warm-up", warm, " ".join([warm] * 3)):
            self.proba([text], [warm] if self.uses_evidence else None)

    @property
    def name(self) -> str:
        return self.cfg["run"]["name"]

    def encode(self, texts: list[str], evidence: list[str] | None = None) -> list[list[int]]:
        from factcheck.models.encoder import encode_texts

        if self.uses_evidence:
            pair = " ".join(evidence or [])
            return encode_texts(texts, self.tokenizer, self.max_length, pairs=[pair] * len(texts))
        return encode_texts(texts, self.tokenizer, self.max_length, self.head_tokens)

    @torch.no_grad()
    def proba(
        self, texts: list[str], evidence: list[str] | None = None, batch_size: int = 32
    ) -> np.ndarray:
        """Label probabilities, one row per text (all texts share the same evidence)."""
        out = []
        for start in range(0, len(texts), batch_size):
            ids = self.encode(texts[start : start + batch_size], evidence)
            batch = self.tokenizer.pad({"input_ids": ids}, return_tensors="pt").to(self.device)
            if self.qlora:  # 4-bit weights compute in bf16, as in training
                with torch.autocast(self.device.type, dtype=torch.bfloat16):
                    logits = self.model(**batch).logits
            else:
                logits = self.model(**batch).logits
            out.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        return np.concatenate(out)

    def predict(self, claim: str, evidence: list[str] | None = None) -> Prediction:
        start = time.perf_counter()
        probs = self.proba([claim], evidence)[0]
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        return Prediction(
            label_probs={name: float(p) for name, p in zip(self.info.labels, probs, strict=True)},
            fake_prob=float(probs[self.info.fake_label_ids].sum()),
            seconds=seconds,
        )


# --------------------------------------------------------------------------- evidence


@dataclass
class Sentence:
    text: str
    source: str  # evidence pool: pubhealth or climate_fever
    score: float  # cosine similarity to the claim


@dataclass
class Found:
    similar: list  # retrieval.Hit: most similar TRAINING claims of the dataset, with verdicts
    sentences: list[Sentence]  # most similar evidence sentences over the pools, best first
    seconds: float  # embedding the claim plus every search
    dataset: str | None = None

    def sentence_texts(self, k: int = SENTENCE_K) -> list[str]:
        return [s.text for s in self.sentences[:k]]


@dataclass
class EvidenceSearch:
    """Open retrieval for a pasted claim over the cached claim indexes and evidence pools."""

    pools: tuple[str, ...] = EVIDENCE_POOLS
    _pool_emb: dict = field(default_factory=dict, repr=False)

    @cached_property
    def stores(self) -> dict:
        from factcheck.retrieval import load_evidence

        return {name: load_evidence(name) for name in self.pools}

    def pool_matrix(self, name: str) -> np.ndarray:
        """The pool's sentence embeddings as float32, converted once (the cache is float16)."""
        if name not in self._pool_emb:
            self._pool_emb[name] = self.stores[name].emb.astype(np.float32)
        return self._pool_emb[name]

    def warm(self, dataset: str | None = None) -> None:
        """Load the caches and the embedding model before the first timed search."""
        from factcheck.retrieval import embed, load_claim_index

        for name in self.pools:
            self.pool_matrix(name)
        if dataset in ("covid", "liar", "pubhealth", "isot"):
            load_claim_index(dataset)
        embed(["warm-up"])

    def find(
        self,
        claim: str,
        dataset: str | None,
        k_similar: int = SIMILAR_K,
        k_sentences: int = SENTENCE_K,
    ) -> Found:
        from factcheck.retrieval import embed, load_claim_index, top_k_rows

        start = time.perf_counter()
        query = embed([claim])  # (1, d), unit length; evidence queries use no prefix (T08)
        similar = []
        if dataset in ("covid", "liar", "pubhealth", "isot"):  # Climate-FEVER has no train split
            index = load_claim_index(dataset)
            similar = index.hits(*index.search(query, k_similar))[0]
        sentences = []
        for name in self.pools:
            sims = query @ self.pool_matrix(name).T
            top, scores = top_k_rows(sims, k_sentences)
            store = self.stores[name]
            sentences += [
                Sentence(store.sentences[p], name, float(s))
                for p, s in zip(top[0], scores[0], strict=True)
            ]
        sentences.sort(key=lambda s: -s.score)
        return Found(similar, sentences[:k_sentences], time.perf_counter() - start, dataset)


# --------------------------------------------------------------------------- why


@dataclass
class Explanation:
    tokens: list[str]  # the claim as SHAP split it (each piece keeps its own spacing)
    towards_false: np.ndarray  # per token, in log-odds of false: + towards false, - towards true
    base: float  # log-odds of false with every token masked
    probs: list[float]  # the model's label probabilities for the full claim
    seconds: float
    evals: int

    def words(self) -> list[tuple[str, float]]:
        """Whole words with their summed push, rejoining sub-word pieces ("CO" "VID" "-"
        "19 " is one word, "COVID-19 "). Each word keeps its trailing space."""
        out: list[tuple[str, float]] = []
        joined = True  # the next piece starts a new word
        for token, value in zip(self.tokens, self.towards_false, strict=True):
            if not token:
                continue
            if joined or token[0].isspace() or not out:
                out.append((token, float(value)))
            else:
                text, total = out[-1]
                out[-1] = (text + token, total + float(value))
            joined = token[-1].isspace()
        return out


def log_odds_false(probs: np.ndarray, fake_ids: list[int]) -> np.ndarray:
    p = np.clip(probs[:, fake_ids].sum(axis=1), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def explain(
    clf: Classifier, claim: str, evidence: list[str] | None = None, max_evals: int = EXPLAIN_EVALS
) -> Explanation:
    """SHAP token attributions of one claim for an already loaded classifier.

    The explained output is the log-odds that the claim is false (the false labels summed, as
    in the app's P(false)). On probabilities, a confident model's attributions all round to
    zero: COVID-Twitter-BERT still says 99% fake with every word masked. Log-odds keep them
    readable. The masker and the partition explainer are the same as M4's ``factcheck.explain``.

    A run trained with evidence is explained with the evidence held fixed: SHAP masks the
    claim's words only, so the attributions say which words of the claim mattered.
    """
    import shap

    fake = clf.info.fake_label_ids

    def predict(texts) -> np.ndarray:  # noqa: ANN001  (SHAP passes an array of strings)
        return log_odds_false(clf.proba([str(t) for t in texts], evidence, batch_size=64), fake)

    explainer = shap.Explainer(predict, shap.maskers.Text(clf.tokenizer))
    start = time.perf_counter()
    sv = explainer([claim], max_evals=max_evals, silent=True)
    seconds = time.perf_counter() - start
    return Explanation(
        tokens=[str(t) for t in sv.data[0]],
        towards_false=np.asarray(sv.values[0], dtype=float),
        base=float(np.asarray(sv.base_values).ravel()[0]),
        probs=clf.proba([claim], evidence)[0].tolist(),
        seconds=seconds,
        evals=max_evals,
    )
