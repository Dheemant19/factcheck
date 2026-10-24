"""Embedding index build, cache and search (owner M3, T08; input builders T14).

    python -m factcheck.retrieval build                   # embed every dataset and cache it
    python -m factcheck.retrieval build --datasets liar   # just these
    python -m factcheck.retrieval check                   # neighbours, kNN-vote accuracy, duplicate flags

    from factcheck.retrieval import load_claim_index, load_evidence
    index = load_claim_index("liar")                      # TRAIN claims only; cached after first use
    hits = index.search_texts(["Says the economy grew 5% last year."], k=8)[0]
    hits[0].text, hits[0].label_name, hits[0].score       # most similar first
    load_evidence("pubhealth").top_k("33456", k=5)        # best 5 sentences of that claim's article

What is embedded (plan Section 8.4, M3 guide T08):

* **Claim index**, one per dataset, from the train split only, so a prompt can never show a
  validation or test verdict. Climate-FEVER is test-only and has no claim index. ISOT rows are
  represented by the title plus the first ``ISOT_INDEX_WORDS`` words of the article.
* **Evidence**: for PUBHEALTH, the sentences of each claim's own fact-check article
  (``main_text``), ranked against that claim; for Climate-FEVER, each claim's 5 evidence
  sentences, plus the pool of every evidence sentence for open retrieval.

Embeddings come from ``BAAI/bge-small-en-v1.5`` with unit length, so cosine similarity is a dot
product and search is one matrix multiply in NumPy. They are cached as float16 ``.npy`` files in
``CACHE_DIR/retrieval/``; the file name holds a hash of the model, the exact ids and texts, so a
change in the split, the cleaning or the model rebuilds the cache automatically.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import cache, cached_property

import numpy as np
import pandas as pd

from factcheck.data import get_dataset, load_splits
from factcheck.data.cleaning import basic_clean
from factcheck.data.prepare import excluded_ids
from factcheck.data.registry import DatasetInfo
from factcheck.paths import CACHE_DIR, RESULTS_DIR

MODEL_ID = "BAAI/bge-small-en-v1.5"
MAX_SEQ_LENGTH = 512  # tokens; bge-small's limit
# bge-v1.5 offers a query instruction ("Represent this sentence for searching relevant
# passages: ") for short-query -> passage search. On 150 PUBHEALTH validation claims it made no
# difference (mean similarity of the top-5 evidence to the expert explanation 0.696 with and
# without; 90% the same sentences), so evidence queries use no prefix.
EVIDENCE_PREFIX = ""
ISOT_INDEX_WORDS = 100  # title + lead paragraph: identifies the story; 2x faster than 200 on a CPU
MIN_EVIDENCE_WORDS = 5  # shorter "sentences" are captions, bylines and fragments
DUPLICATE_THRESHOLD = 0.97  # val/test claims this close to a train claim are flagged for T13
DEFAULT_K = 8

RETRIEVAL_DIR = CACHE_DIR / "retrieval"
DUPLICATE_FLAGS_CSV = RESULTS_DIR / "retrieval_duplicate_flags.csv"

Encoder = Callable[[list[str]], np.ndarray]  # texts -> (n, d) unit-length float32


# --- embedding ------------------------------------------------------------------------------


@cache
def get_model(model_id: str = MODEL_ID):  # noqa: ANN201  (SentenceTransformer)
    """Load the embedding model once per process, on the GPU when there is one."""
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_id, device=device)
    model.max_seq_length = MAX_SEQ_LENGTH
    return model


def embed(texts: Sequence[str], prefix: str = "", batch_size: int = 64) -> np.ndarray:
    """Unit-length embeddings (float32), so cosine similarity == dot product.

    sentence-transformers sorts by length inside ``encode``, so batches need little padding.
    """
    model = get_model()
    return model.encode(
        [prefix + t for t in texts],
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=len(texts) > 2000,
    ).astype(np.float32)


def cached_embed(
    kind: str,
    name: str,
    ids: Sequence[str],
    texts: Sequence[str],
    prefix: str = "",
    encoder: Encoder | None = None,
) -> np.ndarray:
    """Embed ``texts`` once and cache them; ``encoder`` (tests) bypasses model and cache."""
    if encoder is not None:
        return encoder([prefix + t for t in texts])
    key = hashlib.sha1()
    for part in (MODEL_ID, str(MAX_SEQ_LENGTH), prefix, "\n".join(ids), "\n".join(texts)):
        key.update(part.encode("utf-8"))
        key.update(b"\0")
    path = RETRIEVAL_DIR / f"{name}-{kind}-{key.hexdigest()[:12]}.npy"
    if path.exists():
        return np.load(path).astype(np.float32)
    start = time.perf_counter()
    emb = embed(list(texts), prefix=prefix)
    print(f"  embedded {len(texts):,} {name} {kind} texts in {time.perf_counter() - start:.0f} s")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npy")
    np.save(tmp, emb.astype(np.float16))
    os.replace(tmp, path)
    for old in path.parent.glob(f"{name}-{kind}-*.npy"):  # drop outdated caches
        if old != path:
            old.unlink(missing_ok=True)
    return emb


def index_texts(df: pd.DataFrame, name: str) -> list[str]:
    """How a row is represented in the claim index and as a query: its cleaned text, except
    ISOT, where the title plus the start of the article identifies the story."""
    texts = df["text"].astype(str)
    if name == "isot":
        texts = texts.str.split(n=ISOT_INDEX_WORDS).str[:ISOT_INDEX_WORDS].str.join(" ")
    return texts.tolist()


def top_k_rows(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Positions and values of the k largest scores in every row, largest first."""
    k = min(k, scores.shape[1])
    top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1, kind="stable")
    top = np.take_along_axis(top, order, axis=1)
    return top, np.take_along_axis(scores, top, axis=1)


# --- claim index ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Hit:
    """One retrieved training claim."""

    id: str
    text: str
    label: int
    label_name: str
    score: float  # cosine similarity to the query


@dataclass
class ClaimIndex:
    """Already fact-checked TRAINING claims of one dataset, searchable by similarity."""

    info: DatasetInfo
    ids: np.ndarray  # (n,) str
    texts: list[str]
    labels: np.ndarray  # (n,) int class ids
    emb: np.ndarray  # (n, d) float32, unit length
    encoder: Encoder | None = field(default=None, repr=False)

    def __len__(self) -> int:
        return len(self.ids)

    @cached_property
    def position(self) -> dict[str, int]:
        return {cid: i for i, cid in enumerate(self.ids)}

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        return self.encoder(texts) if self.encoder else embed(texts)

    def search(
        self,
        queries: np.ndarray,
        k: int = DEFAULT_K,
        exclude_ids: Sequence[str] | None = None,
        chunk: int = 1024,
    ) -> tuple[np.ndarray, np.ndarray]:
        """For each query row: positions of the k most similar claims (most similar first) and
        their similarities. ``exclude_ids[i]`` is left out of row i's results, so a training
        claim used as a query never retrieves itself."""
        if exclude_ids is not None and len(exclude_ids) != len(queries):
            raise ValueError("exclude_ids needs one id per query row")
        positions, scores = [], []
        for start in range(0, len(queries), chunk):  # bounded memory for big indexes (ISOT)
            sims = queries[start : start + chunk] @ self.emb.T
            if exclude_ids is not None:
                for row, cid in enumerate(exclude_ids[start : start + chunk]):
                    if cid in self.position:
                        sims[row, self.position[cid]] = -np.inf
            top, top_sims = top_k_rows(sims, k)
            positions.append(top)
            scores.append(top_sims)
        return np.vstack(positions), np.vstack(scores)

    def hits(self, positions: np.ndarray, scores: np.ndarray) -> list[list[Hit]]:
        names = self.info.labels
        return [
            [
                Hit(
                    self.ids[p], self.texts[p], int(self.labels[p]), names[self.labels[p]], float(s)
                )
                for p, s in zip(row_pos, row_scores, strict=True)
            ]
            for row_pos, row_scores in zip(positions, scores, strict=True)
        ]

    def search_texts(
        self, texts: Sequence[str], k: int = DEFAULT_K, exclude_ids: Sequence[str] | None = None
    ) -> list[list[Hit]]:
        """Embed new texts (e.g. a claim typed into the app) and return their hits."""
        return self.hits(*self.search(self.embed_queries(texts), k, exclude_ids))

    def vote(
        self, queries: np.ndarray, k: int = DEFAULT_K, exclude_ids: Sequence[str] | None = None
    ) -> np.ndarray:
        """kNN-vote class probabilities (rows sum to 1): each of the k nearest training claims
        votes for its label with weight = its similarity, so ties are practically impossible."""
        positions, scores = self.search(queries, k, exclude_ids)
        weights = np.clip(scores, 1e-6, None)
        probs = np.zeros((len(queries), self.info.num_labels))
        np.add.at(probs, (np.arange(len(queries))[:, None], self.labels[positions]), weights)
        return probs / probs.sum(axis=1, keepdims=True)


def build_claim_index(
    name: str, encoder: Encoder | None = None, drop_excluded: bool = True
) -> ClaimIndex:
    """Embed (or load from cache) the train claims of one dataset.

    The whole train split is embedded and cached once; rows T13 excluded (near-copies of a
    validation/test claim) are then left out, so they can never be retrieved as examples.
    """
    info = get_dataset(name)
    train = load_splits(name, drop_excluded=False)["train"]
    if train.empty:
        raise ValueError(f"{name} has no train split, so it has no claim index (test only)")
    texts = index_texts(train, name)
    emb = cached_embed("claims", name, train["id"].tolist(), texts, encoder=encoder)
    keep = ~train["id"].isin(excluded_ids(name)).to_numpy() if drop_excluded else slice(None)
    train = train[keep]
    return ClaimIndex(
        info=info,
        ids=train["id"].to_numpy(dtype=str),
        texts=train["text"].tolist(),
        labels=train["label"].to_numpy(dtype=np.int64),
        emb=emb[keep],
        encoder=encoder,
    )


@cache
def load_claim_index(name: str) -> ClaimIndex:
    """The claim index of one dataset, built on first use and kept for the process."""
    return build_claim_index(name)


def embed_split(index: ClaimIndex, df: pd.DataFrame, split: str) -> np.ndarray:
    """Query embeddings for every row of a split (cached like the index)."""
    texts = index_texts(df, index.info.name)
    kind = f"queries-{split}"
    return cached_embed(kind, index.info.name, df["id"].tolist(), texts, encoder=index.encoder)


# --- evidence -------------------------------------------------------------------------------


@cache
def _punkt() -> None:
    import nltk

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


def split_sentences(text: str, min_words: int = MIN_EVIDENCE_WORDS) -> list[str]:
    """Sentences of an article (nltk punkt), without fragments shorter than ``min_words``."""
    from nltk.tokenize import sent_tokenize

    _punkt()
    return [s for s in sent_tokenize(basic_clean(text)) if len(s.split()) >= min_words]


@dataclass
class EvidenceStore:
    """Evidence sentences of one dataset and, for every claim, which sentences are its own."""

    name: str
    sentences: list[str]  # every evidence sentence once
    emb: np.ndarray  # (n_sentences, d) float16, unit length
    members: dict[str, np.ndarray]  # claim id -> positions of its own evidence sentences
    claim_emb: dict[str, np.ndarray]  # claim id -> query embedding (float32)
    encoder: Encoder | None = field(default=None, repr=False)

    def top_k_scored(self, claim_id: str, k: int = 5) -> list[tuple[str, float]]:
        """The k sentences of this claim's own evidence most similar to the claim."""
        own = self.members.get(claim_id)
        if own is None or len(own) == 0:
            return []
        sims = self.emb[own].astype(np.float32) @ self.claim_emb[claim_id]
        top, top_sims = top_k_rows(sims[None, :], k)
        return [
            (self.sentences[own[p]], float(s)) for p, s in zip(top[0], top_sims[0], strict=True)
        ]

    def top_k(self, claim_id: str, k: int = 5) -> list[str]:
        return [sentence for sentence, _ in self.top_k_scored(claim_id, k)]

    def search_pool(self, claims: Sequence[str], k: int = 5) -> list[list[tuple[str, float]]]:
        """Open retrieval: the k most similar sentences from the WHOLE pool for new claim texts
        (the honest RAG setting for Climate-FEVER, where a claim's own evidence is not given)."""
        queries = [EVIDENCE_PREFIX + c for c in claims]
        q = self.encoder(queries) if self.encoder else embed(queries)
        top, sims = top_k_rows(q @ self.emb.astype(np.float32).T, k)
        return [
            [(self.sentences[p], float(s)) for p, s in zip(row, row_sims, strict=True)]
            for row, row_sims in zip(top, sims, strict=True)
        ]


def all_rows(name: str) -> pd.DataFrame:
    """Every claim of a dataset, all splits, nothing excluded (evidence exists for every one)."""
    return pd.concat(load_splits(name, drop_excluded=False).values(), ignore_index=True)


def build_evidence(name: str, encoder: Encoder | None = None) -> EvidenceStore:
    """PUBHEALTH: sentences of each claim's own article. Climate-FEVER: the 5 given evidence
    sentences of each claim, stored once each so the whole pool can be searched too."""
    df = all_rows(name)
    sentences: list[str] = []
    members: dict[str, np.ndarray] = {}
    if name == "pubhealth":
        for cid, article in zip(df["id"], df["main_text"], strict=True):
            own = split_sentences(article)
            members[cid] = np.arange(len(sentences), len(sentences) + len(own))
            sentences.extend(own)
    elif name == "climate_fever":
        position: dict[str, int] = {}
        for cid, texts, ev_ids in zip(df["id"], df["evidence"], df["evidence_ids"], strict=True):
            own = []
            for text, ev_id in zip(texts, ev_ids, strict=True):
                if ev_id not in position:
                    position[ev_id] = len(sentences)
                    sentences.append(basic_clean(text))
                own.append(position[ev_id])
            members[cid] = np.array(own, dtype=np.int64)
    else:
        raise ValueError(f"{name} has no evidence; evidence exists for pubhealth and climate_fever")

    emb = cached_embed("evidence", name, [str(i) for i in range(len(sentences))], sentences,
                       encoder=encoder)  # fmt: skip
    claims = cached_embed("evidence-queries", name, df["id"].tolist(), df["text"].tolist(),
                          prefix=EVIDENCE_PREFIX, encoder=encoder)  # fmt: skip
    return EvidenceStore(
        name=name,
        sentences=sentences,
        emb=emb.astype(np.float16),
        members=members,
        claim_emb=dict(zip(df["id"], claims, strict=True)),
        encoder=encoder,
    )


@cache
def load_evidence(name: str) -> EvidenceStore:
    """The evidence store of pubhealth or climate_fever, built on first use."""
    return build_evidence(name)


# --- kNN-vote baseline (a model family for factcheck.train) ---------------------------------


def score_knn(cfg: dict) -> dict:
    """kNN-vote baseline: the similarity-weighted majority label of the k nearest training
    claims. ``python -m factcheck.train --config configs/knn/knn_liar.yaml``."""
    from factcheck.evaluate import log_run

    data, model = cfg["data"], cfg["model"]
    if data.get("variant", "raw") != "raw":
        raise ValueError("the claim index is built on the raw variant")
    k = int(model.get("k", DEFAULT_K))
    start = time.perf_counter()
    index = load_claim_index(data["dataset"])
    build_seconds = time.perf_counter() - start
    splits = load_splits(data["dataset"])
    scored = {}
    for split in ("validation", "test"):
        queries = embed_split(index, splits[split], split)
        scored[split] = (splits[split], index.vote(queries, k))
    extra = {
        "trainable_params": 0,  # nothing is trained: the index is the model
        "train_seconds": build_seconds,
        "device": str(get_model().device),
    }
    return {"rows": log_run(cfg, index.info, scored, extra=extra)}


# --- sanity checks --------------------------------------------------------------------------


def show_neighbours(index: ClaimIndex, df: pd.DataFrame, n: int = 3, k: int = 5) -> None:
    """Print a few validation claims with their nearest training claims, to read by eye."""
    sample = df.sample(n=min(n, len(df)), random_state=0)
    all_hits = index.search_texts(index_texts(sample, index.info.name), k)
    for row, hits in zip(sample.itertuples(), all_hits, strict=True):
        print(f"\n  QUERY [{row.label_name}] {row.text[:110]}")
        for hit in hits:
            print(f"    {hit.score:.3f} [{hit.label_name}] {hit.text[:100]}")


def duplicate_flags(index: ClaimIndex, splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Validation/test claims whose nearest training claim has similarity >= the threshold:
    likely duplicates of a training claim, handed to T13."""
    flags = []
    for split in ("validation", "test"):
        df = splits[split]
        positions, scores = index.search(embed_split(index, df, split), k=1)
        close = scores[:, 0] >= DUPLICATE_THRESHOLD
        for i in np.flatnonzero(close):
            p = positions[i, 0]
            flags.append({
                "dataset": index.info.name, "split": split, "id": df["id"].iloc[i],
                "label_name": df["label_name"].iloc[i], "train_id": index.ids[p],
                "train_label_name": index.info.labels[index.labels[p]],
                "similarity": round(float(scores[i, 0]), 4),
                "text": df["text"].iloc[i][:300], "train_text": index.texts[p][:300],
            })  # fmt: skip
    return pd.DataFrame(flags)


def knn_accuracy(index: ClaimIndex, df: pd.DataFrame, split: str, k: int = DEFAULT_K) -> dict:
    from sklearn.metrics import accuracy_score, f1_score

    pred = index.vote(embed_split(index, df, split), k).argmax(axis=1)
    return {
        "accuracy": accuracy_score(df["label"], pred),
        "macro_f1": f1_score(df["label"], pred, average="macro"),
    }


def check(names: list[str]) -> None:
    """Read a few neighbours, score the kNN vote on validation, write duplicate flags."""
    all_flags = []
    for name in names:
        if name == "climate_fever":
            store = load_evidence(name)
            cid = next(iter(store.members))
            print(f"\n== climate_fever: {len(store.sentences):,} evidence sentences in the pool")
            print("  ", store.top_k_scored(cid, k=2))
            continue
        index = load_claim_index(name)
        splits = load_splits(name)
        scores = knn_accuracy(index, splits["validation"], "validation")
        print(f"\n== {name}: {len(index):,} train claims; kNN vote (k={DEFAULT_K}) on validation: "
              f"accuracy {scores['accuracy']:.3f}, macro-F1 {scores['macro_f1']:.3f} "
              f"(majority class {splits['validation']['label'].value_counts(normalize=True).iloc[0]:.3f})")  # fmt: skip
        show_neighbours(index, splits["validation"])
        flags = duplicate_flags(index, splits)
        print(f"  likely duplicates of a train claim (similarity >= {DUPLICATE_THRESHOLD}): "
              f"{len(flags)} ({flags['split'].value_counts().to_dict() if len(flags) else {}})")  # fmt: skip
        all_flags.append(flags)
        if name == "pubhealth":
            store = load_evidence(name)
            cid = splits["validation"]["id"].iloc[0]
            print(f"  evidence: {len(store.sentences):,} sentences; top 2 for claim {cid}:")
            for sentence, score in store.top_k_scored(cid, k=2):
                print(f"    {score:.3f} {sentence[:110]}")
    if all_flags:
        flags = pd.concat(all_flags, ignore_index=True)
        if DUPLICATE_FLAGS_CSV.exists():  # keep other datasets' rows when checking a subset
            old = pd.read_csv(DUPLICATE_FLAGS_CSV, dtype=str)
            flags = pd.concat([old[~old["dataset"].isin(names)], flags], ignore_index=True)
        DUPLICATE_FLAGS_CSV.parent.mkdir(parents=True, exist_ok=True)
        flags.to_csv(DUPLICATE_FLAGS_CSV, index=False)
        print(f"\nwrote {DUPLICATE_FLAGS_CSV} ({len(flags)} rows)")


def build(names: list[str]) -> None:
    """Embed and cache every index and evidence store (one command, e.g. on the cluster)."""
    for name in names:
        start = time.perf_counter()
        if name != "climate_fever":
            index = load_claim_index(name)
            splits = load_splits(name)
            for split in ("validation", "test"):
                embed_split(index, splits[split], split)
        if name in ("pubhealth", "climate_fever"):
            load_evidence(name)
        print(f"{name}: ready in {time.perf_counter() - start:.0f} s")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("command", choices=["build", "check"])
    parser.add_argument("--datasets", nargs="+", default=None, help="default: all five")
    args = parser.parse_args(argv)
    names = args.datasets or ["covid", "liar", "pubhealth", "isot", "climate_fever"]
    build(names) if args.command == "build" else check(names)


if __name__ == "__main__":
    main()
