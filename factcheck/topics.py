"""LDA topics by label (owner M3, T56; plan experiment E10, interpretation step 0).

    python -m factcheck.topics            # ISOT (cleaned text) and PUBHEALTH

Do "real" and "fake" items differ by TOPIC rather than by truth? If they do, a classifier can
score well by recognising the subject of an item without judging whether it is true. For each
dataset this module:

1. fits LDA (Week 3) with ``N_TOPICS`` topics on the TRAIN text only, as a bag of words
   (English stop words plus a few reporting fillers such as "said" and "just", numbers and
   words under 3 letters dropped; words in fewer than 10 documents or in more than half of
   them dropped);
2. measures each topic's prevalence per label: the mean share of that topic in the label's
   training texts;
3. runs a topic-only probe: logistic regression on the ``N_TOPICS`` topic proportions alone,
   trained on train and scored on validation (macro-F1, against always guessing the bigger
   class). A high score means the subject alone gives the label away.

Labels are the binary mapping used across the project (``binary_label``: 1 = fake / false,
0 = real / true); PUBHEALTH's mixture and unproven claims are left out of the label comparison.
ISOT uses the cleaned text (T13), so the source fingerprints are already gone and what is left
is what the articles are about. PUBHEALTH uses the claims, the text its models read.

Writes ``results/topics/<dataset>_topics.csv``, ``results/topics/summary.csv`` and
``results/figures/t56_lda_topics.png``. About seven minutes on a laptop CPU (ISOT is most of it).
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import LatentDirichletAllocation  # noqa: E402
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from factcheck.data import load_splits  # noqa: E402
from factcheck.paths import FIGURES_DIR, RESULTS_DIR  # noqa: E402

DATASETS = {"isot": "clean", "pubhealth": "raw"}  # dataset -> variant
TITLES = {"isot": "ISOT news articles (cleaned)", "pubhealth": "PUBHEALTH claims"}
N_TOPICS = 10
TOP_WORDS = 8
SEED = 42
OUT_DIR = RESULTS_DIR / "topics"
FIGURE = FIGURES_DIR / "t56_lda_topics.png"
LABEL_NAMES = {0: "real / true", 1: "fake / false"}
# Reporting fillers that otherwise top every topic and say nothing about its subject.
FILLERS = {"said", "says", "say", "just", "like", "new", "year", "years", "people", "time",
           "know", "think", "going", "told", "make", "want", "way", "did", "does", "don",
           "really", "according", "shows", "image", "featured"}  # fmt: skip

# Team palette (notebooks/02_results_v1.py); blue and orange were checked for colour-blind
# separation in T32 (OKLab dE >= 24 under protan, deutan and tritan simulation).
INK, MUTED, GRID, SURFACE = "#13294b", "#52514e", "#e1e0d9", "#fcfcfb"
REAL, FAKE = "#2a78d6", "#eb6834"


def vectorizer() -> CountVectorizer:
    return CountVectorizer(
        stop_words=sorted(ENGLISH_STOP_WORDS | FILLERS),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]{2,}\b",  # words of 3+ letters, no numbers
        min_df=10,
        max_df=0.5,
        max_features=8000,
    )


def fit_topics(texts: list[str], n_topics: int = N_TOPICS, seed: int = SEED):  # noqa: ANN201
    """Bag of words and an LDA model fitted on ``texts``; returns (vectorizer, lda)."""
    vec = vectorizer()
    counts = vec.fit_transform(texts)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        learning_method="online",
        batch_size=1024,
        max_iter=15,
        random_state=seed,
        n_jobs=-1,
    ).fit(counts)
    return vec, lda


def top_words(vec: CountVectorizer, lda: LatentDirichletAllocation, n: int = TOP_WORDS) -> list:
    words = vec.get_feature_names_out()
    return [[words[i] for i in np.argsort(-topic)[:n]] for topic in lda.components_]


def topic_table(theta: np.ndarray, labels: np.ndarray, words: list[list[str]]) -> pd.DataFrame:
    """Per topic: its top words and its mean share of the real and of the fake texts."""
    real, fake = theta[labels == 0].mean(axis=0), theta[labels == 1].mean(axis=0)
    table = pd.DataFrame({
        "topic": range(len(words)),
        "top_words": [" ".join(w) for w in words],
        "share_real": real.round(4),
        "share_fake": fake.round(4),
    })  # fmt: skip
    table["fake_minus_real"] = (table["share_fake"] - table["share_real"]).round(4)
    return table.sort_values("fake_minus_real", ascending=False).reset_index(drop=True)


def topic_probe(theta_train, y_train, theta_val, y_val) -> dict:
    """How well do the topic proportions alone predict the label (validation macro-F1)?"""
    probe = LogisticRegression(max_iter=2000).fit(theta_train, y_train)
    majority = np.full_like(y_val, np.bincount(y_train).argmax())
    return {
        "probe_macro_f1": round(f1_score(y_val, probe.predict(theta_val), average="macro"), 4),
        "majority_macro_f1": round(f1_score(y_val, majority, average="macro"), 4),
        "probe_accuracy": round(float((probe.predict(theta_val) == y_val).mean()), 4),
        "majority_accuracy": round(float((majority == y_val).mean()), 4),
    }


def analyse(name: str, variant: str) -> tuple[pd.DataFrame, dict]:
    splits = load_splits(name, variant)
    train = splits["train"][splits["train"]["binary_label"] >= 0]
    val = splits["validation"][splits["validation"]["binary_label"] >= 0]
    vec, lda = fit_topics(train["text"].tolist())
    theta_train = lda.transform(vec.transform(train["text"]))
    theta_val = lda.transform(vec.transform(val["text"]))
    y_train, y_val = train["binary_label"].to_numpy(), val["binary_label"].to_numpy()
    table = topic_table(theta_train, y_train, top_words(vec, lda))
    summary = {
        "dataset": name,
        "variant": variant,
        "train_texts": len(train),
        "topics": N_TOPICS,
        **topic_probe(theta_train, y_train, theta_val, y_val),
        # share of training texts whose main topic is one where a single label has >= 80%
        "texts_in_one_sided_topics": round(float(one_sided_share(theta_train, y_train)), 4),
    }
    return table, summary


def one_sided_share(theta: np.ndarray, labels: np.ndarray, purity: float = 0.8) -> float:
    """Share of texts whose dominant topic is at least ``purity`` one label."""
    main = theta.argmax(axis=1)
    share = 0
    for k in np.unique(main):
        in_k = labels[main == k]
        if max(in_k.mean(), 1 - in_k.mean()) >= purity:
            share += len(in_k)
    return share / len(labels)


def plot(tables: dict[str, pd.DataFrame], summaries: dict[str, dict]) -> None:
    fig, axes = plt.subplots(1, len(tables), figsize=(16, 6.8), facecolor=SURFACE, sharex=True)
    for ax, (name, table) in zip(axes, tables.items(), strict=True):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#c3c2b7")
        ax.tick_params(colors=MUTED, labelsize=10, length=0)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        y = np.arange(len(table))[::-1]
        h = 0.38
        ax.barh(y + h / 2 + 0.01, table["share_real"] * 100, height=h, color=REAL,
                label=f"{LABEL_NAMES[0]} texts")  # fmt: skip
        ax.barh(y - h / 2 - 0.01, table["share_fake"] * 100, height=h, color=FAKE,
                label=f"{LABEL_NAMES[1]} texts")  # fmt: skip
        names = [", ".join(w.split()[:4]) for w in table["top_words"]]
        ax.set_yticks(y, names, fontsize=10, color=INK)
        ax.set_xlabel("Share of the label's training text in this topic (%)", fontsize=10.5,
                      color=MUTED)  # fmt: skip
        s = summaries[name]
        ax.set_title(TITLES[name], fontsize=13, color=INK, loc="left", fontweight="bold", pad=24)
        ax.text(0, 1.015, f"topics alone predict the label: macro-F1 {s['probe_macro_f1']:.2f}"
                f" (bigger class only: {s['majority_macro_f1']:.2f})",
                transform=ax.transAxes, fontsize=10, color=MUTED)  # fmt: skip
        ax.tick_params(axis="x", labelbottom=True)
    fig.text(0.01, 0.01, f"LDA with {N_TOPICS} topics on the training text; topics named by their "
             "top words and sorted from most fake-leaning (top) to most real-leaning.",
             fontsize=9, color=MUTED)  # fmt: skip
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=10.5, labelcolor=INK, ncol=2,
               loc="upper right")  # fmt: skip
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    args = parser.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tables, summaries = {}, {}
    for name in args.datasets:
        table, summary = analyse(name, DATASETS[name])
        table.to_csv(OUT_DIR / f"{name}_topics.csv", index=False)
        tables[name], summaries[name] = table, summary
        print(f"\n== {name}: {summary}")
        print(table.to_string(index=False))
    pd.DataFrame(summaries.values()).to_csv(OUT_DIR / "summary.csv", index=False)
    if len(tables) == len(DATASETS):
        plot(tables, summaries)
    print(f"\nwrote {OUT_DIR} and {FIGURE}")


if __name__ == "__main__":
    main()
