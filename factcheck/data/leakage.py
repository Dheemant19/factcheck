"""Duplicate and train/test-overlap checks (owner M3, T13).

    python -m factcheck.data.leakage            # writes data/splits/exclusions.csv and
                                                # results/leakage_report.csv

Social posts, political claims and news wires are often repeated, so the same (or almost the
same) text can sit in train and in validation/test. A model is then graded on something it
memorised. This module finds those rows and lists them in ``data/splits/exclusions.csv``;
``load_splits`` drops the listed rows. The rules:

* Only TRAIN rows are ever excluded. Validation and test stay exactly as published, so our
  test scores remain comparable with the papers' baselines.
* A train row is excluded if its normalised text (lower-case; links, punctuation and spaces
  removed) equals a validation/test row, or if it is a near-copy of one: embedding cosine
  similarity >= ``NEAR_DUPLICATE`` (T08, bge-small) AND the same numbers. The numbers condition
  matters for templated posts: "States reported 941 new deaths" and "States reported 2492 new
  deaths" are 0.97 similar but state different facts, so they are not copies. Without it, the
  0.95 threshold alone would remove 13% of COVID's training posts, mostly official daily
  updates.
* Repeated texts inside train keep their first copy; if the copies disagree on the label, all
  copies are excluded (the label is noise).

The committed split files are not changed: exclusions are a separate, reviewable list with a
reason for every row.
"""

from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd

from factcheck.data import get_dataset, load_splits
from factcheck.data.prepare import exclusions_path
from factcheck.paths import RESULTS_DIR

NEAR_DUPLICATE = 0.95  # cosine similarity of bge-small embeddings (M3 guide, T13)
LEAKAGE_REPORT = RESULTS_DIR / "leakage_report.csv"
DATASETS_WITH_TRAIN = ["covid", "liar", "pubhealth", "isot"]
NON_ALNUM = re.compile(r"[^a-z0-9]+")
LINK = re.compile(r"(?:https?://|www\.)\S+", re.I)
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def normalise(text: str) -> str:
    """Lower-case; links, punctuation and spacing removed: "COVID-19 fake!! https://t.co/x"
    -> "covid 19 fake"."""
    return NON_ALNUM.sub(" ", LINK.sub(" ", text).lower()).strip()


def numbers(text: str) -> list[str]:
    """The numbers a text states, sorted: "1,326 deaths in 2 days" -> ["1326", "2"]."""
    return sorted(n.replace(",", "") for n in NUMBER.findall(LINK.sub(" ", text)))


def exact_exclusions(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Train rows that repeat a held-out text, or another train row."""
    train = splits["train"].assign(norm=lambda d: d["text"].map(normalise))
    held_out = pd.concat([splits["validation"], splits["test"]], ignore_index=True)
    held_out = held_out.assign(norm=lambda d: d["text"].map(normalise))
    first_held_out = held_out.drop_duplicates("norm").set_index("norm")["id"]

    rows = []
    overlap = train[train["norm"].isin(first_held_out.index)]
    for row in overlap.itertuples():
        rows.append((row.id, "same_text_as_heldout", first_held_out[row.norm], 1.0))

    rest = train[~train["norm"].isin(first_held_out.index)]
    groups = rest.groupby("norm")
    conflicting = groups["label"].transform("nunique") > 1
    for row in rest[conflicting].itertuples():
        rows.append((row.id, "conflicting_labels_in_train", "", 1.0))
    repeats = rest[~conflicting]
    first_id = repeats.groupby("norm")["id"].transform("first")
    for row, first in zip(repeats.itertuples(), first_id, strict=True):
        if row.id != first:
            rows.append((row.id, "duplicate_in_train", first, 1.0))
    return pd.DataFrame(rows, columns=["id", "reason", "matched_id", "similarity"])


def near_exclusions(name: str, splits: dict[str, pd.DataFrame], threshold: float) -> pd.DataFrame:
    """Train rows that are near-copies of a held-out row: embedding similarity >= threshold
    and the same numbers (compared on the same text the index embeds)."""
    from factcheck.retrieval import build_claim_index, embed_split, index_texts

    index = build_claim_index(name, drop_excluded=False)  # earlier exclusions are re-found
    train_text = index_texts(splits["train"], name)
    found: dict[str, tuple[str, float]] = {}  # train id -> (held-out id, similarity)
    for split in ("validation", "test"):
        df = splits[split]
        held_text = index_texts(df, name)
        queries = embed_split(index, df, split)
        for start in range(0, len(queries), 1024):
            sims = queries[start : start + 1024] @ index.emb.T  # (chunk, n_train)
            for q, t in zip(*np.nonzero(sims >= threshold), strict=True):
                j, train_id = start + q, index.ids[t]
                if train_id in found or numbers(train_text[t]) != numbers(held_text[j]):
                    continue
                found[train_id] = (df["id"].iloc[j], round(float(sims[q, t]), 4))
    return pd.DataFrame(
        [(cid, "near_duplicate_of_heldout", match, sim) for cid, (match, sim) in found.items()],
        columns=["id", "reason", "matched_id", "similarity"],
    )


def heldout_overlap(splits: dict[str, pd.DataFrame]) -> int:
    """Validation rows whose normalised text is also in test (reported, never removed)."""
    test = set(splits["test"]["text"].map(normalise))
    return int(splits["validation"]["text"].map(normalise).isin(test).sum())


def build(names: list[str], threshold: float = NEAR_DUPLICATE) -> tuple[pd.DataFrame, list[dict]]:
    """Exclusions for every dataset and the matching report rows."""
    exclusions, report = [], []
    for name in names:
        get_dataset(name)
        splits = load_splits(name, drop_excluded=False)
        exact = exact_exclusions(splits)
        near = near_exclusions(name, splits, threshold)
        near = near[~near["id"].isin(exact["id"])]  # count each row once, exact reason first
        found = pd.concat([exact, near], ignore_index=True).assign(dataset=name)
        exclusions.append(found)
        n_train = len(splits["train"])
        for reason, count in found["reason"].value_counts().items():
            report.append({"dataset": name, "check": reason, "rows_affected": int(count),
                           "share_of_train": round(count / n_train, 4),
                           "action": "excluded from train (data/splits/exclusions.csv)"})  # fmt: skip
        report.append({"dataset": name, "check": "validation_text_also_in_test",
                       "rows_affected": heldout_overlap(splits), "share_of_train": None,
                       "action": "kept: validation and test are never changed"})  # fmt: skip
    columns = ["dataset", "id", "reason", "matched_id", "similarity"]
    return pd.concat(exclusions, ignore_index=True)[columns], report


def shortcut_report() -> list[dict]:
    """Rows changed by each ISOT shortcut rule and by COVID's link/handle masking."""
    from factcheck.data.cleaning import MENTION, URL, basic_clean
    from factcheck.data.shortcuts import SHORTCUT_RULES, remove_shortcuts

    rows = []
    isot = pd.concat(load_splits("isot", drop_excluded=False).values(), ignore_index=True)
    fired = isot["text_raw"].map(lambda t: set(remove_shortcuts(basic_clean(t))[1]))
    for rule in SHORTCUT_RULES:
        hit = fired.map(lambda names, rule=rule: rule in names)
        fake, real = hit[isot["label_name"] == "fake"], hit[isot["label_name"] == "real"]
        rows.append({"dataset": "isot", "check": f"shortcut_rule:{rule}",
                     "rows_affected": int(hit.sum()), "share_of_train": None,
                     "action": f"removed in variant=clean (fake {fake.mean():.1%}, "
                               f"real {real.mean():.1%} of articles)"})  # fmt: skip
    covid = pd.concat(load_splits("covid", drop_excluded=False).values(), ignore_index=True)
    for check, pattern in (("social_masking:url", URL), ("social_masking:handle", MENTION)):
        hit = covid["text"].str.contains(pattern)
        fake, real = hit[covid["label_name"] == "fake"], hit[covid["label_name"] == "real"]
        rows.append({"dataset": "covid", "check": check, "rows_affected": int(hit.sum()),
                     "share_of_train": None,
                     "action": f"masked in variant=clean (fake {fake.mean():.1%}, "
                               f"real {real.mean():.1%} of posts)"})  # fmt: skip
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--datasets", nargs="+", default=DATASETS_WITH_TRAIN)
    parser.add_argument("--threshold", type=float, default=NEAR_DUPLICATE)
    args = parser.parse_args(argv)

    exclusions, report = build(args.datasets, args.threshold)
    path = exclusions_path()
    if path.exists():  # keep other datasets' rows when re-checking a subset
        old = pd.read_csv(path, dtype={"id": str, "matched_id": str})
        exclusions = pd.concat([old[~old["dataset"].isin(args.datasets)], exclusions])
    exclusions.sort_values(["dataset", "reason", "id"]).to_csv(path, index=False,
                                                               lineterminator="\n")  # fmt: skip
    report = pd.DataFrame(report + shortcut_report())
    LEAKAGE_REPORT.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(LEAKAGE_REPORT, index=False)
    print(report.to_string(index=False))
    print(f"\nwrote {path} ({len(exclusions)} train rows excluded) and {LEAKAGE_REPORT}")


if __name__ == "__main__":
    main()
