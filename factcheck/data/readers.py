"""One reader per dataset (owner M3, T03; plan Section 6, M3 guide Section 10).

A reader turns the raw files in ``data/raw/<name>/`` (from ``scripts/download_data.py``) into
one DataFrame of every usable row, with at least:

    id          stable string id, unique across the whole dataset
    text_raw    the model input exactly as published (claim, post, or title + article)
    label_name  the label as a string from the registry's ``labels``
    split       train / validation / test, for datasets with an official split

Readers do no cleaning and no splitting; ``prepare.load_splits`` does both, the same way for
every dataset. Labels are mapped by string, never by a source's own integer order.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pandas as pd

from factcheck.data.registry import register
from factcheck.paths import RAW_DIR


def require(path: Path) -> Path:
    """Fail with a download hint if a raw file is missing (the app catches this error)."""
    if not path.exists():
        try:
            dataset = path.relative_to(RAW_DIR).parts[0]
        except ValueError:
            dataset = "<dataset>"
        raise FileNotFoundError(
            f"{path} not found. Download it with: python scripts/download_data.py {dataset}"
        )
    return path


def stable_id(text: str) -> str:
    """Short SHA-1 of a text, for datasets without official ids (ISOT)."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


# --- D1 COVID-19 fake news (Patwa et al., 2021) ---------------------------------------------

COVID_FILES = {
    "train": "Constraint_Train.csv",
    "validation": "Constraint_Val.csv",
    "test": "english_test_with_labels.csv",
}


@register(
    "covid",
    labels=("real", "fake"),
    fake_labels={"fake"},
    real_labels={"real"},
    max_length=128,
    social_media=True,
)
def read_covid(raw_dir: Path) -> pd.DataFrame:
    """Official 6,420 / 2,140 / 2,140 split. The official ids restart at 1 in every file,
    so our id is ``<split>-<official id>`` (e.g. ``validation-17``)."""
    frames = []
    for split, filename in COVID_FILES.items():
        df = pd.read_csv(require(raw_dir / filename), dtype={"id": str})
        frames.append(
            pd.DataFrame(
                {
                    "id": split + "-" + df["id"],
                    "split": split,
                    "text_raw": df["tweet"].fillna(""),
                    "label_name": df["label"].str.strip().str.lower(),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


# --- D2 LIAR (Wang, 2017) -------------------------------------------------------------------

LIAR_FILES = {"train": "train.tsv", "validation": "valid.tsv", "test": "test.tsv"}
LIAR_COLUMNS = [
    "id", "label", "statement", "subject", "speaker", "job_title", "state_info",
    "party_affiliation", "barely_true_counts", "false_counts", "half_true_counts",
    "mostly_true_counts", "pants_on_fire_counts", "context",
]  # fmt: skip
LIAR_META = {  # schema column -> LIAR column. The five *_counts columns are left out on
    "meta_speaker": "speaker",  # purpose: they summarise the speaker's past verdicts and
    "meta_party": "party_affiliation",  # can leak the answer (M3 guide, T15).
    "meta_subject": "subject",
    "meta_context": "context",
    "meta_job": "job_title",
    "meta_state": "state_info",
}


@register(
    "liar",
    labels=("pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"),
    fake_labels={"pants-fire", "false", "barely-true"},
    real_labels={"half-true", "mostly-true", "true"},
    max_length=128,
)
def read_liar(raw_dir: Path) -> pd.DataFrame:
    """Official 10,269 / 1,284 / 1,283 split. Labels run from most false to most true, so
    confusion matrices read naturally (this is not the Hugging Face integer order)."""
    frames = []
    for split, filename in LIAR_FILES.items():
        df = pd.read_csv(
            require(raw_dir / filename),
            sep="\t",
            header=None,
            names=LIAR_COLUMNS,
            dtype=str,
            quoting=csv.QUOTE_NONE,  # statements contain unbalanced quote marks
            keep_default_na=False,
        )
        out = pd.DataFrame(
            {
                "id": df["id"].str.removesuffix(".json"),
                "split": split,
                "text_raw": df["statement"],
                "label_name": df["label"].str.strip(),
            }
        )
        for column, source in LIAR_META.items():
            out[column] = df[source].str.strip()
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


# --- D3 PUBHEALTH (Kotonya and Toni, 2020) --------------------------------------------------

PUBHEALTH_FILES = {split: f"default/{split}/0000.parquet" for split in COVID_FILES}
PUBHEALTH_HF_LABELS = ("false", "mixture", "true", "unproven")  # the parquet's ClassLabel names


@register(
    "pubhealth",
    labels=("false", "mixture", "true", "unproven"),
    fake_labels={"false"},
    real_labels={"true"},
    max_length=384,  # claim + top-5 evidence sentences
)
def read_pubhealth(raw_dir: Path) -> pd.DataFrame:
    """Official split. Rows with label -1 are broken in the source (empty claims, shifted
    columns) and are dropped. ``explanation`` is kept as a column for scoring generated
    explanations, but it is NEVER model input: it usually states the verdict."""
    frames = []
    for split, filename in PUBHEALTH_FILES.items():
        df = pd.read_parquet(require(raw_dir / filename))
        df = df[df["label"].between(0, len(PUBHEALTH_HF_LABELS) - 1)]
        df = df[df["claim"].str.strip() != ""]
        frames.append(
            pd.DataFrame(
                {
                    "id": df["claim_id"].astype(str).str.strip(),
                    "split": split,
                    "text_raw": df["claim"],
                    "label_name": df["label"].map(dict(enumerate(PUBHEALTH_HF_LABELS))),
                    "main_text": df["main_text"],  # the fact-check article; evidence for T08
                    "explanation": df["explanation"],
                    "subjects": df["subjects"],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


# --- D4 ISOT fake and real news (Ahmed et al., 2017) ----------------------------------------

ISOT_FILES = {"fake": "Fake.csv", "real": "True.csv"}


@register(
    "isot",
    labels=("real", "fake"),
    fake_labels={"fake"},
    real_labels={"real"},
    max_length=512,  # head 128 + tail 382 tokens
    official_split=False,
)
def read_isot(raw_dir: Path) -> pd.DataFrame:
    """Fake.csv + True.csv as ``title. text``. We make the split (prepare.make_isot_split).

    Two decisions, both recorded in the README:
    * exact duplicates (same title and text) are dropped, keeping the first copy;
    * articles with an empty body are dropped. 630 fake rows but only 1 real row have one,
      so keeping them would hand the model a "title only means fake" shortcut.

    ``subject`` and ``date`` are kept for analysis and the app, but they are giveaways
    (subject values differ completely between classes), so they are never model input.
    """
    frames = []
    for label_name, filename in ISOT_FILES.items():
        df = pd.read_csv(require(raw_dir / filename), dtype=str, keep_default_na=False)
        df = df[df["text"].str.strip() != ""]
        frames.append(
            pd.DataFrame(
                {
                    "text_raw": df["title"] + ". " + df["text"],
                    "label_name": label_name,
                    "subject": df["subject"],
                    "date": df["date"],
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    df.insert(0, "id", df["text_raw"].map(stable_id))  # from RAW text: same id in every variant
    return df.drop_duplicates("id", keep="first").reset_index(drop=True)


# --- D5 Climate-FEVER (Diggelmann et al., 2020): extrapolation only -------------------------

CLIMATE_FEVER_FILE = "data/test-00000-of-00001.parquet"
CLIMATE_FEVER_HF_LABELS = ("SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED")


@register(
    "climate_fever",
    labels=("supports", "refutes", "not_enough_info", "disputed"),
    fake_labels={"refutes"},
    real_labels={"supports"},
    max_length=384,
)
def read_climate_fever(raw_dir: Path) -> pd.DataFrame:
    """1,535 claims, test split only (never trained on). Each claim has 5 evidence sentences."""
    df = pd.read_parquet(require(raw_dir / CLIMATE_FEVER_FILE))
    hf_labels = dict(enumerate(CLIMATE_FEVER_HF_LABELS))
    return pd.DataFrame(
        {
            "id": df["claim_id"].astype(str),
            "split": "test",
            "text_raw": df["claim"],
            "label_name": df["claim_label"].map(hf_labels).str.lower(),
            "evidence": df["evidences"].map(lambda ev: [e["evidence"] for e in ev]),
            "evidence_ids": df["evidences"].map(lambda ev: [e["evidence_id"] for e in ev]),
        }
    )
