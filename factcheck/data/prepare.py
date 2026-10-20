"""Cleaning and fixed train/validation/test splits (owner M3, T03).

    from factcheck.data import load_splits
    splits = load_splits("liar")                    # {"train": df, "validation": df, "test": df}
    test = load_splits("covid", variant="clean")["test"]

Every DataFrame has the standard schema (M3 guide Section 3.1), in this column order:

    id             stable string id (same id in the raw and clean variants)
    text           model input after cleaning
    text_raw       the text exactly as published (shortcut study, app)
    label          class id, the position of label_name in get_dataset(name).labels
    label_name     e.g. "pants-fire", "fake", "mixture"
    binary_label   1 = false/fake, 0 = true/real, -1 = not used in binary tests (E8)
    in_llm_subset  True for the fixed 2,000-article ISOT test subset scored by the 8B LLM

followed by dataset-specific columns: ``meta_*`` (LIAR), ``main_text`` and ``explanation``
(PUBHEALTH; never model input), ``evidence`` (Climate-FEVER), ``subject`` and ``date`` (ISOT;
never model input). Climate-FEVER is test-only, so its train and validation frames are empty.

Which row goes to which split is read from the committed files ``data/splits/<name>.csv``
(columns id, split, in_llm_subset), so the cluster and every laptop use identical rows. They
are written once by ``python scripts/make_splits.py``; loading never re-splits.
"""

from __future__ import annotations

import hashlib
import os
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from factcheck.data.cleaning import basic_clean
from factcheck.data.registry import DatasetInfo, get_dataset
from factcheck.paths import CACHE_DIR, RAW_DIR, SPLITS_DIR

SPLITS = ("train", "validation", "test")
VARIANTS = ("raw", "clean")
SCHEMA = ["id", "text", "text_raw", "label", "label_name", "binary_label", "in_llm_subset"]

SPLIT_SEED = 42  # ISOT: stratified 80/10/10, then a stratified LLM subset of the test split
LLM_SUBSET_SIZE = 2000

Splits = dict[str, pd.DataFrame]


def load_splits(name: str, variant: str = "raw", use_cache: bool = True) -> Splits:
    """Load one dataset in the standard schema, split by its committed split file.

    ``variant="raw"`` applies only ``basic_clean``. ``variant="clean"`` also masks URLs and
    @mentions in social-media posts (COVID) and, for ISOT, removes source shortcuts (T13).

    The prepared table is cached as parquet in ``CACHE_DIR/datasets/`` (cleaning ISOT takes
    about a minute; reading the cache about a second). The cache key covers the data code,
    the raw files and the split file, so any change rebuilds it automatically.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, not {variant!r}")
    info = get_dataset(name)
    cache = cache_path(name, variant) if use_cache else None
    if cache is not None and cache.exists():
        df = read_cache(cache)
    else:
        df = prepare(info, variant)
        if cache is not None:
            write_cache(df, cache)
    return {
        split: df[df["split"] == split].drop(columns="split").reset_index(drop=True)
        for split in SPLITS
    }


def prepare(info: DatasetInfo, variant: str) -> pd.DataFrame:
    """Read, check, split and clean one dataset: one table with a ``split`` column."""
    df = attach_split(read_raw(info.name), read_split_file(info.name), info)
    df["text"] = clean_texts(df["text_raw"], info, variant)
    df["label"] = df["label_name"].map(info.label2id).astype("int64")
    df["binary_label"] = df["label_name"].map(info.binary_label).astype("int64")
    extras = [c for c in df.columns if c not in SCHEMA and c != "split"]
    return df[SCHEMA + extras + ["split"]]


def read_raw(name: str) -> pd.DataFrame:
    """Every usable row of a dataset from data/raw/<name>/, checked but not cleaned or split."""
    info = get_dataset(name)
    df = info.reader(RAW_DIR / name)
    duplicated = df["id"][df["id"].duplicated()]
    if not duplicated.empty:
        raise ValueError(f"{name}: {len(duplicated)} duplicate ids, e.g. {duplicated.iloc[0]!r}")
    unknown = set(df["label_name"]) - set(info.labels)
    if unknown:
        raise ValueError(f"{name}: unknown labels {sorted(unknown)}; expected {info.labels}")
    return df


def clean_texts(texts: pd.Series, info: DatasetInfo, variant: str) -> pd.Series:
    """Apply the cleaning for one variant to a column of raw texts."""
    if variant == "clean" and info.name == "isot":
        raise NotImplementedError(
            "variant='clean' for ISOT removes source shortcuts (task T13), which is not "
            "written yet. Use variant='raw' until then."
        )
    mask_social = variant == "clean" and info.social_media
    return texts.map(partial(basic_clean, mask_social=mask_social))


# --- cache ----------------------------------------------------------------------------------

CODE_FILES = [Path(__file__).with_name(f) for f in ("cleaning.py", "readers.py", "registry.py")]


def cache_path(name: str, variant: str) -> Path:
    """CACHE_DIR/datasets/<name>-<variant>-<key>.parquet, with a key that changes whenever the
    data code, the raw files or the split file change."""
    key = hashlib.sha1()
    for path in [Path(__file__), *CODE_FILES, split_path(name)]:
        key.update(path.read_bytes() if path.exists() else b"missing")
    for path in sorted((RAW_DIR / name).rglob("*")):
        if path.is_file():
            stat = path.stat()
            key.update(f"{path.relative_to(RAW_DIR)}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return CACHE_DIR / "datasets" / f"{name}-{variant}-{key.hexdigest()[:12]}.parquet"


def write_cache(df: pd.DataFrame, path: Path) -> None:
    """Write atomically (safe with several jobs at once) and delete outdated copies."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    stem = path.stem.rsplit("-", 1)[0]
    for old in path.parent.glob(f"{stem}-*.parquet"):
        if old != path:
            old.unlink(missing_ok=True)


def read_cache(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    for column in df.columns:  # parquet returns list columns (evidence) as numpy arrays
        if len(df) and df[column].dtype == object and isinstance(df[column].iloc[0], np.ndarray):
            df[column] = df[column].map(list)
    return df


# --- split files ----------------------------------------------------------------------------


def split_path(name: str) -> Path:
    return SPLITS_DIR / f"{name}.csv"


def read_split_file(name: str) -> pd.DataFrame:
    """The committed id -> split table for one dataset."""
    path = split_path(name)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Create it with: python scripts/make_splits.py")
    table = pd.read_csv(path, dtype={"id": str, "split": str, "in_llm_subset": bool})
    bad = set(table["split"]) - set(SPLITS)
    if bad:
        raise ValueError(f"{path}: unknown split names {sorted(bad)}")
    return table


def write_split_file(name: str, table: pd.DataFrame) -> Path:
    path = split_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    table[["id", "split", "in_llm_subset"]].to_csv(path, index=False, lineterminator="\n")
    return path


def make_split_table(name: str) -> pd.DataFrame:
    """Build (not load) the id -> split table: the official split where the dataset has one,
    otherwise a stratified 80/10/10 split with a stratified LLM test subset (ISOT)."""
    info = get_dataset(name)
    df = read_raw(name)
    if info.official_split:
        return df[["id", "split"]].assign(in_llm_subset=False)
    return stratified_split(df)


def stratified_split(
    df: pd.DataFrame, seed: int = SPLIT_SEED, llm_subset_size: int = LLM_SUBSET_SIZE
) -> pd.DataFrame:
    """80/10/10 split stratified by label, plus a stratified subset of the test split."""
    train, rest = train_test_split(df, test_size=0.2, stratify=df["label_name"], random_state=seed)
    val, test = train_test_split(
        rest, test_size=0.5, stratify=rest["label_name"], random_state=seed
    )
    llm, _ = train_test_split(
        test, train_size=llm_subset_size, stratify=test["label_name"], random_state=seed
    )
    split = pd.Series("train", index=df.index)
    split[val.index] = "validation"
    split[test.index] = "test"
    return pd.DataFrame(
        {"id": df["id"], "split": split, "in_llm_subset": df.index.isin(llm.index)}
    ).reset_index(drop=True)


def attach_split(df: pd.DataFrame, table: pd.DataFrame, info: DatasetInfo) -> pd.DataFrame:
    """Add the committed split to every row, failing loudly if data and split file disagree."""
    missing = ~df["id"].isin(table["id"])
    extra = ~table["id"].isin(df["id"])
    if missing.any() or extra.any():
        raise ValueError(
            f"{info.name}: the raw data does not match {split_path(info.name)} "
            f"({missing.sum()} rows not in the split file, {extra.sum()} split-file ids not in "
            "the data). Re-download the raw files; regenerate the split file only if the "
            "whole team agrees, because every result depends on it."
        )
    merged = df.rename(columns={"split": "official_split"}).merge(
        table, on="id", how="left", validate="one_to_one"
    )
    if "official_split" in merged:
        moved = merged["official_split"] != merged["split"]
        if moved.any():
            raise ValueError(f"{info.name}: {moved.sum()} rows differ from the official split")
        merged = merged.drop(columns="official_split")
    return merged
