"""Tests for the data loaders, cleaning and fixed splits (T03).

The cleaning, registry and split-logic tests need no data. The dataset tests run on the real
files and are skipped on a machine where ``python scripts/download_data.py`` has not been run.
"""

from __future__ import annotations

from functools import cache

import pandas as pd
import pytest

from factcheck.data import SPLITS, dataset_names, get_dataset, load_splits
from factcheck.data.cleaning import basic_clean
from factcheck.data.prepare import SCHEMA, read_split_file, stratified_split
from factcheck.paths import RAW_DIR

# Rows per split in the dataset papers (plan Section 6). Ours may differ by at most 1%,
# after dropping PUBHEALTH's broken label -1 rows.
PAPER_ROWS = {
    "covid": (6420, 2140, 2140),
    "liar": (10269, 1284, 1283),
    "pubhealth": (9832, 1225, 1235),
    "climate_fever": (0, 0, 1535),
}
# ISOT has no official split. 44,898 articles minus 631 empty-body articles minus 5,609
# exact duplicates leaves 38,658, split 80/10/10 (docs/m3_isot_and_splits.md).
ISOT_ROWS = 38_658

ALL = ["covid", "liar", "pubhealth", "isot", "climate_fever"]


@cache
def splits(name: str, variant: str = "raw") -> dict[str, pd.DataFrame]:
    """The full committed split; T13's train exclusions are tested in test_shortcuts.py."""
    return load_splits(name, variant, drop_excluded=False)


def needs_data(name: str):  # noqa: ANN201
    return pytest.mark.skipif(
        not (RAW_DIR / name).exists(), reason=f"run: python scripts/download_data.py {name}"
    )


def with_data(names: list[str]) -> list:
    return [pytest.param(n, marks=needs_data(n)) for n in names]


# --- cleaning -------------------------------------------------------------------------------


def test_basic_clean_repairs_text_without_changing_it():
    assert basic_clean("CafÃ© &amp; <b>bar</b>\n\n  ok ") == "Café & bar ok"
    assert basic_clean("COVID Is NOT a Hoax") == "COVID Is NOT a Hoax"  # no lower-casing
    assert basic_clean("a < b and c > d, I <3 it") == "a < b and c > d, I <3 it"
    assert basic_clean(None) == ""  # type: ignore[arg-type]


def test_social_masking_is_a_flag():
    post = "@CDCgov: wear masks https://t.co/abc www.cdc.gov mail me@x.org"
    assert basic_clean(post) == post
    assert basic_clean(post, mask_social=True) == "@USER: wear masks HTTPURL HTTPURL mail me@x.org"


# --- registry -------------------------------------------------------------------------------


def test_registry_has_every_dataset():
    assert dataset_names() == ALL
    with pytest.raises(KeyError, match="Unknown dataset"):
        get_dataset("nope")


def test_label_conventions():
    """M3 guide Section 3.2: fake = 1 for binary sets; LIAR ordered from false to true."""
    for name in ("covid", "isot"):
        assert get_dataset(name).label2id == {"real": 0, "fake": 1}
        assert get_dataset(name).fake_label_ids == [1]
    liar = get_dataset("liar")
    assert liar.labels == ("pants-fire", "false", "barely-true", "half-true", "mostly-true", "true")
    assert [liar.binary_label(x) for x in liar.labels] == [1, 1, 1, 0, 0, 0]
    pubhealth = get_dataset("pubhealth")
    assert pubhealth.label2id == {"false": 0, "mixture": 1, "true": 2, "unproven": 3}
    assert [pubhealth.binary_label(x) for x in pubhealth.labels] == [1, -1, 0, -1]
    assert get_dataset("climate_fever").fake_label_ids == [1]


# --- split logic (synthetic data) -----------------------------------------------------------


def test_stratified_split_sizes_and_llm_subset():
    df = pd.DataFrame({"id": [str(i) for i in range(5000)], "label_name": ["fake", "real"] * 2500})
    table = stratified_split(df, llm_subset_size=200)
    counts = table["split"].value_counts()
    assert (counts["train"], counts["validation"], counts["test"]) == (4000, 500, 500)
    subset = table[table["in_llm_subset"]]
    assert len(subset) == 200 and set(subset["split"]) == {"test"}
    labels = df.set_index("id").loc[subset["id"], "label_name"]
    assert labels.value_counts().to_dict() == {"fake": 100, "real": 100}
    assert stratified_split(df, llm_subset_size=200).equals(table)  # deterministic


# --- the real datasets ----------------------------------------------------------------------


@pytest.mark.parametrize("name", with_data(ALL))
def test_schema(name):
    info = get_dataset(name)
    for split, df in splits(name).items():
        assert list(df.columns[: len(SCHEMA)]) == SCHEMA, split
        if df.empty:
            continue
        assert df["id"].map(type).eq(str).all()
        assert df["text"].str.len().gt(0).all(), f"{split}: empty text"
        assert df["label"].between(0, info.num_labels - 1).all()
        assert (df["label"] == df["label_name"].map(info.label2id)).all()
        assert (df["binary_label"] == df["label_name"].map(info.binary_label)).all()


@pytest.mark.parametrize("name", with_data(list(PAPER_ROWS)))
def test_sizes_match_the_papers(name):
    for split, expected in zip(SPLITS, PAPER_ROWS[name], strict=True):
        assert len(splits(name)[split]) == pytest.approx(expected, rel=0.01), split


@pytest.mark.parametrize("name", with_data(ALL))
def test_no_id_in_two_splits_and_split_file_matches(name):
    ids = pd.concat([df["id"] for df in splits(name).values()])
    assert ids.is_unique
    assert set(ids) == set(read_split_file(name)["id"])


@pytest.mark.parametrize("name", with_data(["isot"]))
def test_isot_split(name):
    parts = splits(name)
    total = sum(len(df) for df in parts.values())
    assert total == ISOT_ROWS
    assert len(parts["train"]) == pytest.approx(0.8 * total, abs=2)
    for df in parts.values():  # stratified: every split has the same share of fake
        assert df["binary_label"].mean() == pytest.approx(0.4517, abs=0.002)
    subset = parts["test"][parts["test"]["in_llm_subset"]]
    assert len(subset) == 2000
    assert subset["binary_label"].mean() == pytest.approx(0.4517, abs=0.002)
    assert not parts["train"]["in_llm_subset"].any()


@pytest.mark.parametrize("name", with_data(["pubhealth"]))
def test_pubhealth_never_feeds_the_explanation(name):
    """The model text comes from the source's claim field only. The explanation usually
    states the verdict; in 4 source rows it is just a copy of the claim, which is harmless."""
    source = pd.concat(
        pd.read_parquet(RAW_DIR / name / f"default/{split}/0000.parquet") for split in SPLITS
    )
    source = source[source["label"] >= 0]  # broken rows (label -1) share junk ids
    claims = source.assign(id=source["claim_id"].astype(str).str.strip()).set_index("id")["claim"]
    for df in splits(name).values():
        assert (df["text"] == df["id"].map(claims).map(basic_clean)).all()
        assert (df["text"] == df["explanation"]).sum() <= 4


@pytest.mark.parametrize("name", with_data(["liar"]))
def test_liar_metadata_without_credit_history(name):
    df = splits(name)["train"]
    assert {"meta_speaker", "meta_party", "meta_subject", "meta_context"} <= set(df.columns)
    assert not [c for c in df.columns if c.endswith("_counts")]


@pytest.mark.parametrize("name", with_data(["covid"]))
def test_covid_clean_variant_masks_links_and_handles(name):
    raw, clean = splits(name)["train"], splits(name, "clean")["train"]
    assert raw["text"].str.contains("http").any()
    assert not clean["text"].str.contains(r"https?://|(?<![\w@])@(?!USER)\w", regex=True).any()
    assert (raw["id"] == clean["id"]).all()


@pytest.mark.parametrize("name", with_data(["climate_fever"]))
def test_climate_fever_evidence_and_cache_round_trip(name):
    cached = splits(name)["test"]
    assert cached["evidence"].map(lambda ev: isinstance(ev, list) and len(ev) == 5).all()
    fresh = load_splits(name, use_cache=False)["test"]
    pd.testing.assert_frame_equal(cached, fresh)
