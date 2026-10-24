"""Tests for shortcut removal and the leakage checks (T13)."""

from __future__ import annotations

import pandas as pd
import pytest

from factcheck.data import load_splits
from factcheck.data.leakage import exact_exclusions, normalise
from factcheck.data.prepare import excluded_ids
from factcheck.data.shortcuts import remove_shortcuts
from factcheck.paths import RAW_DIR

# (raw text, cleaned text): one real example per rule, taken from ISOT articles.
EXAMPLES = [
    ("Pup gift. SOCHI, Russia (Reuters) - Russian President Vladimir Putin",
     "Pup gift. Russian President Vladimir Putin"),
    ("Talks. NEW YORK/WASHINGTON (Reuters) - The head of the bank told Reuters on Monday.",
     "Talks. The head of the bank told on Monday."),
    ("By Alison Frankel (Reuters) - Donald Trump", "By Alison Frankel Donald Trump"),
    ("It ended. READ MORE SYRIA NEWS AT: 21st Century Wire SYRIA Files", "It ended."),
    ("Done.SUPPORT OUR WORK BY SUBSCRIBING & BECOMING A MEMBER @21WIRE.TV", "Done."),
    ("He spoke.Via: Breitbart News", "He spoke."),
    ("Wow pic.twitter.com/XtZW5 Donald J. Trump (@realDonaldTrump) December 23, 2017He left.",
     "Wow Donald J. Trump He left."),
    ("See https://t.co/abc and @CNN now.", "See and now."),
    ("Ugly.Featured image via Win McNamee/Getty Images", "Ugly."),
    ("WATCH: Trump Lies Again [VIDEO]. Text", "Trump Lies Again. Text"),
    ("It rose 2% (1600 GMT) today.", "It rose 2% today."),
    ("He didn t know Trump s plan, we re told, I m sure.",
     "He didn't know Trump's plan, we're told, I'm sure."),
]  # fmt: skip


@pytest.mark.parametrize(("raw", "clean"), EXAMPLES)
def test_rules_remove_fingerprints(raw, clean):
    assert remove_shortcuts(raw)[0] == clean


def test_rules_keep_ordinary_text():
    text = "The money went via a charity, officials said on Wednesday in the U.S. capital."
    assert remove_shortcuts(text) == (text, [])
    assert remove_shortcuts("A t-shirt costs 5 m.")[0] == "A t-shirt costs 5 m."


def test_normalise():
    assert normalise("COVID-19 is FAKE!!  ") == "covid 19 is fake"


def test_exact_exclusions_only_touch_train():
    splits = {
        "train": pd.DataFrame({"id": ["t1", "t2", "t3", "t4", "t5"], "label": [0, 0, 0, 1, 0],
                               "text": ["Same as test!", "unique", "dup", "dup.", "Unique"]}),
        "validation": pd.DataFrame({"id": ["v1"], "label": [0], "text": ["other"]}),
        "test": pd.DataFrame({"id": ["x1"], "label": [1], "text": ["same as TEST"]}),
    }  # fmt: skip
    found = exact_exclusions(splits).set_index("id")["reason"].to_dict()
    assert found == {
        "t1": "same_text_as_heldout",  # equals the test row after normalising
        "t3": "conflicting_labels_in_train",  # "dup" (label 0) vs "dup." (label 1)
        "t4": "conflicting_labels_in_train",
        "t5": "duplicate_in_train",  # repeats t2 with the same label; t2 is kept
    }


@pytest.mark.skipif(not (RAW_DIR / "isot").exists(), reason="no raw ISOT data")
def test_isot_clean_variant():
    raw, clean = load_splits("isot", "raw"), load_splits("isot", "clean")
    for split in raw:
        assert (raw[split]["id"] == clean[split]["id"]).all()  # same rows, same order
        assert not clean[split]["text"].str.contains("(Reuters)", regex=False).any()
        assert not clean[split]["text"].str.contains("Featured image", regex=False).any()


@pytest.mark.parametrize("name", ["covid", "liar", "pubhealth", "isot"])
def test_exclusions_never_touch_validation_or_test(name):
    if not (RAW_DIR / name).exists():
        pytest.skip(f"no raw {name} data")
    excluded = excluded_ids(name)
    full = load_splits(name, drop_excluded=False)
    kept = load_splits(name)
    for split in ("validation", "test"):
        assert not excluded & set(full[split]["id"])
        assert len(kept[split]) == len(full[split])
    assert len(kept["train"]) == len(full["train"]) - len(excluded & set(full["train"]["id"]))
