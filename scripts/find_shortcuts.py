"""Find source shortcuts with a bag-of-words model, before and after cleaning (task T13).

    python scripts/find_shortcuts.py                  # ISOT and COVID, raw vs clean
    python scripts/find_shortcuts.py --datasets isot

For each dataset and variant it fits TF-IDF + logistic regression on the train split, scores
the validation split (never test) and lists the words with the largest weights for each class.
On raw ISOT those are source fingerprints ("reuters", "via", "image", "pic.twitter.com"); the
rules in factcheck/data/shortcuts.py were added until the top words were about content.

Writes to results/shortcuts/:
    tfidf_summary.csv            dataset, variant, validation accuracy and macro-F1, top words
    top_words_<dataset>_<variant>.csv
    isot_rule_hits.csv           per rule: how many fake and real ISOT articles it changed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from factcheck.data import get_dataset, load_splits  # noqa: E402
from factcheck.data.cleaning import basic_clean  # noqa: E402
from factcheck.data.shortcuts import SHORTCUT_RULES, remove_shortcuts  # noqa: E402
from factcheck.paths import RESULTS_DIR  # noqa: E402

OUT_DIR = RESULTS_DIR / "shortcuts"
TOP_N = 30
# Words keep inner apostrophes and dots ("didn't", "u.s"), and "(", ")", "@", "#" count as
# tokens, because those characters are themselves fingerprints in ISOT.
TOKEN_PATTERN = r"(?u)\b\w[\w'.]*\w\b|[@#()]"


def bag_of_words_probe(name: str, variant: str) -> tuple[dict, pd.DataFrame]:
    """Fit on train, score validation, return the summary row and the top words per class."""
    splits = load_splits(name, variant)
    train, val = splits["train"], splits["validation"]
    vectorizer = TfidfVectorizer(
        token_pattern=TOKEN_PATTERN, min_df=5, max_features=50_000, sublinear_tf=True
    )
    x_train = vectorizer.fit_transform(train["text"])
    model = LogisticRegression(max_iter=2000, C=4.0).fit(x_train, train["binary_label"])
    pred = model.predict(vectorizer.transform(val["text"]))

    words = vectorizer.get_feature_names_out()
    weights = model.coef_[0]  # positive = pushes towards fake (binary_label 1)
    fake, real = np.argsort(-weights)[:TOP_N], np.argsort(weights)[:TOP_N]
    top = pd.DataFrame({
        "rank": range(1, TOP_N + 1),
        "fake_word": words[fake], "fake_weight": weights[fake].round(2),
        "real_word": words[real], "real_weight": weights[real].round(2),
    })  # fmt: skip
    summary = {
        "dataset": name,
        "variant": variant,
        "val_accuracy": round(accuracy_score(val["binary_label"], pred), 4),
        "val_macro_f1": round(f1_score(val["binary_label"], pred, average="macro"), 4),
        "top_fake_words": " ".join(words[fake][:15]),
        "top_real_words": " ".join(words[real][:15]),
    }
    return summary, top


def isot_rule_hits() -> pd.DataFrame:
    """How many fake and real ISOT articles each rule changed (all splits)."""
    df = pd.concat(load_splits("isot", "raw").values(), ignore_index=True)
    fired = df["text_raw"].map(lambda t: remove_shortcuts(basic_clean(t))[1])
    rows = []
    for rule in SHORTCUT_RULES:
        hit = fired.map(lambda names, rule=rule: rule in names)
        rows.append({
            "rule": rule,
            "fake_articles": int(hit[df["label_name"] == "fake"].sum()),
            "real_articles": int(hit[df["label_name"] == "real"].sum()),
            "fake_share": round(hit[df["label_name"] == "fake"].mean(), 4),
            "real_share": round(hit[df["label_name"] == "real"].mean(), 4),
        })  # fmt: skip
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--datasets", nargs="+", default=["isot", "covid"])
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summaries = []
    for name in args.datasets:
        get_dataset(name)  # fail early on a typo
        for variant in ("raw", "clean"):
            summary, top = bag_of_words_probe(name, variant)
            top.to_csv(OUT_DIR / f"top_words_{name}_{variant}.csv", index=False)
            summaries.append(summary)
            print(f"\n{name} {variant}: validation accuracy {summary['val_accuracy']:.4f}, "
                  f"macro-F1 {summary['val_macro_f1']:.4f}")  # fmt: skip
            print(f"  fake words: {summary['top_fake_words']}")
            print(f"  real words: {summary['top_real_words']}")
    pd.DataFrame(summaries).to_csv(OUT_DIR / "tfidf_summary.csv", index=False)

    if "isot" in args.datasets:
        hits = isot_rule_hits()
        hits.to_csv(OUT_DIR / "isot_rule_hits.csv", index=False)
        print("\nISOT articles changed per rule:\n" + hits.to_string(index=False))
    print(f"\nwrote {OUT_DIR}")


if __name__ == "__main__":
    main()
