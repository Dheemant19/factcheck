"""Write the fixed split files data/splits/<dataset>.csv (task T03). Run ONCE, then commit.

    python scripts/make_splits.py              # every dataset; existing files are kept
    python scripts/make_splits.py isot         # just these
    python scripts/make_splits.py --check      # verify the committed files match your raw data
    python scripts/make_splits.py --force      # overwrite (only with the team's agreement)

Each file has the columns id, split, in_llm_subset. COVID, LIAR and PUBHEALTH keep their
official splits and Climate-FEVER is test-only; ISOT gets a stratified 80/10/10 split (seed 42)
and a stratified 2,000-article test subset for the LLM runs. The loaders read these files
instead of re-splitting, so every machine trains and tests on identical rows.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from factcheck.data import dataset_names, load_splits  # noqa: E402
from factcheck.data.prepare import (  # noqa: E402
    make_split_table,
    split_path,
    write_split_file,
)


def summary(name: str) -> str:
    """Row counts per split (and the LLM subset), loaded back through the real loader."""
    splits = load_splits(name)
    counts = "  ".join(f"{split} {len(df):>6,}" for split, df in splits.items())
    llm = int(splits["test"]["in_llm_subset"].sum())
    return f"{name:14s} {counts}" + (f"  llm_subset {llm:,}" if llm else "")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("datasets", nargs="*", help=f"any of {dataset_names()} (default: all)")
    parser.add_argument("--check", action="store_true", help="verify only, write nothing")
    parser.add_argument("--force", action="store_true", help="overwrite existing split files")
    args = parser.parse_args()

    names = args.datasets or dataset_names()
    unknown = sorted(set(names) - set(dataset_names()))
    if unknown:
        parser.error(f"unknown dataset(s) {unknown}; choose from {dataset_names()}")

    for name in names:
        path = split_path(name)
        if not args.check and (args.force or not path.exists()):
            write_split_file(name, make_split_table(name))
            print(f"wrote {path}")
        print(summary(name))  # loading checks that the raw data matches the split file


if __name__ == "__main__":
    main()
