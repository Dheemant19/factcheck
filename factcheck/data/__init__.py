"""Datasets: one loader per dataset behind a common registry; cleaning; shortcut removal.

Owner: M3 (T03 loaders and fixed splits, T13 shortcut removal, T15 LIAR metadata/evidence).
Raw files come from ``python scripts/download_data.py`` (T02) into ``data/raw/<dataset>/``.

    from factcheck.data import load_splits, get_dataset
    splits = load_splits("liar")              # {"train": df, "validation": df, "test": df}
    get_dataset("liar").labels                # ("pants-fire", "false", ..., "true")

``load`` is the same function as ``load_splits`` (the name used in the M3 task guide).
The Streamlit app imports ``get_dataset`` and ``load_splits`` from here, so keep those names.
"""

from factcheck.data.prepare import SPLITS, load_splits
from factcheck.data.registry import DatasetInfo, dataset_names, get_dataset

load = load_splits

__all__ = ["SPLITS", "DatasetInfo", "dataset_names", "get_dataset", "load", "load_splits"]
