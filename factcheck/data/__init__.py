"""Datasets: one loader per dataset behind a common registry; cleaning; shortcut removal.

Owner: M3 (T03 loaders and fixed splits, T13 shortcut removal, T15 LIAR metadata/evidence).
Raw files come from ``python scripts/download_data.py`` (T02) into ``data/raw/<dataset>/``.

The Streamlit app imports ``get_dataset`` and ``load_splits`` from here, so keep those names.
"""

from factcheck.data.prepare import load_splits
from factcheck.data.registry import get_dataset

__all__ = ["get_dataset", "load_splits"]
