"""Cleaning and fixed train/validation/test splits (owner M3, T03).

``load_splits(name, variant)`` returns ``{"train": df, "validation": df, "test": df}``, each
with a ``text`` column; ``variant`` is "raw" or "clean" (shortcut-removed, T13).
"""

from __future__ import annotations


def load_splits(name: str, variant: str = "raw") -> dict:
    raise NotImplementedError("T03 (M3): loaders and fixed splits")
