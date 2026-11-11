"""Dataset registry (owner M3, T03).

Every dataset is described once, by a ``DatasetInfo``, and registered with the function that
reads its raw files. ``get_dataset(name)`` returns that info: the app, the training code and
the metrics code read ``labels`` (label names, indexed by class id), ``label2id``,
``fake_label_ids`` (class ids that count as false) and ``max_length`` from it.

Adding a dataset means writing one reader function in ``readers.py`` and decorating it with
``@register(...)``; nothing else changes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# A reader takes data/raw/<name>/ and returns one DataFrame with every usable row and at
# least the columns id, text_raw and label_name (plus split, if the dataset has official ones).
Reader = Callable[[Path], pd.DataFrame]


@dataclass(frozen=True)
class DatasetInfo:
    """Everything the rest of the project needs to know about one dataset."""

    name: str
    labels: tuple[str, ...]  # label names; the position is the class id
    fake_labels: frozenset[str]  # binary_label 1 (false / fake) in cross-domain tests
    real_labels: frozenset[str]  # binary_label 0 (true / real); any other label gets -1
    max_length: int  # tokens; plan Section 7, confirmed in the EDA notebook (T09)
    official_split: bool  # False: we make the split ourselves (ISOT)
    social_media: bool  # True: the clean variant masks URLs and @mentions
    reader: Reader

    @property
    def num_labels(self) -> int:
        return len(self.labels)

    @property
    def label2id(self) -> dict[str, int]:
        return {label: i for i, label in enumerate(self.labels)}

    @property
    def id2label(self) -> dict[int, str]:
        return dict(enumerate(self.labels))

    @property
    def fake_label_ids(self) -> list[int]:
        """Class ids that count as false, e.g. for P(false) in the app."""
        return [i for i, label in enumerate(self.labels) if label in self.fake_labels]

    @property
    def real_label_ids(self) -> list[int]:
        """Class ids that count as true, e.g. for P(true) in the app."""
        return [i for i, label in enumerate(self.labels) if label in self.real_labels]

    def binary_label(self, label_name: str) -> int:
        """1 = false/fake, 0 = true/real, -1 = left out of binary (cross-domain) tests."""
        if label_name in self.fake_labels:
            return 1
        if label_name in self.real_labels:
            return 0
        return -1


DATASETS: dict[str, DatasetInfo] = {}


def register(
    name: str,
    *,
    labels: tuple[str, ...],
    fake_labels: set[str],
    real_labels: set[str],
    max_length: int,
    official_split: bool = True,
    social_media: bool = False,
) -> Callable[[Reader], Reader]:
    """Decorator that registers a reader function under ``name``."""
    unknown = (fake_labels | real_labels) - set(labels)
    if unknown:
        raise ValueError(f"{name}: binary mapping uses unknown labels {sorted(unknown)}")

    def wrap(reader: Reader) -> Reader:
        DATASETS[name] = DatasetInfo(
            name=name,
            labels=labels,
            fake_labels=frozenset(fake_labels),
            real_labels=frozenset(real_labels),
            max_length=max_length,
            official_split=official_split,
            social_media=social_media,
            reader=reader,
        )
        return reader

    return wrap


def _register_builtin() -> None:
    import factcheck.data.readers  # noqa: F401  (importing it registers the five datasets)


def dataset_names() -> list[str]:
    """Names of every registered dataset, e.g. ``["covid", "liar", ...]``."""
    _register_builtin()
    return list(DATASETS)


def get_dataset(name: str) -> DatasetInfo:
    """Info for one dataset. Example: ``get_dataset("liar").labels[0] == "pants-fire"``."""
    _register_builtin()
    try:
        return DATASETS[name]
    except KeyError:
        raise KeyError(f"Unknown dataset {name!r}; choose from {list(DATASETS)}") from None
