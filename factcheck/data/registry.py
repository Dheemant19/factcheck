"""Dataset registry (owner M3, T03).

``get_dataset(name)`` returns the info for one dataset. The app reads ``labels`` (label names,
indexed by class id), ``fake_label_ids`` (class ids that count as false) and ``max_length``.
"""

from __future__ import annotations


def get_dataset(name: str):  # noqa: ANN201
    raise NotImplementedError("T03 (M3): dataset registry")
