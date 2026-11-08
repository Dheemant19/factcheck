"""T38: the PUBHEALTH -> Climate-FEVER label map covers both label sets one to one."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from factcheck.data import get_dataset

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "transfer_climate_fever.py"


def load_script():
    spec = importlib.util.spec_from_file_location("transfer_climate_fever", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_label_map_is_one_to_one():
    label_map = load_script().LABEL_MAP
    assert set(label_map) == set(get_dataset("pubhealth").labels)
    assert set(label_map.values()) == set(get_dataset("climate_fever").labels)
    assert label_map["true"] == "supports" and label_map["false"] == "refutes"


def test_false_maps_to_a_fake_label():
    label_map, target = load_script().LABEL_MAP, get_dataset("climate_fever")
    assert target.binary_label(label_map["false"]) == 1
    assert target.binary_label(label_map["true"]) == 0
