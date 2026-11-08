"""Style stripping (T59): the surface layer goes, the words stay, and every model family and
the app read the same stripped text."""

from __future__ import annotations

import pandas as pd
import pytest

from factcheck import serve
from factcheck.inputs import build_inputs, features
from factcheck.style import strip_style


@pytest.mark.parametrize(
    "post, plain",
    [
        ("RT @WHO: New cases https://t.co/abc", "new cases"),
        ("#COVID19 cases rise in #New_York", "covid19 cases rise in new york"),
        ("Masks work. #StaySafe #Corona #Covid", "masks work"),
        ("HTTPURL @USER the vaccine is SAFE!!!", "the vaccine is safe"),
        ("Says 5G spreads the virus 🦠📡", "5g spreads the virus"),
        ('A "miracle" cure?? Doctors\' advice...', "a miracle cure doctors advice"),
        ("It's what they don't tell you", "it's what they don't tell you"),
        ("", ""),
    ],
)
def test_strip_style_keeps_only_the_words(post, plain):
    assert strip_style(post) == plain


def test_strip_style_never_fails_on_non_text():
    assert strip_style(None) == ""  # type: ignore[arg-type]


def test_inputs_switch_strips_the_model_text_but_keeps_the_claim():
    cfg = {"data": {"dataset": "covid"}, "features": {"strip_style": True}}
    assert features(cfg)["strip_style"] is True
    df = pd.DataFrame({"text": ["#Covid cases UP!! https://t.co/x"], "label": [0]})
    out = build_inputs(df, cfg)
    assert out["text"].tolist() == ["covid cases up"]
    assert out["claim"].tolist() == ["#Covid cases UP!! https://t.co/x"]  # for retrieval
    off = build_inputs(df, {"data": {"dataset": "covid"}})
    assert off["text"].tolist() == df["text"].tolist()


def test_serve_prepares_claims_like_training():
    assert serve.strips_style({"features": {"strip_style": True}})
    assert not serve.strips_style({"features": {}}) and not serve.strips_style({})
    assert serve.prepare(["#A!! b"], True) == ["a b"]
    assert serve.prepare(["#A!! b"], False) == ["#A!! b"]
