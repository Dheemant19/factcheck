"""Model inputs from config switches: LIAR metadata and evidence (owner M3, T15).

A run config can switch two extra-effort features on (plan Section 8.5, ablation E5):

    features:
      use_metadata: true    # LIAR only: "[speaker: ... | party: ... | subject: ... | context: ...] claim"
      use_evidence: false   # PUBHEALTH and Climate-FEVER only: claim + top-k evidence sentences
      evidence_k: 5
      evidence_source: own  # own (the claim's article / given sentences) or pool (Climate-FEVER)
      strip_style: false    # any dataset: links, @mentions, hashtags, emoji, "!", capitals removed
                            # from the text the model reads (``factcheck.style``, T59, owner M1)

All are off by default. ``load_inputs(cfg)`` is ``load_splits`` plus these switches, and every
model family calls it, so the Bi-LSTM, the encoders and the LLM see identical inputs:

    text        the first input: the claim, with the metadata prefix when use_metadata is on
    text_pair   the evidence sentences joined, most similar first (only when use_evidence is on)
    claim       the claim alone, whatever the switches (for retrieval and the app)

Encoders tokenise ``text`` and ``text_pair`` as a pair; models that read one sequence (the
Bi-LSTM) use ``single_text``. LIAR's five credit-history columns are never used: they count the
speaker's past verdicts and can leak the answer, so the loader does not even keep them.
"""

from __future__ import annotations

import pandas as pd

from factcheck.data import load_splits

FEATURE_DEFAULTS = {
    "use_metadata": False,
    "use_evidence": False,
    "evidence_k": 5,
    "evidence_source": "own",
    "strip_style": False,
}
METADATA_DATASETS = {"liar"}
EVIDENCE_DATASETS = {"pubhealth", "climate_fever"}
# prefix field -> LIAR column (job and state are left out: 21-28% missing, little signal)
METADATA_FIELDS = {
    "speaker": "meta_speaker",
    "party": "meta_party",
    "subject": "meta_subject",
    "context": "meta_context",
}

Splits = dict[str, pd.DataFrame]


def features(cfg: dict) -> dict:
    """The run's ``features`` block with defaults filled in, checked against the dataset."""
    given = cfg.get("features") or {}
    unknown = set(given) - set(FEATURE_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown features {sorted(unknown)}; known: {list(FEATURE_DEFAULTS)}")
    feats = {**FEATURE_DEFAULTS, **given}
    dataset = cfg["data"]["dataset"]
    if feats["use_metadata"] and dataset not in METADATA_DATASETS:
        raise ValueError(f"use_metadata is for {sorted(METADATA_DATASETS)} only, not {dataset}")
    if feats["use_evidence"] and dataset not in EVIDENCE_DATASETS:
        raise ValueError(f"use_evidence is for {sorted(EVIDENCE_DATASETS)} only, not {dataset}")
    return feats


def metadata_prefix(row: pd.Series) -> str:
    """The bracketed metadata of one LIAR row; a missing field reads "unknown".

    Example: [speaker: rick perry | party: republican | subject: immigration | context: radio]
    """
    parts = []
    for field, column in METADATA_FIELDS.items():
        value = str(row.get(column, "") or "").strip()
        if field == "speaker":
            value = value.replace("-", " ")  # LIAR speakers are slugs: "rick-perry"
        elif field == "subject":
            value = value.replace(",", ", ")  # "immigration,jobs"
        parts.append(f"{field}: {value or 'unknown'}")
    return "[" + " | ".join(parts) + "]"


def build_inputs(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """One split with the config's switches applied (a copy; ``df`` is not changed)."""
    feats = features(cfg)
    df = df.assign(claim=df["text"])
    if feats["strip_style"]:
        from factcheck.style import strip_style

        df["text"] = df["text"].map(strip_style)
    if feats["use_metadata"] and len(df):
        prefixes = df.apply(metadata_prefix, axis=1)
        df["text"] = prefixes + " " + df["claim"]
    if feats["use_evidence"]:
        from factcheck.retrieval import evidence_for  # needs the T08 evidence cache

        evidence = evidence_for(
            cfg["data"]["dataset"], df, int(feats["evidence_k"]), feats["evidence_source"]
        )
        df["text_pair"] = [" ".join(sentences) for sentences in evidence]
    return df


def load_inputs(cfg: dict) -> Splits:
    """``load_splits`` for the run's dataset and variant, with its feature switches applied."""
    data = cfg["data"]
    splits = load_splits(data["dataset"], variant=data.get("variant", "raw"))
    return {split: build_inputs(df, cfg) for split, df in splits.items()}


def single_text(df: pd.DataFrame) -> pd.Series:
    """``text`` followed by ``text_pair`` when there is one, for one-sequence models."""
    if "text_pair" not in df:
        return df["text"]
    return df["text"] + " " + df["text_pair"]
