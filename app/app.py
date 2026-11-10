"""Fact-Check Lab: paste a claim, get an explained verdict (T11, T22, T35, T58).

    streamlit run app/app.py

Check a claim: the verdict on the dataset's rating ladder, the words behind it twice over (SHAP
for a transformer, exact weights for the TF-IDF + logistic regression baseline, on the same
claim), the evidence (similar already fact-checked claims with their verdicts, and the closest
evidence sentences from PUBHEALTH's fact-check articles and Climate-FEVER), and an optional
second opinion from Qwen3-8B with its QLoRA adapter. Compare models up the ladder side by side,
run the shortcut demo (a Reuters dateline flips the raw-trained ISOT models), and read the
ladder of results and the specs board from results/runs.csv.

The screen shows what a person checking a claim needs: the verdict, how sure the model is, the
words behind it and the evidence. Run details (seeds, scores, timings, specs) live in Results.

Models are found under outputs/<run>/seed<N>/model; the engine is ``factcheck.serve``. The
theme lives in app/.streamlit/config.toml, plus the small stylesheet in ``STYLE``.
"""

from __future__ import annotations

import html
import importlib
import re
import sys
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from factcheck.data import get_dataset
from factcheck.data.prepare import load_splits
from factcheck.evaluate import check_specs
from factcheck.paths import OUTPUT_DIR
from factcheck.results import load_runs
from factcheck.retrieval import RETRIEVAL_DIR

EXAMPLE_CLAIMS = {  # link text -> (the claim it loads, the dataset whose models read it)
    "Hot water kills the virus": (
        "Drinking hot water every 15 minutes kills the coronavirus.",
        "covid",
    ),
    "CDC on masks": ("The CDC recommends that people wear masks in public settings.", "covid"),
    "Unemployment has doubled": (
        "Says the unemployment rate has doubled since the president took office.",
        "liar",
    ),
    "Vitamin D cure": ("Vitamin D supplements cure COVID-19 in 24 hours.", "covid"),
}
DATELINE = "WASHINGTON (Reuters) - "
# The engine (factcheck.serve) imports torch and transformers, which takes seconds on a cold
# start, so main() imports it after a skeleton of the page is on screen.
serve: Any = None
SOURCES = {  # where an evidence sentence comes from, as a phrase
    "pubhealth": "a PUBHEALTH fact-check article",
    "climate_fever": "the Climate-FEVER evidence set",
}

VIEWS = {
    "check": "Check a claim",
    "compare": "Compare models",
    "shortcut": "Shortcut demo",
    "results": "Results",
}
CLAIM_KEYS = ("claim_check", "claim_compare")
DATASETS = {
    "covid": "COVID",
    "liar": "LIAR",
    "pubhealth": "PUBHEALTH",
    "isot": "ISOT",
    "climate_fever": "Climate-FEVER",
}
STYLES = {  # the text a model was trained on (T59)
    "written": "As written",
    "stripped": "Style stripped",
}
DATASET_CHOICES = {  # what each dataset's models were trained to read
    "covid": "COVID-19 social media posts (COVID)",
    "liar": "Political statements (LIAR)",
    "pubhealth": "Public-health claims (PUBHEALTH)",
    "isot": "News articles (ISOT)",
}
MODEL_NAMES = {
    "digitalepidemiologylab/covid-twitter-bert-v2": "COVID-Twitter-BERT",
    "allenai/scibert_scivocab_uncased": "SciBERT",
    "Qwen/Qwen3-8B": "Qwen3-8B QLoRA",
    "tfidf": "TF-IDF + logistic regression",
    "distilbert-base-uncased": "DistilBERT",
    "bert-base-uncased": "BERT-base",
    "FacebookAI/roberta-base": "RoBERTa-base",
    "FacebookAI/roberta-large": "RoBERTa-large",
    "microsoft/deberta-v3-base": "DeBERTa-v3-base",
    "microsoft/deberta-v3-large": "DeBERTa-v3-large",
}

# The ladder of models (plan Section 8), bottom rung first.
RUNGS = ["TF-IDF", "Frozen embeddings", "Bi-LSTM", "Transformer encoder", "LLM (Qwen3-8B)"]

# Where each dataset label sits between false (0) and true (1). Labels with no place on
# that scale (PUBHEALTH "unproven", Climate-FEVER "not_enough_info") go last, in grey.
TRUTH_SCORE = {
    "pants-fire": 0.0,
    "fake": 0.0,
    "refutes": 0.0,
    "false": 0.15,
    "barely-true": 0.35,
    "half-true": 0.5,
    "mixture": 0.5,
    "disputed": 0.5,
    "mostly-true": 0.75,
    "true": 1.0,
    "real": 1.0,
    "supports": 1.0,
}
# Rung outlines from the theme's verdict palette.
RUNG_FILL = {
    "pants-fire": "#8a241d",
    "fake": "#b8332a",
    "refutes": "#b8332a",
    "false": "#b8332a",
    "barely-true": "#a04a1c",
    "half-true": "#8a5200",
    "mixture": "#8a5200",
    "disputed": "#8a5200",
    "mostly-true": "#3f6e3a",
    "true": "#0b6e4f",
    "real": "#0b6e4f",
    "supports": "#0b6e4f",
}
RUNG_NAMES = {
    "pants-fire": "Pants on fire",
    "not_enough_info": "Not enough info",
}
UNLIT_FILL, UNLIT_INK, OFF_SCALE_FILL, LIT_OUTLINE_INK = "#ebe6dd", "#52525b", "#52525b", "#18181b"
PAPER = "#faf8f5"
# A lit rung: the theme's pale verdict tint, its verdict-colour outline and darker ink.
RUNG_TINT = {"red": "#f8e3e0", "orange": "#f7ead0", "green": "#dcefe6", "gray": "#ebe6dd"}
RUNG_INK = {"red": "#9e2b23", "orange": "#8a5200", "green": "#0a5f44", "gray": "#18181b"}

# "Likely true" needs P(true) >= VERDICT_AT and "Likely false" needs P(false) > VERDICT_AT,
# each summed over the dataset's true or false labels. On a two-label dataset that is P(false)
# <= 0.4 and > 0.6; labels that are neither (PUBHEALTH mixture and unproven, Climate-FEVER
# disputed and not enough info) count towards neither side, so they pull a claim to Mixed.
VERDICT_AT = 0.6
MIXED = "Mixed / uncertain"
VERDICTS = {
    "Likely true": ("green", ":material/check_circle:"),
    MIXED: ("orange", ":material/help:"),
    "Likely false": ("red", ":material/cancel:"),
}

# The few things the native theme cannot do: the word-push bars (aligned rows, values in their
# own column, so nothing overlaps), section headings a step heavier, no heading link icons.
STYLE = """
<style>
[data-testid="stHeaderActionElements"] { display: none; }
::selection { background: #dde7f3; }
h3 { font-weight: 700 !important; }
.pushes { margin: 0.4rem 0 0.2rem; font-size: 1rem; }
.pushes .row { display: grid; grid-template-columns: minmax(6rem, 11rem) 1fr 3.6rem;
  align-items: center; column-gap: 0.9rem; padding: 0.22rem 0; }
.pushes .key { color: #52525b; font-size: 0.85rem; padding-bottom: 0.35rem; }
.pushes .key .ends { grid-column: 2 / 4; display: flex; justify-content: space-between;
  white-space: nowrap; gap: 0.6rem; }
[data-testid="stElementToolbar"] { display: none; }
.pushes .word { text-align: right; overflow: hidden; text-overflow: ellipsis;
  white-space: nowrap; }
.pushes .strong .word { font-weight: 700; }
.pushes .track { position: relative; height: 0.95rem; }
.pushes .zero { position: absolute; top: -0.3rem; bottom: -0.3rem; border-left: 1px solid #a8a29e; }
.pushes .bar { position: absolute; top: 0; height: 100%; border-radius: 2px; }
.pushes .false { background: #b8332a; }
.pushes .true { background: #0b6e4f; }
.pushes .value { font-family: 'IBM Plex Mono', monospace; font-size: 0.88rem;
  text-align: right; font-variant-numeric: tabular-nums; }
.source { color: #52525b; font-size: 0.88rem; }
/* Skeletons: where results will land, shown while models load and compute. */
@keyframes shimmer { from { background-position: 100% 0; } to { background-position: -100% 0; } }
.skel { background: linear-gradient(90deg, #ece7de 25%, #f5f1ea 37%, #ece7de 63%);
  background-size: 400% 100%; animation: shimmer 1.4s ease-in-out infinite; border-radius: 2px; }
@media (prefers-reduced-motion: reduce) { .skel { animation: none; } }
.skel-frame { border: 1px solid #d6d1c7; border-radius: 2px; padding: 1.4rem 1.2rem; }
.skel-stack > * + * { margin-top: 0.75rem; }
.skel-grid { display: grid; gap: 2.5rem; margin-top: 2rem; }
.skel-bars { display: grid; grid-template-columns: minmax(6rem, 11rem) 1fr 3.6rem;
  column-gap: 0.9rem; row-gap: 0.7rem; align-items: center; }
/* The chosen-models row scrolls sideways; show its scrollbar as a thin pill under the chips. */
[data-testid="stMultiSelectTagsContainer"] { scrollbar-width: thin !important;
  scrollbar-color: #8fa3bd transparent; padding-bottom: 3px; }
</style>
"""


# --------------------------------------------------------------------------- models


@st.cache_data(ttl=60)
def logged_val_f1() -> dict[tuple[str, int], float]:
    """Validation macro-F1 of every logged (run, seed), read from results/runs.csv. Used only
    to pick each run's best seed and the default model, never shown next to a verdict. Not
    the test score, so the test set never picks the model the app (and T40) measures."""
    runs = load_runs()
    val = runs[(runs["split"] == "validation") & runs["subset"].isna()]
    return {(r.run_name, int(r.seed)): float(r.macro_f1) for r in val.itertuples()}


@st.cache_data(max_entries=128)
def run_config(model_dir: str) -> dict:
    return serve.run_config(model_dir)


def val_f1(model_dir: str) -> float | None:
    run = run_config(model_dir)["run"]
    return logged_val_f1().get((run["name"], int(run["seed"])))


def dataset_of(model_dir: str) -> str:
    return run_config(model_dir)["data"]["dataset"]


def variant_of(model_dir: str) -> str:
    return run_config(model_dir)["data"].get("variant", "raw")


def style_of(model_dir: str) -> str:
    """'stripped' for a run trained on style-stripped text (T59), else 'written'."""
    return "stripped" if serve.strips_style(run_config(model_dir)) else "written"


def is_baseline(model_dir: str) -> bool:
    return run_config(model_dir)["model"].get("family") == "classic"


def ladder_rank(model_dir: str) -> int:
    """Where a servable model sits on the ladder: baseline, encoder, then the LLM."""
    return 0 if is_baseline(model_dir) else 2 if serve.is_qlora(model_dir) else 1


def find_models(include_llm: bool = False) -> dict[str, Path]:
    """Every model the app can run, one per run (its seed with the best validation macro-F1),
    by dataset, up the ladder (TF-IDF, then encoders best first, then Qwen3-8B).

    QLoRA adapters (Qwen3-8B, about 6 GB of GPU memory) are listed only when the LLM is on.
    """
    found = [
        p
        for p in OUTPUT_DIR.glob("*/seed*/model")
        if serve.is_servable(p) and (include_llm or not serve.is_qlora(p))
    ]
    order = list(DATASETS)

    def key(p: Path) -> tuple:
        dataset = dataset_of(str(p))
        rank = order.index(dataset) if dataset in order else 99
        return (rank, ladder_rank(str(p)), -(val_f1(str(p)) or 0.0))

    best: dict[str, Path] = {}
    for p in sorted(found, key=key):
        best.setdefault(run_config(str(p))["run"]["name"], p)  # the best seed of each run
    return {str(p): p for p in best.values()}


def llm_adapters() -> dict[str, str]:
    """Dataset -> the QLoRA adapter folder for it (the first seed found)."""
    adapters = {}
    for p in sorted(OUTPUT_DIR.glob("qwen3_8b_qlora_*/seed*/model")):
        if serve.is_servable(p):
            adapters.setdefault(run_config(str(p))["data"]["dataset"], str(p))
    return adapters


def model_label(model_dir: str) -> str:
    name = run_config(model_dir)["model"]["name"]
    return MODEL_NAMES.get(name, name.split("/")[-1])


def model_name(model_dir: str) -> str:
    """The model as a person would name it: 'RoBERTa-base on cleaned text'."""
    cfg = run_config(model_dir)
    name = model_label(model_dir)
    if (cfg.get("features") or {}).get("use_evidence"):
        name += " with evidence"
    if variant_of(model_dir) != "raw":
        name += " on cleaned text"
    elif dataset_of(model_dir) == "isot":
        name += " on raw text"
    return name


def counterpart(model_dir: str, models: dict[str, Path]) -> str | None:
    """The model to read the same claim next to this one: the TF-IDF baseline of the same
    dataset for a transformer, the best transformer of that dataset for the baseline."""
    dataset, variant = dataset_of(model_dir), variant_of(model_dir)
    same = [
        m
        for m in models
        if m != model_dir and dataset_of(m) == dataset and style_of(m) == style_of(model_dir)
    ]
    if is_baseline(model_dir):
        pool = [m for m in same if not is_baseline(m) and not serve.is_qlora(m)]
    else:
        pool = [m for m in same if is_baseline(m)]
    pool.sort(key=lambda m: variant_of(m) != variant)  # same variant first, order kept
    return pool[0] if pool else None


@st.cache_resource(max_entries=4, show_spinner=False)
def load_model(model_dir: str) -> serve.Classifier | serve.Baseline:
    return serve.load(model_dir)  # an encoder or a baseline


@st.cache_resource(max_entries=1, show_spinner=False)
def load_llm(model_dir: str) -> serve.Classifier:
    return serve.Classifier(model_dir)  # 4-bit Qwen3-8B + adapter; one dataset at a time


def classifier(model_dir: str) -> serve.Classifier | serve.Baseline:
    """The loaded model (the views show skeletons while a first load runs)."""
    model = load_llm(model_dir) if serve.is_qlora(model_dir) else load_model(model_dir)
    warmed().add(model_dir)
    return model


# --------------------------------------------------------------------------- evidence


def retrieval_ready(dataset: str | None) -> bool:
    """Whether the retrieval caches exist, so a check never starts a minutes-long build."""
    needed = [f"{pool}-evidence-" for pool in serve.EVIDENCE_POOLS]
    if dataset in ("covid", "liar", "pubhealth", "isot"):
        needed.append(f"{dataset}-claims-")
    return all(any(RETRIEVAL_DIR.glob(f"{prefix}*.npy")) for prefix in needed)


@st.cache_resource(show_spinner=False)
def evidence_search() -> serve.EvidenceSearch:
    search = serve.EvidenceSearch()
    search.warm()
    return search


def find_evidence(claim: str, dataset: str) -> serve.Found | None:
    if not retrieval_ready(dataset):
        return None
    search = evidence_search()
    search.warm(dataset)
    return search.find(claim, dataset)


def score(model_dir: str, claim: str) -> tuple[serve.Prediction, tuple[str, ...] | None]:
    """One model's verdict on a claim, and the evidence sentences it read (if it reads any)."""
    clf = classifier(model_dir)
    found = find_evidence(claim, clf.info.name) if clf.uses_evidence else None
    evidence = tuple(found.sentence_texts()) if found else None
    return clf.predict(claim, list(evidence) if evidence else None), evidence


def check(model_dir: str, claim: str, use_llm: bool) -> dict:
    """Everything the Check view shows: the verdict, the evidence and the LLM's opinion."""
    clf = classifier(model_dir)
    found = find_evidence(claim, clf.info.name)
    evidence = tuple(found.sentence_texts()) if found and clf.uses_evidence else None
    pred = clf.predict(claim, list(evidence) if evidence else None)
    llm_dir = llm_adapters().get(clf.info.name) if use_llm and not clf.qlora else None
    llm = classifier(llm_dir).predict(claim) if llm_dir else None
    return {
        "claim": claim,
        "model": model_dir,
        "pred": pred,
        "evidence": evidence,
        "found": found,
        "llm_dir": llm_dir,
        "llm": llm,
        "use_llm": use_llm,
    }


@st.cache_data(max_entries=64, show_spinner=False)
def explanation(model_dir: str, claim: str, evidence: tuple[str, ...] | None) -> serve.Explanation:
    """The words behind one model's verdict: exact weights for a baseline, SHAP otherwise.
    Cached, so switching views or toggles never recomputes SHAP for the same claim."""
    clf = load_model(model_dir)
    if isinstance(clf, serve.Baseline):
        return serve.explain_linear(clf, claim)
    return serve.explain(clf, claim, list(evidence) if evidence else None)


def why_of(model_dir: str, claim: str, evidence: tuple[str, ...] | None) -> serve.Explanation:
    classifier(model_dir)
    why = explanation(model_dir, claim, None if is_baseline(model_dir) else evidence)
    warmed().add((model_dir, claim))
    return why


# --------------------------------------------------------------------------- verdicts


def share(p: float) -> str:
    """A probability as a percentage that never claims certainty: '>99%', '<1%'."""
    if p >= 0.995:
        return ">99%"
    if p < 0.005:
        return "<1%"
    return f"{p:.0%}"


def verdict_word(pred: serve.Prediction) -> str:
    if pred.true_prob >= VERDICT_AT:
        return "Likely true"
    return "Likely false" if pred.fake_prob > VERDICT_AT else MIXED


def verdict(pred: serve.Prediction) -> tuple[str, str, str]:
    word = verdict_word(pred)
    return (word, *VERDICTS[word])


def sure_of(pred: serve.Prediction, subject: str = "the claim") -> str:
    """How sure the model is, said in the verdict's own direction: 'with a >99% probability
    that the claim is true' for a true verdict, rather than '<1% probability ... false'.
    P(true) is the model's own, never 1 - P(false), which would count mixed labels as true."""
    if verdict_word(pred) == "Likely true":
        return f"with a **{share(pred.true_prob)}** probability that {subject} is true"
    return f"with a **{share(pred.fake_prob)}** probability that {subject} is false"


def verdict_sentence(pred: serve.Prediction, rungs: pd.DataFrame | None = None) -> str:
    """'**Likely false**, with a >99% probability that the claim is false.' For a dataset
    with more than two labels the most likely label is named too."""
    word, color, icon = verdict(pred)
    text = f":{color}[{icon} **{word}**], {sure_of(pred)}"
    if rungs is not None and len(rungs) > 2:
        lit = rungs[rungs["lit"]].iloc[0]
        text += f", most likely rated **{lit['name']}**"
    return text + "."


def rung_name(label: str) -> str:
    return RUNG_NAMES.get(label, label.replace("-", " ").replace("_", " ").capitalize())


def rating_color(label: str) -> str:
    """Theme colour of a verdict label: red false, orange mixed, green true, grey off-scale."""
    score_ = TRUTH_SCORE.get(label)
    if score_ is None:
        return "gray"
    return "red" if score_ <= 0.15 else "orange" if score_ < 0.75 else "green"


def ladder_rungs(
    labels: tuple[str, ...], probs: dict[str, float] | None = None, confident: bool = False
) -> pd.DataFrame:
    """The dataset's labels ordered from false to true; the most probable one is lit.

    The lit rung is outlined in its verdict colour; it is also tinted only when the verdict
    is confident, so a 21% top label never looks like a ruling.
    """
    rungs = pd.DataFrame({"label": list(labels)})
    rungs["score"] = rungs["label"].map(TRUTH_SCORE)
    rungs = rungs.sort_values("score", na_position="last", kind="stable").reset_index(drop=True)
    rungs["name"] = [rung_name(x) for x in rungs["label"]]
    rungs["order"] = range(len(rungs))
    rungs["prob"] = rungs["label"].map(probs) if probs else float("nan")
    rungs["pct"] = rungs["prob"].map(lambda p: "" if pd.isna(p) else share(p))
    rungs["lit"] = rungs["prob"] == rungs["prob"].max() if probs else False
    family = [rating_color(label) for label in rungs["label"]]
    lit = list(rungs["lit"])
    rungs["fill"] = [
        RUNG_TINT[f] if on and confident else PAPER if on else UNLIT_FILL
        for f, on in zip(family, lit, strict=True)
    ]
    rungs["stroke"] = [
        RUNG_FILL.get(label, OFF_SCALE_FILL) if on else UNLIT_FILL
        for label, on in zip(rungs["label"], lit, strict=True)
    ]
    rungs["ink"] = [
        (RUNG_INK[f] if confident else LIT_OUTLINE_INK) if on else UNLIT_INK
        for f, on in zip(family, lit, strict=True)
    ]
    return rungs


def ladder_chart(rungs: pd.DataFrame, height: int = 36, names: str = "all") -> alt.LayerChart:
    """A thin row of rungs from false to true, each reading 'name  probability'.

    ``names="lit"`` is for narrow ladders: with more than three labels a name does not fit
    inside a rung, so every rung shows its probability only (the lit one is outlined, and the
    sentence above the ladder names it); with three labels or fewer every rung is named.
    """
    every = names != "lit" or len(rungs) <= 3
    named = pd.Series(every, index=rungs.index)
    rungs = rungs.assign(
        text=[
            f"{name}  {pct}".strip() if show else pct
            for name, pct, show in zip(rungs["name"], rungs["pct"], named, strict=True)
        ],
        weight=["bold" if lit else "normal" for lit in rungs["lit"]],
        stroke_width=[2 if lit else 0 for lit in rungs["lit"]],
    )
    x = alt.X("order:O", axis=None, scale=alt.Scale(paddingInner=0.03, paddingOuter=0))
    base = alt.Chart(rungs).encode(x=x)
    rects = base.mark_rect(cornerRadius=2).encode(
        y=alt.value(1),
        y2=alt.value(height - 1),
        color=alt.Color("fill:N", scale=None),
        stroke=alt.Stroke("stroke:N", scale=None),
        strokeWidth=alt.StrokeWidth("stroke_width:Q", scale=None),
    )
    labels = [
        base.mark_text(fontSize=14, baseline="middle", fontWeight=weight)
        .encode(text="text:N", color=alt.Color("ink:N", scale=None), y=alt.value(height / 2))
        .transform_filter(f"datum.weight == '{weight}'")
        for weight in ("bold", "normal")
    ]
    return alt.layer(rects, *labels).properties(height=height).configure_view(strokeWidth=0)


def show_ladder(rungs: pd.DataFrame, names: str = "all") -> None:
    lit = rungs[rungs["lit"]]
    alt_text = (
        f"Rating ladder from false to true, {lit['name'].iloc[0]} lit at {lit['pct'].iloc[0]}."
        if len(lit)
        else "Rating ladder from false to true, waiting for a check."
    )
    st.altair_chart(ladder_chart(rungs, names=names), width="stretch", alt=alt_text)
    if len(rungs) > 3:  # a long ladder needs its direction; Fake / Real says it already
        with st.container(horizontal=True, horizontal_alignment="distribute"):
            st.caption(":material/arrow_back: false", width="content")
            st.caption("true :material/arrow_forward:", width="content")


def prediction_rungs(pred: serve.Prediction) -> pd.DataFrame:
    return ladder_rungs(tuple(pred.label_probs), pred.label_probs, verdict_word(pred) != MIXED)


# --------------------------------------------------------------------------- shared pieces


def init_state() -> None:
    """Session defaults. A link can open a view and check a claim at once (for the demo):
    ?view=compare, ?claim=Vitamin+D+cures+COVID-19 (checked on arrival), ?llm=1."""
    if "view" not in st.session_state:
        linked = st.query_params.get("view")
        st.session_state["view"] = linked if linked in VIEWS else "check"
        claim = st.query_params.get("claim", "").strip()
        if claim:
            st.session_state["claim"] = claim[:5000]
            st.session_state["auto_check"] = True
        st.session_state["use_llm"] = st.query_params.get("llm") == "1"
    first_claim, first_dataset = next(iter(EXAMPLE_CLAIMS.values()))
    st.session_state.setdefault("claim", first_claim)
    st.session_state.setdefault("dataset", first_dataset)
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("auto_check", False)
    st.session_state.setdefault("use_llm", False)


def use_example(text: str, dataset: str) -> None:
    st.session_state["claim"] = text
    for key in CLAIM_KEYS:
        st.session_state[key] = text
    st.session_state["dataset"] = dataset
    for key in ("dataset_check", "dataset_compare"):
        st.session_state.pop(key, None)  # the pickers re-read the shared dataset
    st.session_state["auto_check"] = True


def claim_box(key: str) -> str:
    """A claim text box per view that shares one value across views."""
    st.session_state.setdefault(key, st.session_state["claim"])
    claim = st.text_area(
        "Claim or post",
        key=key,
        height=118,
        max_chars=5000,
        placeholder="Paste a claim, a post or a headline…",
    )
    st.session_state["claim"] = claim
    return claim


def example_links(where: str) -> None:
    """The example claims as quiet links under the claim box: one click loads and checks."""
    with st.container(horizontal=True, vertical_alignment="center", gap="medium"):
        st.caption("Try", width="content")
        for i, (short, (text, dataset)) in enumerate(EXAMPLE_CLAIMS.items()):
            st.button(
                short,
                key=f"example_{where}_{i}",
                on_click=use_example,
                args=(text, dataset),
                type="tertiary",
            )


def dataset_picker(where: str, models: dict[str, Path]) -> str | None:
    """Which dataset's models read the claim. Shared across views through ``dataset``."""
    present = [d for d in DATASET_CHOICES if any(dataset_of(m) == d for m in models)]
    if not present:
        return None
    if st.session_state["dataset"] not in present:
        st.session_state["dataset"] = present[0]
    key = f"dataset_{where}"
    st.session_state.setdefault(key, st.session_state["dataset"])
    chosen = st.selectbox(
        "Trained on",
        present,
        format_func=DATASET_CHOICES.get,
        key=key,
        on_change=lambda: st.session_state.update(dataset=st.session_state[key]),
    )
    st.session_state["dataset"] = chosen
    return chosen


def style_picker(where: str, models: list[str]) -> list[str]:
    """The dataset's models for the chosen text style. The switch shows only when the dataset
    has models trained both on the text as written and on the style-stripped text (T59)."""
    present = [s for s in STYLES if any(style_of(m) == s for m in models)]
    if len(present) < 2:
        return models
    st.session_state.setdefault("text_style", "written")
    key = f"style_{where}"
    st.session_state.setdefault(key, st.session_state["text_style"])
    chosen = st.segmented_control(
        "Text",
        present,
        format_func=STYLES.get,
        required=True,
        key=key,
        on_change=lambda: st.session_state.update(text_style=st.session_state[key]),
    )
    st.session_state["text_style"] = chosen
    return [m for m in models if style_of(m) == chosen]


def default_model(choices: list[str]) -> str:
    """The best transformer of the dataset (choices are ordered up the ladder, best first)."""
    encoders = [m for m in choices if ladder_rank(m) == 1]
    return (encoders or choices)[0]


def llm_switch() -> None:
    """The Qwen3-8B switch. Its state lives in ``use_llm``, so it survives switching views
    (a widget's own key is dropped when its view is not shown)."""
    st.toggle(
        "Second opinion from Qwen3-8B",
        value=st.session_state["use_llm"],
        key="llm_switch",
        on_change=lambda: st.session_state.update(use_llm=st.session_state["llm_switch"]),
        disabled=not llm_adapters(),
    )


def waiting_panel(title: str, body: str, command: str | None = None) -> None:
    """Content that is not available here yet: what will appear and what fills it."""
    st.markdown(f"**{title}**")
    st.markdown(body)
    if command:
        st.code(command, language="bash", wrap_lines=True)


def no_models_panel() -> None:
    waiting_panel(
        "No trained models yet",
        "The verdict appears once there is a trained model under `outputs/`. The TF-IDF "
        "baseline trains in a minute on a CPU.",
        "python -m factcheck.train --config configs/classic/tfidf_lr_covid.yaml",
    )


def escape_markdown(text: str) -> str:
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|>~$:<])", r"\\\1", text)


def shorten(text: str, limit: int = 220) -> str:
    text = " ".join(str(text).replace('""', '"').split())  # PUBHEALTH quotes are CSV-doubled
    return text if len(text) <= limit else text[: limit - 1].rsplit(" ", 1)[0] + "…"


# --------------------------------------------------------------------------- why


def highlighted(why: serve.Explanation, top: int = 8) -> str:
    """The claim as markdown, its ``top`` strongest words tinted by their push: brick towards
    false, green towards true; the two strongest also bold. The rest stay plain."""
    words = why.words()
    ranked = sorted(
        (i for i, (w, v) in enumerate(words) if abs(v) >= 0.02 and re.search(r"\w", w)),
        key=lambda i: -abs(words[i][1]),
    )
    tinted, strongest = set(ranked[:top]), set(ranked[:2])
    out = []
    for i, (word, value) in enumerate(words):
        core = word.strip()
        lead, trail = word[: len(word) - len(word.lstrip())], word[len(word.rstrip()) :]
        text = escape_markdown(core)
        if i in tinted:
            text = f":{'red' if value > 0 else 'green'}-background[{text}]"
            if i in strongest:
                text = f"**{text}**"
        out.append(f"{lead}{text}{trail}")
    return "".join(out)


def top_pushes(why: serve.Explanation, top: int = 8) -> list[tuple[str, float]]:
    """The strongest pushes, strongest first: n-grams for a baseline (pairs kept whole),
    words for SHAP. Punctuation alone is not a reason."""
    pairs = why.features if why.features is not None else why.words()
    pairs = [(w.strip(), v) for w, v in pairs if re.search(r"\w", w) and abs(v) >= 0.005]
    return sorted(pairs, key=lambda wv: -abs(wv[1]))[:top]


def push_range(whys: list[serve.Explanation], top: int = 8) -> tuple[float, float]:
    """One scale for word bars that sit side by side, so equal pushes draw equal bars."""
    values = [v for why in whys for _, v in top_pushes(why, top)] or [0.0]
    return min(0.0, min(values)), max(0.0, max(values)) or 1.0


def push_bars(
    why: serve.Explanation, top: int = 8, scale: tuple[float, float] | None = None
) -> None:
    """The strongest pushes as aligned rows: word, a bar from zero, the value in its own
    column. Brick bars push towards false, green towards true."""
    pushes = top_pushes(why, top)
    if not pushes:
        st.caption("No single word moved this verdict.")
        return
    low, high = scale or push_range([why], top)
    span = (high - low) or 1.0
    zero = -low / span * 100
    ends = ("<span>&larr; true</span>" if low < 0 else "<span></span>") + (
        "<span>false &rarr;</span>" if high > 0 else "<span></span>"
    )
    rows = [f'<div class="row key"><span></span><span class="ends">{ends}</span></div>']
    for rank, (word, value) in enumerate(pushes):
        width = abs(value) / span * 100
        left = zero if value > 0 else zero - width
        rows.append(
            f'<div class="row{" strong" if rank < 2 else ""}">'
            f'<span class="word">{html.escape(word)}</span>'
            f'<span class="track"><span class="zero" style="left:{zero:.2f}%"></span>'
            f'<span class="bar {"false" if value > 0 else "true"}" '
            f'style="left:{left:.2f}%;width:{max(width, 0.6):.2f}%"></span></span>'
            f'<span class="value">{value:+.2f}</span></div>'
        )
    st.markdown(f'<div class="pushes">{"".join(rows)}</div>', unsafe_allow_html=True)


def why_column(
    model_dir: str,
    pred: serve.Prediction,
    why: serve.Explanation,
    scale: tuple[float, float] | None = None,
) -> None:
    """One model's reading of the claim: its verdict, the claim with its words tinted, and
    the strongest pushes."""
    st.markdown(f"#### {model_name(model_dir)}")
    st.markdown(verdict_sentence(pred))
    with st.container(border=True):
        st.markdown(highlighted(why))
    push_bars(why, scale=scale)


# --------------------------------------------------------------------------- skeletons


def bar(width: str, height: str = "1rem") -> str:
    return f'<div class="skel" style="width:{width};height:{height}"></div>'


def skeleton_reading(rows: int = 6) -> str:
    """One model's column: name, sentence, the claim exhibit and the word bars."""
    bars = "".join(
        bar(f"{w}%", "0.9rem") + bar(f"{b}%", "0.9rem") + bar("100%", "0.8rem")
        for w, b in zip(
            [70, 55, 80, 45, 65, 50][:rows], [60, 45, 35, 30, 20, 15][:rows], strict=True
        )
    )
    return (
        '<div class="skel-stack">'
        + bar("45%", "1.4rem")
        + bar("85%")
        + f'<div class="skel-frame skel-stack">{bar("95%")}{bar("60%")}</div>'
        + f'<div class="skel-bars">{bars}</div></div>'
    )


def skeleton_columns(count: int) -> str:
    return (
        f'<div class="skel-grid" style="grid-template-columns:repeat({count},1fr)">'
        + "".join(skeleton_reading() for _ in range(count))
        + "</div>"
    )


def skeleton_verdict() -> str:
    return (
        f'<div class="skel-frame skel-stack">{bar("28%", "2.4rem")}{bar("52%")}'
        f'{bar("100%", "2.25rem")}</div>'
    )


def skeleton_evidence() -> str:
    column = (
        '<div class="skel-stack">'
        + bar("55%", "1.1rem")
        + "".join(bar(f"{w}%") for w in (95, 80, 90, 70, 85))
        + "</div>"
    )
    return f'<div class="skel-grid" style="grid-template-columns:1fr 1fr">{column}{column}</div>'


def skeleton(html_: str) -> None:
    st.html(f'<div role="status" aria-busy="true" aria-label="Loading">{html_}</div>')


def skeleton_page() -> str:
    """The whole first screen, while the engine and the models load for the first time."""
    controls = (
        '<div style="display:grid;grid-template-columns:6.4fr 3.6fr;gap:1rem">'
        f'<div class="skel-stack">{bar("12%", "0.9rem")}{bar("100%", "7.4rem")}'
        f'{bar("55%", "0.9rem")}</div>'
        f'<div class="skel-stack">{bar("25%", "0.9rem")}{bar("100%", "2.6rem")}'
        f'{bar("20%", "0.9rem")}{bar("100%", "2.6rem")}{bar("100%", "2.6rem")}</div></div>'
    )
    return controls + '<div style="height:2rem"></div>' + skeleton_verdict()


@st.cache_resource
def warmed() -> set:
    """What this app process has already loaded or computed, so a skeleton shows only while
    something is slow (a model's first load, a claim's first explanation)."""
    return set()


# --------------------------------------------------------------------------- check panels


def verdict_block(result: dict | None, rungs: pd.DataFrame) -> None:
    """The ruling: a serif verdict line, one sentence on how sure the model is, the ladder."""
    with st.container(border=True):
        if result:
            pred = result["pred"]
            word, color, icon = verdict(pred)
            st.header(f":{color}[{icon} {word}]", anchor=False)
            text = f"The model gives {sure_of(pred, 'this claim')[5:]}"
            if len(rungs) > 2:
                lit = rungs[rungs["lit"]].iloc[0]
                text += f", and rates it most likely **{lit['name']}**"
            st.markdown(text + ".")
        else:
            st.header(":gray[No claim checked yet]", anchor=False)
            st.markdown("Pick an example under the box, or paste a claim and press **Check**.")
        show_ladder(rungs)


def llm_block(result: dict, encoder_pred: serve.Prediction) -> None:
    llm = result["llm"]
    agree = verdict_word(encoder_pred) == verdict_word(llm)
    with st.container(border=True):
        st.markdown("#### Second opinion from Qwen3-8B")
        st.markdown(verdict_sentence(llm, prediction_rungs(llm)))
        if agree:
            st.markdown("It agrees with the model above.")
        else:
            st.markdown(
                ":material/call_split: **It disagrees with the model above**, so this claim "
                "is worth a human check."
            )


def evidence_section(result: dict, clf) -> None:  # noqa: ANN001  (Classifier or Baseline)
    st.subheader("Evidence", anchor=False)
    found = result["found"]
    if found is None:
        waiting_panel(
            "The evidence index is not built on this machine",
            "Similar fact-checked claims and evidence sentences appear here once it is built.",
            "python -m factcheck.retrieval build",
        )
        return
    dataset = DATASETS.get(clf.info.name, clf.info.name)
    similar_col, sentence_col = st.columns(2, gap="large")
    with similar_col:
        st.markdown(f"**Similar claims already fact-checked in {dataset}**")
        for hit in found.similar or []:
            st.markdown(
                f":{rating_color(hit.label_name)}[**{rung_name(hit.label_name)}**] "
                f"{escape_markdown(shorten(hit.text))}"
            )
        if not found.similar:
            st.markdown(":gray[None found.]")
    with sentence_col:
        st.markdown("**Closest sentences from fact-check articles**")
        for sentence in found.sentences:
            source = SOURCES.get(sentence.source, sentence.source)
            st.markdown(
                f"{escape_markdown(shorten(sentence.text, 300))}  \n"
                f'<span class="source">From {html.escape(source)}</span>',
                unsafe_allow_html=True,
            )


# --------------------------------------------------------------------------- views


def check_view(models: dict[str, Path]) -> None:
    col_claim, col_side = st.columns([6.4, 3.6], gap="medium")
    with col_claim:
        claim = claim_box("claim_check")
        example_links("check")
    with col_side:
        dataset = dataset_picker("check", models)
        choices = style_picker("check", [m for m in models if dataset_of(m) == dataset])
        model_dir = st.selectbox(
            "Model",
            choices,
            index=choices.index(default_model(choices)) if choices else None,
            format_func=model_name,
            placeholder="No trained models yet",
            disabled=not choices,
            key=f"check_model_{dataset}_{st.session_state.get('text_style', 'written')}",
        )
        llm_switch()
        pressed = st.button(
            "Check",
            type="primary",
            icon=":material/fact_check:",
            width="stretch",
            disabled=not models,
        )

    use_llm = st.session_state["use_llm"]
    auto = st.session_state.pop("auto_check", False)
    if model_dir and claim.strip() and (pressed or auto):
        slot = st.empty()
        with slot.container():
            st.space("small")
            skeleton(skeleton_verdict() + skeleton_columns(2) + skeleton_evidence())
        st.session_state["result"] = check(model_dir, claim, use_llm)
        st.session_state["result"]["readings"] = readings_of(model_dir, claim, models)
        slot.empty()
    result = st.session_state["result"]
    if result and (
        result["claim"] != claim or result["model"] != model_dir or result["use_llm"] != use_llm
    ):
        result = None  # the claim, the model or the LLM switch changed since the last check

    st.space("small")
    if result:
        rungs = prediction_rungs(result["pred"])
    elif model_dir:
        rungs = ladder_rungs(tuple(get_dataset(dataset_of(model_dir)).labels))
    else:
        rungs = ladder_rungs(("fake", "mixture", "real"))
        rungs["name"] = ["Likely false", "Mixed", "Likely true"]
    verdict_block(result, rungs)
    if not result:
        if not models:
            no_models_panel()
        return
    if result["llm"] is not None:
        llm_block(result, result["pred"])

    st.space("medium")
    st.subheader("Why", anchor=False)
    readings = result.get("readings") or readings_of(model_dir, claim, models)
    whys = [why for _, _, why in readings]
    for column, (m, pred, why) in zip(
        st.columns(len(readings), gap="large"), readings, strict=True
    ):
        with column:
            why_column(m, pred, why, push_range(whys))

    st.space("medium")
    evidence_section(result, classifier(model_dir))


def readings_of(model_dir: str, claim: str, models: dict[str, Path]) -> list:
    """(model, prediction, explanation) for the checked model and its counterpart, bottom of
    the ladder first: everything the Why section draws, computed while the skeleton shows."""
    readings = [(model_dir, *score(model_dir, claim))]
    pair = counterpart(model_dir, models)
    if pair:
        readings.append((pair, *score(pair, claim)))
    readings.sort(key=lambda r: ladder_rank(r[0]))  # the ladder reads left to right
    return [(m, pred, why_of(m, claim, evidence)) for m, pred, evidence in readings]


def compare_defaults(choices: list[str]) -> list[str]:
    """Up the ladder: the TF-IDF baseline, the fast DistilBERT and the best transformer."""
    baseline = [m for m in choices if is_baseline(m)]
    encoders = [m for m in choices if ladder_rank(m) == 1]
    fast = [m for m in encoders[1:] if "distilbert" in run_config(m)["model"]["name"]]
    return [*baseline[:1], *fast[:1], *encoders[:1]] or choices[:1]


def compare_view(models: dict[str, Path]) -> None:
    col_claim, col_side = st.columns([6.4, 3.6], gap="medium")
    with col_claim:
        claim = claim_box("claim_compare")
        example_links("compare")
    st.session_state.pop("auto_check", False)  # every comparison below runs on each rerun
    if not models:
        no_models_panel()
        return
    with col_side:
        dataset = dataset_picker("compare", models)
        choices = style_picker("compare", [m for m in models if dataset_of(m) == dataset])
        chosen = st.multiselect(
            "Models",
            choices,
            default=compare_defaults(choices),
            format_func=model_name,
            max_selections=4,
            key=f"compare_models_{dataset}_{st.session_state.get('text_style', 'written')}",
        )
    if not (claim.strip() and chosen):
        st.markdown("Choose up to four models to read the claim side by side.")
        return

    chosen = sorted(chosen, key=lambda m: (ladder_rank(m), choices.index(m)))
    slot = st.empty()
    if any((m, claim) not in warmed() for m in chosen):
        with slot.container():
            skeleton(skeleton_columns(len(chosen)))
    results = {m: score(m, claim) for m in chosen}
    whys = {m: why_of(m, claim, evidence) for m, (_, evidence) in results.items()}
    slot.empty()
    scale = push_range(list(whys.values()), top=5)
    st.space("small")
    for column, (model_dir, (pred, _)) in zip(
        st.columns(len(chosen), gap="large"), results.items(), strict=True
    ):
        with column:
            st.markdown(f"#### {model_name(model_dir)}")
            st.markdown(verdict_sentence(pred, prediction_rungs(pred)))
            show_ladder(prediction_rungs(pred), names="lit")
            with st.container(border=True):
                st.markdown(highlighted(whys[model_dir]))
            push_bars(whys[model_dir], top=5, scale=scale)
    verdicts = {verdict_word(p) for p, _ in results.values()}
    st.space("small")
    if len(results) > 1 and len(verdicts) > 1:
        st.markdown(
            ":material/call_split: **The models disagree on this claim.** Each one leans on "
            "different words."
        )
    elif len(results) > 1:
        st.markdown(
            f"All {len(results)} models agree that the claim is **{verdicts.pop().lower()}**."
        )


@st.cache_data(show_spinner=False)
def fake_articles(limit: int = 50) -> list[str]:
    """Real fake articles from the cleaned ISOT test split, short enough to read on screen.

    Every fake ISOT article is partisan; "Government News" ones come first because they are
    the least inflammatory to open a recorded demo on.
    """
    try:
        test = load_splits("isot", "clean")["test"]
    except FileNotFoundError:
        return []
    fake = test[(test["label_name"] == "fake") & test["text"].str.len().between(400, 900)]
    fake = fake.sort_values("subject", key=lambda s: s != "Government News", kind="stable")
    return fake["text"].head(limit).astype(str).tolist()


def next_article(count: int) -> None:
    st.session_state["article_index"] = (st.session_state.get("article_index", 0) + 1) % count


@st.cache_data(show_spinner=False)
def reuters_share() -> dict[str, float] | None:
    """Share of real ISOT training articles that contain '(Reuters)', raw vs cleaned."""
    try:
        shares = {}
        for variant in ("raw", "clean"):
            train = load_splits("isot", variant)["train"]
            real = train[train["label_name"] == "real"]["text"]
            shares[variant] = float(real.str.contains("(Reuters)", regex=False).mean())
        return shares
    except FileNotFoundError:
        return None


def shortcut_reading(model_dir: str, text: str) -> None:
    """One ISOT model's reading of the article; a baseline also shows the words it leaned on."""
    pred, _ = score(model_dir, text)
    warmed().add((model_dir, text))
    st.markdown(f"#### {model_label(model_dir)}")
    st.markdown(verdict_sentence(pred))
    show_ladder(prediction_rungs(pred))
    if is_baseline(model_dir):
        push_bars(why_of(model_dir, text, None), top=4)


def shortcut_view(models: dict[str, Path]) -> None:
    shares = reuters_share()
    share_text = (
        f" In the raw training text **{shares['raw']:.1%}** of real articles contain "
        f"“(Reuters)”, and **{shares['clean']:.1%}** after cleaning."
        if shares
        else ""
    )
    st.markdown(
        "Real ISOT articles almost all open with a Reuters dateline, so a model trained on raw "
        "text can learn the dateline instead of the content." + share_text + " Add the dateline "
        "to a fake article and watch which models flip."
    )

    articles = fake_articles()
    if not articles:
        st.markdown(
            "Prepare ISOT first: `python scripts/download_data.py isot && python scripts/prepare_data.py isot`."
        )
        return
    index = st.session_state.get("article_index", 0) % len(articles)
    article = articles[index]
    with st.container(horizontal=True, vertical_alignment="center", gap="medium"):
        add_dateline = st.toggle(f"Add the “{DATELINE.strip()}” dateline", key="shortcut_dateline")
        st.button(
            "Another article",
            icon=":material/shuffle:",
            on_click=next_article,
            args=(len(articles),),
            type="tertiary",
        )
    text = DATELINE + article if add_dateline else article
    with st.container(border=True):
        st.markdown(
            (f"**:blue-background[{escape_markdown(DATELINE.strip())}]** " if add_dateline else "")
            + escape_markdown(article)
        )

    picks: dict[tuple[str, bool], str] = {}
    for m in models:  # the best model per variant and kind (models are ordered best first)
        if dataset_of(m) == "isot" and not serve.is_qlora(m) and style_of(m) == "written":
            picks.setdefault((variant_of(m), is_baseline(m)), m)
    commands = {
        ("raw", False): "python -m factcheck.train --config configs/encoder/roberta_isot_raw.yaml",
        (
            "clean",
            False,
        ): "python -m factcheck.train --config configs/encoder/roberta_isot_clean.yaml",
        ("raw", True): "python -m factcheck.train --config configs/classic/tfidf_lr_isot_raw.yaml",
        (
            "clean",
            True,
        ): "python -m factcheck.train --config configs/classic/tfidf_lr_isot_clean.yaml",
    }
    titles = {"raw": "Trained on raw text", "clean": "Trained on cleaned text"}
    st.space("small")
    slot = st.empty()
    if any((m, text) not in warmed() for m in picks.values()):
        with slot.container():
            skeleton(skeleton_columns(2))
        for m in picks.values():  # load and score while the skeleton shows
            score(m, text)
            if is_baseline(m):
                why_of(m, text, None)
    slot.empty()
    for column, variant in zip(st.columns(2, gap="large"), ("raw", "clean"), strict=True):
        with column:
            st.subheader(titles[variant], anchor=False)
            for baseline in (True, False):  # bottom of the ladder first
                model_dir = picks.get((variant, baseline))
                if model_dir is None:
                    waiting_panel(
                        f"No {'TF-IDF baseline' if baseline else 'transformer'} for this text yet",
                        f"Needs an ISOT model trained on the {variant} text.",
                        commands[(variant, baseline)],
                    )
                else:
                    shortcut_reading(model_dir, text)
                st.space("small")


SPEC_STATUS = {
    "PASS": ":material/check: Pass",
    "FAIL": ":red[:material/close: **Fail**]",
    "no runs yet": ":gray[:material/hourglass_empty: No runs yet]",
    "check by hand": ":gray[:material/back_hand: Check by hand]",
}
RUN_COLUMNS = {
    "run_name": st.column_config.TextColumn("Run", pinned=True),
    "dataset": "Dataset",
    "variant": "Variant",
    "model_name": "Model",
    "method": "Method",
    "seed": st.column_config.NumberColumn("Seed", format="plain"),
    "macro_f1": st.column_config.NumberColumn("Macro-F1", format="%.3f"),
    "accuracy": st.column_config.NumberColumn("Accuracy", format="%.3f"),
    "weighted_f1": st.column_config.NumberColumn("Weighted F1", format="%.3f"),
    "roc_auc_fake": st.column_config.NumberColumn("ROC-AUC (fake)", format="%.3f"),
    "ece": st.column_config.NumberColumn("ECE", format="%.3f"),
    "trainable_params": st.column_config.NumberColumn("Trainable params", format="compact"),
    "latency_ms": st.column_config.TextColumn("Latency"),
}
METHOD_NAMES = {
    "tfidf_lr": "TF-IDF + LR",
    "tfidf_nb": "TF-IDF + NB",
    "tfidf_svm": "TF-IDF + SVM",
    "glove_svm": "GloVe + SVM",
    "knn_vote": "bge-small kNN",
    "zeroshot": "zero-shot",
    "knn": "kNN prompt",
    "rag": "RAG",
    "qlora": "QLoRA",
}


def rung_of(model_name_: str, method) -> int:  # noqa: ANN001  (method may be NaN)
    if model_name_ == "tfidf":
        return 0
    if method in ("glove_svm", "knn_vote"):
        return 1
    if str(model_name_).startswith("bilstm"):
        return 2
    if str(model_name_).startswith("Qwen/"):
        return 4
    return 3


def run_label(model_name_: str, method, run_name: str) -> str:  # noqa: ANN001
    """A short name for the best run on a rung: 'TF-IDF + LR', 'DeBERTa-v3-large LoRA'..."""
    method = method if isinstance(method, str) else ""
    if model_name_ in ("tfidf", "glove-300d") or method == "knn_vote":
        return METHOD_NAMES.get(method, method)
    if str(model_name_).startswith("bilstm"):
        return "Bi-LSTM + GloVe"
    if str(model_name_).startswith("Qwen/"):
        return f"Qwen3-8B {METHOD_NAMES.get(method, method)}"
    label = MODEL_NAMES.get(model_name_, str(model_name_).split("/")[-1])
    if method == "lora":
        label += " LoRA"
    if "evidence" in run_name:
        label += " with evidence"
    return label


def ladder_of_results(test: pd.DataFrame) -> pd.DataFrame:
    """Per dataset and rung, the run with the best MEAN test macro-F1 over its seeds (never
    the luckiest seed). ISOT uses the cleaned text: on raw text every rung scores ~1.0 by
    reading the source tags. Datasets with a single rung are left out."""
    keep = (test["dataset"] != "isot") & (test["variant"].fillna("raw") == "raw")
    keep |= (test["dataset"] == "isot") & (test["variant"] == "clean")
    test = test[keep]
    per_run = test.groupby(["dataset", "model_name", "method", "run_name"], dropna=False).agg(
        macro_f1=("macro_f1", "mean"), seeds=("seed", "nunique")
    )
    per_run = per_run.reset_index()
    per_run["rung"] = [
        rung_of(m, k) for m, k in zip(per_run["model_name"], per_run["method"], strict=True)
    ]
    best = per_run.loc[per_run.groupby(["dataset", "rung"])["macro_f1"].idxmax()].copy()
    best["step"] = [RUNGS[r] for r in best["rung"]]
    best["label"] = [
        f"{f1:.3f}  {run_label(m, k, r)}"
        for f1, m, k, r in zip(
            best["macro_f1"], best["model_name"], best["method"], best["run_name"], strict=True
        )
    ]
    rungs = best.groupby("dataset")["rung"].transform("nunique")
    return best[rungs > 1]


def ladder_results_chart(rows: pd.DataFrame) -> alt.LayerChart:
    """One dataset's ladder: a bar per rung, bottom rung on top, labelled with the score and
    the best run's name, so nothing needs a hover."""
    y = alt.Y(
        "step:N",
        sort=RUNGS,
        title=None,
        scale=alt.Scale(domain=[s for s in RUNGS if s in set(rows["step"])]),
        axis=alt.Axis(labelFontSize=14, labelLimit=200),
    )
    x = alt.X(
        "macro_f1:Q",
        title="Test macro-F1, mean over seeds",
        scale=alt.Scale(domain=[0, 1.55]),
        axis=alt.Axis(
            values=[0, 0.25, 0.5, 0.75, 1.0],
            format=".2f",
            titleFontWeight="normal",
            titleFontSize=13,
        ),
    )
    base = alt.Chart(rows).encode(y=y, x=x)
    bars = base.mark_bar(cornerRadius=2, height=18, color="#3a5a80")
    text = base.mark_text(align="left", dx=6, fontSize=13, baseline="middle").encode(text="label:N")
    return alt.layer(bars, text).properties(height=40 * rows["step"].nunique() + 20)


def specs_scorecard(specs: pd.DataFrame) -> None:
    passing = int((specs["status"] == "PASS").sum())
    st.subheader(f"Specifications, {passing} of {len(specs)} passing", anchor=False)
    widths = [0.6, 4.5, 3, 1.2, 1.7]
    header = st.columns(widths)
    for column, name in zip(
        header, ("ID", "Specification", "Target", "Best", "Status"), strict=True
    ):
        column.caption(name)
    for row in specs.itertuples():
        cells = st.columns(widths, vertical_alignment="center")
        cells[0].markdown(f"**{row.id}**")
        cells[1].markdown(row.spec)
        cells[2].markdown(row.target.replace(">=", "≥").replace("<=", "≤"))
        cells[3].markdown(str(row.value) if row.value != "" else "–")
        cells[4].markdown(SPEC_STATUS.get(row.status, row.status))


def results_view() -> None:
    runs = load_runs()
    test = runs[runs["split"] == "test"] if "split" in runs else pd.DataFrame()
    if test.empty:
        waiting_panel(
            "No logged runs yet",
            "Every training run appends its scores to `results/runs.csv`, and they appear here.",
            "python -m factcheck.train --config configs/classic/tfidf_lr_covid.yaml",
        )
        return
    debug = test["subset"].notna() if "subset" in test else pd.Series(False, index=test.index)

    st.subheader("The ladder of models, by dataset", anchor=False)
    st.markdown(
        "The best run on each step, from word counts at the top to the 8B LLM at the bottom. "
        "Where TF-IDF is already close, the bigger models add little."
    )
    ladder = ladder_of_results(test[~debug])
    present = [d for d in DATASETS if d in set(ladder["dataset"])]
    for start in range(0, len(present), 2):
        for column, dataset in zip(
            st.columns(2, gap="large"), present[start : start + 2], strict=False
        ):
            with column:
                name = DATASETS[dataset] + (" (cleaned text)" if dataset == "isot" else "")
                st.markdown(f"#### {name}")
                st.altair_chart(
                    ladder_results_chart(ladder[ladder["dataset"] == dataset]),
                    width="stretch",
                    alt=f"Best test macro-F1 per rung of the ladder on {name}.",
                )

    st.space("medium")
    specs_scorecard(check_specs(runs))

    st.space("medium")
    st.subheader("Every run on the test split", anchor=False)
    if debug.any() and not st.toggle(
        f"Include {int(debug.sum())} debug runs on data subsets", key="show_debug"
    ):
        test = test[~debug]
    table = test[[c for c in RUN_COLUMNS if c in test]].dropna(axis="columns", how="all")
    for column in ("variant", "method"):
        if column in table:
            table[column] = table[column].fillna("–")
    if "latency_ms" in table:
        table["latency_ms"] = table["latency_ms"].map(
            lambda v: "–" if pd.isna(v) else f"{v:.1f} ms"
        )
    st.dataframe(
        table,
        column_config=RUN_COLUMNS,
        hide_index=True,
        alt="Every logged test-split run with its scores.",
    )


# --------------------------------------------------------------------------- page


def main() -> None:
    st.set_page_config(
        page_title="Fact-Check Lab",
        page_icon=":material/fact_check:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.html(STYLE)
    global serve
    boot = st.empty()
    if "factcheck.serve" not in sys.modules:  # the first run in this process
        with boot.container():
            st.title("Fact-Check Lab", anchor=False)
            st.divider()
            skeleton(skeleton_page())
    serve = importlib.import_module("factcheck.serve")
    init_state()
    models = find_models(include_llm=st.session_state["use_llm"])
    boot.empty()
    has_isot = any(dataset_of(m) == "isot" for m in models)
    views = [v for v in VIEWS if v != "shortcut" or has_isot]
    if st.session_state["view"] not in views:
        st.session_state["view"] = "check"

    with st.container(horizontal=True, vertical_alignment="center", gap="large"):
        st.title("Fact-Check Lab", anchor=False)
        view = st.segmented_control(
            "View",
            views,
            format_func=VIEWS.get,
            required=True,
            key="view",
            label_visibility="collapsed",
        )
    st.query_params["view"] = view  # ?view=results etc. opens a view directly
    st.divider()
    if view == "compare":
        compare_view(models)
    elif view == "shortcut":
        shortcut_view(models)
    elif view == "results":
        results_view()
    else:
        check_view(models)


main()
