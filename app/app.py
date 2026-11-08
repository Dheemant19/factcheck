"""Fact-Check Lab: paste a claim, get an explained verdict (T11, T22, T35, T58).

    streamlit run app/app.py

Check a claim: the verdict on the dataset's rating ladder from any trained model, the words
behind it twice over (SHAP for a transformer, exact weights for the TF-IDF + logistic
regression baseline, on the same claim), the evidence (similar already fact-checked claims with
their verdicts, and the closest evidence sentences from PUBHEALTH's fact-check articles and
Climate-FEVER), an optional second opinion from Qwen3-8B with its QLoRA adapter, and the
response time against specs S6 (encoder) and S7 (retrieval + LLM verdict). Compare models up
the ladder side by side, run the shortcut demo (a Reuters dateline flips the raw-trained ISOT
models), and read the ladder of results and the specs board from results/runs.csv.

Models are found under outputs/<run>/seed<N>/model; the engine is ``factcheck.serve``. The
theme lives in app/.streamlit/config.toml; the app uses native Streamlit only.
"""

from __future__ import annotations

import re
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import torch

from factcheck import serve
from factcheck.data.prepare import load_splits
from factcheck.evaluate import check_specs
from factcheck.paths import OUTPUT_DIR
from factcheck.results import load_runs
from factcheck.retrieval import RETRIEVAL_DIR

EXAMPLE_CLAIMS = {  # short link text -> the claim it loads
    "Hot water kills the virus": "Drinking hot water every 15 minutes kills the coronavirus.",
    "CDC on masks": "The CDC recommends that people wear masks in public settings.",
    "Unemployment has doubled": "Says the unemployment rate has doubled since the president took office.",
    "Vitamin D cure": "Vitamin D supplements cure COVID-19 in 24 hours.",
}
SPEC_S6_SECONDS = 0.050  # encoder latency per claim
SPEC_S7_SECONDS = 5.0  # full response: retrieval + LLM verdict
SPEC_S8_GB = 8.0  # everything in the app on the laptop GPU
DATELINE = "WASHINGTON (Reuters) - "

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
# Rung colours from the theme's verdict palette, dark enough for white text (WCAG AA).
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
PUSH_FALSE, PUSH_TRUE = "#b8332a", "#0b6e4f"

# (upper bound on P(false), verdict, theme colour, icon)
MIXED = "Mixed / uncertain"
VERDICTS = [
    (0.4, "Likely true", "green", ":material/check_circle:"),
    (0.6, MIXED, "orange", ":material/help:"),
    (1.0, "Likely false", "red", ":material/cancel:"),
]


# --------------------------------------------------------------------------- models


@st.cache_data(ttl=60)
def logged_test_f1() -> dict[tuple[str, int], float]:
    """Test macro-F1 of every logged (run, seed), read from results/runs.csv."""
    runs = load_runs()
    test = runs[(runs["split"] == "test") & runs["subset"].isna()]
    return {(r.run_name, int(r.seed)): float(r.macro_f1) for r in test.itertuples()}


@st.cache_data(max_entries=128)
def run_config(model_dir: str) -> dict:
    return serve.run_config(model_dir)


def test_f1(model_dir: str) -> float | None:
    run = run_config(model_dir)["run"]
    return logged_test_f1().get((run["name"], int(run["seed"])))


def find_models(include_llm: bool = False) -> dict[str, Path]:
    """Every saved model the app can run, by dataset, best logged test macro-F1 first.

    QLoRA adapters (Qwen3-8B, about 6 GB of GPU memory) are listed only when the LLM is on.
    """
    found = [
        p
        for p in OUTPUT_DIR.glob("*/seed*/model")
        if serve.is_servable(p) and (include_llm or not serve.is_qlora(p))
    ]
    order = list(DATASETS)

    def key(p: Path) -> tuple:
        cfg = run_config(str(p))
        dataset = cfg["data"]["dataset"]
        return (order.index(dataset) if dataset in order else 99, -(test_f1(str(p)) or 0.0))

    return {str(p): p for p in sorted(found, key=key)}


def llm_adapters() -> dict[str, str]:
    """Dataset -> the QLoRA adapter folder for it (the first seed found)."""
    adapters = {}
    for p in sorted(OUTPUT_DIR.glob("qwen3_8b_qlora_*/seed*/model")):
        if serve.is_servable(p):
            adapters.setdefault(run_config(str(p))["data"]["dataset"], str(p))
    return adapters


def dataset_of(model_dir: str) -> str:
    return run_config(model_dir)["data"]["dataset"]


def variant_of(model_dir: str) -> str:
    return run_config(model_dir)["data"].get("variant", "raw")


def is_baseline(model_dir: str) -> bool:
    return run_config(model_dir)["model"].get("family") == "classic"


def model_label(model_dir: str) -> str:
    name = run_config(model_dir)["model"]["name"]
    return MODEL_NAMES.get(name, name.split("/")[-1])


def plate(model_dir: str) -> str:
    """One fixed ID plate per model, used everywhere a model is named."""
    cfg = run_config(model_dir)
    data = cfg["data"]
    parts = [DATASETS.get(data["dataset"], data["dataset"]), model_label(model_dir)]
    if (cfg.get("features") or {}).get("use_evidence"):
        parts.append("+ evidence")
    if data.get("variant", "raw") != "raw":
        parts.append(data["variant"])
    if not is_baseline(model_dir):  # a baseline is deterministic: one run is the result
        parts.append(f"seed {cfg['run']['seed']}")
    f1 = test_f1(model_dir)
    if f1 is not None:
        parts.append(f"test F1 {f1:.3f}")
    return " · ".join(parts)


def counterpart(model_dir: str, models: dict[str, Path]) -> str | None:
    """The model to read the same claim next to this one: the TF-IDF baseline of the same
    dataset for a transformer, the best transformer of that dataset for the baseline."""
    dataset, variant = dataset_of(model_dir), variant_of(model_dir)
    same = [m for m in models if m != model_dir and dataset_of(m) == dataset]
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
    """The loaded model; the first use shows a spinner, and loading is never timed."""
    if serve.is_qlora(model_dir):
        with st.spinner(
            "Loading Qwen3-8B in 4-bit with its adapter (first use only, about a minute)…"
        ):
            return load_llm(model_dir)
    with st.spinner("Loading the model (first check only)…"):
        return load_model(model_dir)


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
    with st.spinner("Loading the retrieval index (first check only)…"):
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
    """Everything the Check view shows, timed: retrieval, the model and the LLM."""
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
    classifier(model_dir)  # load outside the cached function, with its spinner
    if is_baseline(model_dir):
        return explanation(model_dir, claim, None)
    with st.spinner(f"Finding the words behind {model_label(model_dir)}'s verdict (SHAP)…"):
        return explanation(model_dir, claim, evidence)


# --------------------------------------------------------------------------- rating ladder


def share(p: float) -> str:
    """A probability as a percentage that never claims certainty: '>99%', '<1%'."""
    if p >= 0.995:
        return ">99%"
    if p < 0.005:
        return "<1%"
    return f"{p:.0%}"


def verdict(fake_prob: float) -> tuple[str, str, str]:
    return next((word, color, icon) for bound, word, color, icon in VERDICTS if fake_prob <= bound)


def verdict_text(fake_prob: float) -> str:
    word, color, icon = verdict(fake_prob)
    return f":{color}[{icon} {word}]"


def rung_name(label: str) -> str:
    return RUNG_NAMES.get(label, label.replace("-", " ").replace("_", " ").capitalize())


def rating_color(label: str) -> str:
    """Theme colour of a verdict label: red false, orange mixed, green true, grey off-scale."""
    score_ = TRUTH_SCORE.get(label)
    if score_ is None:
        return "gray"
    return "red" if score_ <= 0.15 else "orange" if score_ < 0.75 else "green"


def ladder_rungs(
    labels: tuple[str, ...], probs: dict[str, float] | None = None, fake_prob: float | None = None
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
    rungs["pct"] = rungs["prob"].map(lambda p: "–" if pd.isna(p) else share(p))
    rungs["lit"] = rungs["prob"] == rungs["prob"].max() if probs else False
    confident = fake_prob is not None and verdict(fake_prob)[0] != MIXED
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

    ``names="lit"`` names only the lit rung (the others show their probability), for narrow
    ladders with many labels; with three labels or fewer every rung is named anyway.
    """
    every = names != "lit" or len(rungs) <= 3
    named = pd.Series(True, index=rungs.index) if every else rungs["lit"]
    rungs = rungs.assign(
        text=[
            f"{name}  {pct}" if show else pct
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


def show_ladder(rungs: pd.DataFrame, height: int = 36, names: str = "all") -> None:
    lit = rungs[rungs["lit"]]
    alt_text = (
        f"Rating ladder from false to true; {lit['name'].iloc[0]} is lit at {lit['pct'].iloc[0]}."
        if len(lit)
        else "Rating ladder from false to true, waiting for a check."
    )
    st.altair_chart(ladder_chart(rungs, height, names), width="stretch", alt=alt_text)
    with st.container(horizontal=True, horizontal_alignment="distribute"):
        st.caption(":material/arrow_back: false", width="content")
        st.caption("true :material/arrow_forward:", width="content")


def prediction_rungs(pred: serve.Prediction) -> pd.DataFrame:
    return ladder_rungs(tuple(pred.label_probs), pred.label_probs, pred.fake_prob)


def duration(seconds: float) -> str:
    return f"{seconds * 1000:.0f} ms" if seconds < 1 else f"{seconds:.2f} s"


def reading_line(pred: serve.Prediction, rungs: pd.DataFrame) -> str:
    """'91% probability false · most likely label: Pants on fire (48%)'. With two labels the
    second half would only repeat the first, so it is left out."""
    line = f"**{share(pred.fake_prob)}** probability false"
    if len(rungs) <= 2:
        return line
    lit = rungs[rungs["lit"]].iloc[0]
    return f"{line} · most likely label: **{lit['name']}** ({lit['pct']})"


def timing_line(result: dict) -> str:
    """Spec S6 (the model alone) and S7 (retrieval plus the LLM verdict, and the model)."""
    found, pred, llm = result["found"], result["pred"], result["llm"]
    within = pred.seconds <= SPEC_S6_SECONDS
    model = (
        f"Model {duration(pred.seconds)}, {'within' if within else 'over'} the "
        f"{SPEC_S6_SECONDS * 1000:.0f} ms target (S6)."
    )
    parts = [("retrieval", found.seconds if found else None), ("model", pred.seconds)]
    parts.append(("Qwen3-8B", llm.seconds if llm else None))
    timed = [(name, s) for name, s in parts if s is not None]
    total = sum(s for _, s in timed)
    breakdown = " + ".join(f"{name} {duration(s)}" for name, s in timed)
    if llm is None:
        return (
            f"{model} Full response {duration(total)} ({breakdown}); spec S7 times it with the "
            "Qwen3-8B second opinion on."
        )
    within = total <= SPEC_S7_SECONDS
    return (
        f"{model} Full response {duration(total)} ({breakdown}), "
        f"{'within' if within else 'over'} the {SPEC_S7_SECONDS:.0f} s target (S7)."
    )


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
    st.session_state.setdefault("claim", next(iter(EXAMPLE_CLAIMS.values())))
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("auto_check", False)
    st.session_state.setdefault("use_llm", False)


def use_example(text: str) -> None:
    st.session_state["claim"] = text
    for key in CLAIM_KEYS:
        st.session_state[key] = text
    st.session_state["auto_check"] = True


def claim_box(key: str, label: str = "Claim or post") -> str:
    """A claim text box per view that shares one value across views."""
    st.session_state.setdefault(key, st.session_state["claim"])
    claim = st.text_area(
        label,
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
        st.caption("Examples", width="content")
        for i, (short, text) in enumerate(EXAMPLE_CLAIMS.items()):
            st.button(
                short,
                key=f"example_{where}_{i}",
                on_click=use_example,
                args=(text,),
                type="tertiary",
                help=text,
            )


def llm_switch() -> None:
    """The Qwen3-8B switch. Its state lives in ``use_llm``, so it survives switching views
    (a widget's own key is dropped when its view is not shown)."""
    adapters = llm_adapters()
    st.toggle(
        "Qwen3-8B second opinion",
        value=st.session_state["use_llm"],
        key="llm_switch",
        on_change=lambda: st.session_state.update(use_llm=st.session_state["llm_switch"]),
        disabled=not adapters,
        help="Adds the QLoRA-tuned Qwen3-8B verdict for the model's dataset. Loads 4-bit "
        "weights (about 6 GB of GPU memory) the first time.",
    )


def waiting_panel(title: str, body: str, note: str, command: str | None = None) -> None:
    """Content that is not available here yet: what will appear and what fills it."""
    st.markdown(f"**{title}**")
    st.markdown(body)
    st.caption(note)
    if command:
        st.code(command, language="bash", wrap_lines=True)


def no_models_panel() -> None:
    waiting_panel(
        "No trained models yet",
        "The verdict appears once there is a trained model under `outputs/`. The TF-IDF "
        "baseline trains in a minute on a CPU.",
        "Models are found automatically: outputs/<run>/seed<N>/model",
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
        trail = word[len(word.rstrip()) :]
        text = escape_markdown(core)
        if i in tinted:
            text = f":{'red' if value > 0 else 'green'}-background[{text}]"
            if i in strongest:
                text = f"**{text}**"
        out.append(f"{text}{trail}")
    return "".join(out)


def top_pushes(why: serve.Explanation, top: int = 8) -> pd.DataFrame:
    """The strongest pushes: n-grams for a baseline (pairs kept whole), words for SHAP."""
    pairs = why.features if why.features is not None else why.words()
    words = pd.DataFrame({"word": [w.strip() for w, _ in pairs], "push": [v for _, v in pairs]})
    words = words[words["word"].str.contains(r"\w")]  # punctuation alone is not a reason
    return words.reindex(words["push"].abs().sort_values(ascending=False).index).head(top)


def push_unit(model_dir: str) -> str:
    """What a push is measured in: log-odds of false, except for a many-label baseline."""
    many = len(classifier(model_dir).info.labels) > 2
    return (
        "score for the false labels over the rest"
        if is_baseline(model_dir) and many
        else "log-odds of false"
    )


def push_domain(whys: list[serve.Explanation], top: int = 8) -> tuple[float, float]:
    """An x range that holds every chart's bars, with room for the value labels: shared by
    charts that sit side by side, so equal pushes draw equal bars."""
    pushes = pd.concat([top_pushes(why, top)["push"] for why in whys])
    reach = float(pushes.abs().max() or 1.0) * 0.3
    low = min(0.0, float(pushes.min())) - (reach if (pushes < 0).any() else 0)
    high = max(0.0, float(pushes.max())) + (reach if (pushes > 0).any() else 0)
    return low, high


def word_chart(
    why: serve.Explanation,
    unit: str,
    top: int = 8,
    domain: tuple[float, float] | None = None,
) -> alt.LayerChart:
    """Bars of the strongest pushes, strongest first, each labelled with its value (nothing
    hover-only)."""
    words = top_pushes(why, top)
    words["towards"] = ["false" if v > 0 else "true" for v in words["push"]]
    words["word_id"] = [f"{w} ({i})" for i, w in enumerate(words["word"])]
    words["value"] = [f"{v:+.2f}" for v in words["push"]]
    low, high = domain or push_domain([why], top)
    base = alt.Chart(words).encode(
        y=alt.Y(
            "word_id:N",
            sort=list(words["word_id"]),
            title=None,
            axis=alt.Axis(labelExpr="replace(datum.label, / \\(\\d+\\)$/, '')", labelFontSize=14),
        ),
    )
    x = alt.X(
        "push:Q",
        title=f"Push towards false (+) or true (−), {unit}",
        scale=alt.Scale(domain=[low, high], nice=False),
        axis=alt.Axis(labelFontSize=12, titleFontSize=13, titleFontWeight="normal", tickCount=6),
    )
    bars = base.mark_bar(cornerRadius=2, height=15).encode(
        x=x,
        color=alt.Color(
            "towards:N",
            scale=alt.Scale(domain=["false", "true"], range=[PUSH_FALSE, PUSH_TRUE]),
            legend=None,
        ),
    )
    # Two text layers, so a value sits beyond the end of its bar on either side of zero.
    right = (
        base.mark_text(fontSize=13, baseline="middle", align="left", dx=5)
        .encode(x=x, text="value:N")
        .transform_filter("datum.push > 0")
    )
    left = (
        base.mark_text(fontSize=13, baseline="middle", align="right", dx=-5)
        .encode(x=x, text="value:N")
        .transform_filter("datum.push <= 0")
    )
    return alt.layer(bars, right, left).properties(height=32 * len(words) + 10)


def method_line(why: serve.Explanation) -> str:
    if why.evals == 0:
        return (
            f"Exact: each n-gram's weight × its TF-IDF value, no sampling "
            f"({duration(why.seconds)})."
        )
    return f"SHAP partition explainer, {why.evals} model calls ({duration(why.seconds)})."


def why_column(
    model_dir: str,
    pred: serve.Prediction,
    why: serve.Explanation,
    domain: tuple[float, float] | None = None,
) -> None:
    """One model's reading of the claim: its verdict, the claim with its words tinted, and
    the strongest pushes."""
    st.markdown(f"#### {model_label(model_dir)}")
    st.markdown(f"{verdict_text(pred.fake_prob)} · {share(pred.fake_prob)} probability false")
    with st.container(border=True):
        st.markdown(highlighted(why))
    if why.features is not None and abs(why.rest) >= 0.01:
        st.caption(
            f"Character patterns (letter sequences inside words, not drawn on the claim) add "
            f"{why.rest:+.2f} in total."
        )
    st.altair_chart(
        word_chart(why, push_unit(model_dir), domain=domain),
        width="stretch",
        alt=f"The words that moved {model_label(model_dir)}'s verdict most.",
    )
    st.caption(f"{method_line(why)} {plate(model_dir)}")


def push_list(why: serve.Explanation, top: int = 5) -> str:
    """The strongest pushes as one line of text, for narrow columns."""
    words = top_pushes(why, top)
    return " · ".join(
        f":{'red' if v > 0 else 'green'}[**{escape_markdown(w)}**] {v:+.2f}"
        for w, v in zip(words["word"], words["push"], strict=True)
    )


# --------------------------------------------------------------------------- check panels


def verdict_block(result: dict | None, model_dir: str | None, rungs: pd.DataFrame) -> None:
    """The ruling: a serif verdict line, how sure the model is, then the thin ladder."""
    with st.container(border=True):
        if result:
            pred = result["pred"]
            word, color, icon = verdict(pred.fake_prob)
            st.header(f":{color}[{icon} {word}]", anchor=False)
            st.markdown(reading_line(pred, rungs))
        else:
            st.header(":gray[No claim checked yet]", anchor=False)
            st.markdown("Pick an example under the box, or paste a claim and press **Check**.")
        show_ladder(rungs)
        if result:
            st.caption(f"{plate(model_dir)}  \n{timing_line(result)}")


def llm_block(result: dict, encoder_pred: serve.Prediction) -> None:
    llm, llm_dir = result["llm"], result["llm_dir"]
    with st.container(border=True):
        st.markdown("#### Second opinion: Qwen3-8B")
        agree = verdict(encoder_pred.fake_prob)[0] == verdict(llm.fake_prob)[0]
        st.markdown(
            f"{verdict_text(llm.fake_prob)} · {share(llm.fake_prob)} probability false · "
            f"{duration(llm.seconds)} · "
            + (
                "agrees with the model above."
                if agree
                else ":material/call_split: **disagrees with the model above**: a claim worth a "
                "human check."
            )
        )
        show_ladder(prediction_rungs(llm), names="lit")
        st.caption(plate(llm_dir))


def evidence_section(result: dict, clf) -> None:  # noqa: ANN001  (Classifier or Baseline)
    st.subheader("Evidence", anchor=False)
    found = result["found"]
    if found is None:
        waiting_panel(
            "The retrieval index is not built on this machine",
            "Similar fact-checked claims and evidence sentences appear here once it is built "
            "(a few minutes, once per machine).",
            "Built by factcheck.retrieval.",
            "python -m factcheck.retrieval build",
        )
        return
    dataset = DATASETS.get(clf.info.name, clf.info.name)
    similar_col, sentence_col = st.columns(2, gap="large")
    with similar_col:
        st.markdown(f"**Already fact-checked, most similar first** · {dataset} training set")
        for hit in found.similar or []:
            st.markdown(
                f":{rating_color(hit.label_name)}[**{rung_name(hit.label_name)}**] "
                f"{escape_markdown(shorten(hit.text))}  \n"
                f":gray[similarity {hit.score:.2f}]"
            )
        if not found.similar:
            st.caption("No training claims for this dataset in the index.")
    with sentence_col:
        st.markdown("**Closest evidence sentences** · PUBHEALTH fact-check articles, Climate-FEVER")
        for sentence in found.sentences:
            st.markdown(
                f"{escape_markdown(shorten(sentence.text, 300))}  \n"
                f":gray[{serve.POOL_NAMES[sentence.source]} · similarity {sentence.score:.2f}]"
            )
    note = (
        "The model above reads the claim together with these evidence sentences."
        if clf.uses_evidence
        else "Retrieved by meaning (bge-small embeddings); the verdict above reads the claim alone."
    )
    st.caption(f"{note} Retrieval took {duration(found.seconds)}.")


# --------------------------------------------------------------------------- views


def check_view(models: dict[str, Path]) -> None:
    col_claim, col_side = st.columns([6.4, 3.6], gap="medium")
    with col_claim:
        claim = claim_box("claim_check")
        example_links("check")
    with col_side:
        model_dir = st.selectbox(
            "Model",
            list(models),
            format_func=plate,
            placeholder="No trained models yet",
            disabled=not models,
            key="check_model",
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
        st.session_state["result"] = check(model_dir, claim, use_llm)
    result = st.session_state["result"]
    if result and (
        result["claim"] != claim or result["model"] != model_dir or result["use_llm"] != use_llm
    ):
        result = None  # the claim, the model or the LLM switch changed since the last check

    st.space("small")
    if result:
        rungs = prediction_rungs(result["pred"])
    elif model_dir:
        rungs = ladder_rungs(classifier(model_dir).info.labels)
    else:
        rungs = ladder_rungs(("fake", "mixture", "real"))
        rungs["name"] = ["Likely false", "Mixed", "Likely true"]
    verdict_block(result, model_dir, rungs)
    if not result:
        if not models:
            no_models_panel()
        return
    if result["llm"] is not None:
        llm_block(result, result["pred"])

    st.space("medium")
    st.subheader("Why", anchor=False)
    pair = counterpart(model_dir, models)
    readings = [(model_dir, result["pred"], result["evidence"])]
    if pair:
        readings.append((pair, *score(pair, claim)))
    if not is_baseline(model_dir) and pair:
        readings.reverse()  # the ladder reads left to right: the baseline first
    whys = [why_of(m, claim, evidence) for m, _, evidence in readings]
    units = {push_unit(m) for m, _, _ in readings}
    domain = push_domain(whys) if len(units) == 1 else None  # one scale when the units match
    for column, (m, pred, _), why in zip(
        st.columns(len(readings), gap="large"), readings, whys, strict=True
    ):
        with column:
            why_column(m, pred, why, domain)
    st.caption(
        "Brick words push the model towards false, green towards true; the two strongest are "
        "bold."
        + (" Both charts share one scale." if domain else "")
        + (
            " The same claim, read by the bottom and the top of the ladder of models."
            if pair
            else ""
        )
    )

    st.space("medium")
    evidence_section(result, classifier(model_dir))


def ladder_rank(model_dir: str) -> int:
    """Where a servable model sits on the ladder: baseline, encoder, then the LLM."""
    return 0 if is_baseline(model_dir) else 2 if serve.is_qlora(model_dir) else 1


def compare_defaults(models: dict[str, Path]) -> list[str]:
    """Up the ladder for the best model's dataset: its TF-IDF baseline, the fast DistilBERT
    and the best transformer."""
    encoders = [m for m in models if not is_baseline(m) and not serve.is_qlora(m)]
    if not encoders:
        return list(models)[:1]
    best = encoders[0]
    dataset = dataset_of(best)
    baseline = [m for m in models if is_baseline(m) and dataset_of(m) == dataset]
    fast = [
        m
        for m in encoders[1:]
        if dataset_of(m) == dataset and "distilbert" in run_config(m)["model"]["name"]
    ]
    return [*baseline[:1], *fast[:1], best]


def compare_view(models: dict[str, Path]) -> None:
    claim = claim_box("claim_compare")
    example_links("compare")
    st.session_state.pop("auto_check", False)  # every comparison below runs on each rerun
    if not models:
        no_models_panel()
        return
    with st.container(horizontal=True, vertical_alignment="bottom", gap="medium"):
        chosen = st.multiselect(
            "Models to compare, bottom of the ladder first",
            list(models),
            default=compare_defaults(models),
            format_func=plate,
            max_selections=4,
            key="compare_models",
        )
        show_words = st.toggle(
            "Show the words behind each verdict", value=True, key="compare_words"
        )
    if not (claim.strip() and chosen):
        st.markdown("Choose up to four models to read the claim side by side.")
        return

    chosen = sorted(chosen, key=ladder_rank)
    results = {m: score(m, claim) for m in chosen}
    st.space("small")
    for column, (model_dir, (pred, evidence)) in zip(
        st.columns(len(chosen), gap="large"), results.items(), strict=True
    ):
        with column:
            st.markdown(f"#### {model_label(model_dir)}")
            st.caption(plate(model_dir))
            st.subheader(verdict_text(pred.fake_prob), anchor=False)
            rungs = prediction_rungs(pred)
            st.markdown(f"{reading_line(pred, rungs)} · {duration(pred.seconds)}")
            show_ladder(rungs, names="lit")
            if show_words:
                why = why_of(model_dir, claim, evidence)
                with st.container(border=True):
                    st.markdown(highlighted(why))
                st.markdown(push_list(why))
                st.caption(method_line(why))
    verdicts = {verdict(p.fake_prob)[0] for p, _ in results.values()}
    st.space("small")
    if len(results) > 1 and len(verdicts) > 1:
        st.markdown(
            ":material/call_split: **The models disagree on this claim.** Read which words each "
            "one leaned on: the same claim leads them to different rulings."
        )
    elif len(results) > 1:
        st.markdown(f"All {len(results)} models agree: **{verdicts.pop()}**.")
    if show_words:
        st.caption(
            "Brick words push towards false, green towards true. Baseline values are exact "
            "weights × TF-IDF; transformer values are SHAP, in log-odds of false."
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
    st.markdown(f"**{model_label(model_dir)}**")
    st.markdown(f"{verdict_text(pred.fake_prob)} · {share(pred.fake_prob)} probability false")
    show_ladder(prediction_rungs(pred))
    if is_baseline(model_dir):
        why = why_of(model_dir, text, None)
        st.markdown(f"Strongest words: {push_list(why, 4)}")
    st.caption(plate(model_dir))


def shortcut_view(models: dict[str, Path]) -> None:
    shares = reuters_share()
    share_text = (
        f" In the raw training text **{shares['raw']:.1%}** of real articles contain "
        f"“(Reuters)”; after removing source shortcuts, **{shares['clean']:.1%}**."
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
        st.caption(
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
        st.caption(f"Fake article {index + 1} of {len(articles)} · ISOT test set")
        st.markdown(
            (f"**:blue-background[{escape_markdown(DATELINE.strip())}]** " if add_dateline else "")
            + escape_markdown(article)
        )
    if add_dateline:
        st.caption("Blue marks the source shortcut: style, not facts.")

    picks: dict[tuple[str, bool], str] = {}
    for m in models:  # the best logged model per variant and kind (models are sorted best first)
        if dataset_of(m) == "isot" and not serve.is_qlora(m):
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
    for column, variant in zip(st.columns(2, gap="large"), ("raw", "clean"), strict=True):
        with column:
            st.subheader(titles[variant], anchor=False)
            for baseline in (True, False):  # bottom of the ladder first
                model_dir = picks.get((variant, baseline))
                if model_dir is None:
                    waiting_panel(
                        f"No {'TF-IDF baseline' if baseline else 'transformer'} for this variant",
                        f"Needs an ISOT model trained on the {variant} text.",
                        "Shortcut study, experiment E7.",
                        commands[(variant, baseline)],
                    )
                else:
                    shortcut_reading(model_dir, text)
                st.space("small")
    st.markdown(
        "What to look for: the raw-trained transformer flips to *real* on the dateline alone. "
        "The raw TF-IDF baseline shows the same shortcut as numbers (“reuters” pushes towards "
        "real), but blog-style words such as “read more” can outweigh it, so whether it flips "
        "depends on the article. The clean-trained models should barely move."
    )


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


def rung_of(model_name: str, method) -> int:  # noqa: ANN001  (method may be NaN)
    if model_name == "tfidf":
        return 0
    if method in ("glove_svm", "knn_vote"):
        return 1
    if str(model_name).startswith("bilstm"):
        return 2
    if str(model_name).startswith("Qwen/"):
        return 4
    return 3


def run_label(model_name: str, method, run_name: str) -> str:  # noqa: ANN001
    """A short name for the best run on a rung: 'TF-IDF + LR', 'deberta-v3-large LoRA'..."""
    method = method if isinstance(method, str) else ""
    if model_name in ("tfidf", "glove-300d") or method == "knn_vote":
        return METHOD_NAMES.get(method, method)
    if str(model_name).startswith("bilstm"):
        return "Bi-LSTM + GloVe"
    if str(model_name).startswith("Qwen/"):
        return f"Qwen3-8B {METHOD_NAMES.get(method, method)}"
    label = MODEL_NAMES.get(model_name, str(model_name).split("/")[-1])
    if method == "lora":
        label += " LoRA"
    if "evidence" in run_name:
        label += " + evidence"
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
            best["macro_f1"], best["model_name"], best["method"], best["run_name"], strict=False
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
    st.subheader(f"Specifications: {passing} of {len(specs)} passing", anchor=False)
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
            "Every training run appends its scores to `results/runs.csv`; they appear here "
            "and in the specifications, with no numbers typed by hand.",
            "Logged by factcheck.train.",
            "python -m factcheck.train --config configs/classic/tfidf_lr_covid.yaml",
        )
        return
    debug = test["subset"].notna() if "subset" in test else pd.Series(False, index=test.index)

    st.subheader("The ladder of models, by dataset", anchor=False)
    st.markdown(
        "The best run on each rung, from word counts at the top to the 8B LLM at the bottom. "
        "Read the gaps: where TF-IDF is already close, the bigger models add little."
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


def gpu_memory_text() -> str:
    """Live GPU memory against spec S8 (everything in the app fits in 8 GB)."""
    if not torch.cuda.is_available():
        return "CPU only"
    used = torch.cuda.memory_reserved() / 2**30
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    return (
        f"GPU memory held by the app right now {used:.1f} of {total:.1f} GB (spec S8, "
        f"{SPEC_S8_GB:.0f} GB with every model loaded, is checked by hand)"
    )


def footer(models: dict[str, Path]) -> None:
    """Run last, so the GPU memory counts every model this run loaded."""
    st.space("large")
    st.divider()
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    st.caption(
        f"Triage support for human fact-checkers, not a judge of truth. "
        f"{len(models)} trained model{'s' if len(models) != 1 else ''} found on {device} · "
        f"{gpu_memory_text()}."
    )


def main() -> None:
    st.set_page_config(
        page_title="Fact-Check Lab",
        page_icon=":material/fact_check:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    init_state()
    models = find_models(include_llm=st.session_state["use_llm"])
    has_isot = any(dataset_of(m) == "isot" for m in models)
    views = [v for v in VIEWS if v != "shortcut" or has_isot]
    if st.session_state["view"] not in views:
        st.session_state["view"] = "check"

    with st.container(horizontal=True, vertical_alignment="bottom", gap="large"):
        with st.container(gap=None):
            st.title("Fact-Check Lab", anchor=False)
            st.markdown(
                ":gray[Paste a claim, get an explained verdict, from word counts up to an 8B "
                "LLM, on four misinformation datasets.]"
            )
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
    footer(models)


main()
