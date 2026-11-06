"""Fact-Check Lab: paste a claim, get an explained verdict (T11, T22, T35).

    streamlit run app/app.py

Check a claim: the verdict on the dataset's rating ladder from any trained encoder, the
evidence (similar already fact-checked claims with their verdicts, and the closest evidence
sentences from PUBHEALTH's fact-check articles and Climate-FEVER), the words that drove it
(SHAP), an optional second opinion from Qwen3-8B with its QLoRA adapter, and the response time
against specs S6 (encoder) and S7 (retrieval + LLM verdict). Compare models side by side, run
the shortcut demo (a Reuters dateline flips the raw-trained ISOT model), and read every logged
run and the specs board from results/runs.csv.

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

EXAMPLE_CLAIMS = [
    "Drinking hot water every 15 minutes kills the coronavirus.",
    "The CDC recommends that people wear masks in public settings.",
    "Says the unemployment rate has doubled since the president took office.",
    "Vitamin D supplements cure COVID-19 in 24 hours.",
]
SPEC_S6_SECONDS = 0.050  # encoder latency per claim
SPEC_S7_SECONDS = 5.0  # full response: retrieval + LLM verdict
SPEC_S8_GB = 8.0  # everything in the app on the laptop GPU
DATELINE = "WASHINGTON (Reuters) - "

VIEWS = {
    "check": ":material/fact_check: Check a claim",
    "compare": ":material/compare_arrows: Compare models",
    "shortcut": ":material/science: Shortcut demo",
    "results": ":material/leaderboard: Results",
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
}

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
# Rung colours, dark enough for white text (WCAG AA) on every lit rung.
RUNG_FILL = {
    "pants-fire": "#8c0a1f",
    "fake": "#c8102e",
    "refutes": "#c8102e",
    "false": "#c8102e",
    "barely-true": "#c4561a",
    "half-true": "#8f6b00",
    "mixture": "#8f6b00",
    "disputed": "#8f6b00",
    "mostly-true": "#4f7d28",
    "true": "#2e7d3c",
    "real": "#2e7d3c",
    "supports": "#2e7d3c",
}
RUNG_NAMES = {
    "pants-fire": "Pants on fire",
    "not_enough_info": "Not enough info",
}
UNLIT_FILL, UNLIT_INK, OFF_SCALE_FILL, LIT_OUTLINE_INK = "#eef2f7", "#4a5b75", "#4a5b75", "#13294b"

# (upper bound on P(false), verdict, badge colour, icon)
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
    parts.append(f"seed {cfg['run']['seed']}")
    f1 = test_f1(model_dir)
    if f1 is not None:
        parts.append(f"test F1 {f1:.3f}")
    return " · ".join(parts)


@st.cache_resource(max_entries=3, show_spinner=False)
def load_encoder(model_dir: str) -> serve.Classifier:
    return serve.Classifier(model_dir)


@st.cache_resource(max_entries=1, show_spinner=False)
def load_llm(model_dir: str) -> serve.Classifier:
    return serve.Classifier(model_dir)  # 4-bit Qwen3-8B + adapter; one dataset at a time


def classifier(model_dir: str) -> serve.Classifier:
    """The loaded model; the first use shows a spinner, and loading is never timed."""
    loader = load_llm if serve.is_qlora(model_dir) else load_encoder
    message = (
        "Loading Qwen3-8B in 4-bit with its adapter (first use only, about a minute)…"
        if serve.is_qlora(model_dir)
        else "Loading the model (first check only)…"
    )
    with st.spinner(message):
        return loader(model_dir)


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


def score(model_dir: str, claim: str) -> serve.Prediction:
    """One model's verdict on a claim, with retrieved evidence when the model reads it."""
    clf = classifier(model_dir)
    found = find_evidence(claim, clf.info.name) if clf.uses_evidence else None
    return clf.predict(claim, found.sentence_texts() if found else None)


def check(model_dir: str, claim: str, use_llm: bool) -> dict:
    """Everything the Check view shows, timed: retrieval, the encoder and the LLM."""
    clf = classifier(model_dir)
    found = find_evidence(claim, clf.info.name)
    pred = clf.predict(claim, found.sentence_texts() if found and clf.uses_evidence else None)
    llm_dir = llm_adapters().get(clf.info.name) if use_llm and not clf.qlora else None
    llm = classifier(llm_dir).predict(claim) if llm_dir else None
    return {
        "claim": claim,
        "model": model_dir,
        "pred": pred,
        "found": found,
        "llm_dir": llm_dir,
        "llm": llm,
        "use_llm": use_llm,
        "why": None,
    }


# --------------------------------------------------------------------------- rating ladder


def verdict(fake_prob: float) -> tuple[str, str, str]:
    return next((word, color, icon) for bound, word, color, icon in VERDICTS if fake_prob <= bound)


def rung_name(label: str) -> str:
    return RUNG_NAMES.get(label, label.replace("-", " ").replace("_", " ").capitalize())


def rating_color(label: str) -> str:
    """Badge colour of a verdict label: red false, orange mixed, green true, grey off-scale."""
    score_ = TRUTH_SCORE.get(label)
    if score_ is None:
        return "gray"
    return "red" if score_ <= 0.15 else "orange" if score_ < 0.75 else "green"


def ladder_rungs(
    labels: tuple[str, ...], probs: dict[str, float] | None = None, fake_prob: float | None = None
) -> pd.DataFrame:
    """The dataset's labels ordered from false to true; the most probable one is lit.

    The lit rung is filled solid only when the verdict is confident; for a mixed verdict it
    is outlined, so a 21% top label never looks like a ruling.
    """
    rungs = pd.DataFrame({"label": list(labels)})
    rungs["score"] = rungs["label"].map(TRUTH_SCORE)
    rungs = rungs.sort_values("score", na_position="last", kind="stable").reset_index(drop=True)
    rungs["name"] = [rung_name(x) for x in rungs["label"]]
    rungs["order"] = range(len(rungs))
    rungs["prob"] = rungs["label"].map(probs) if probs else float("nan")
    rungs["pct"] = rungs["prob"].map(lambda p: "–" if pd.isna(p) else f"{p:.0%}")
    rungs["lit"] = rungs["prob"] == rungs["prob"].max() if probs else False
    confident = fake_prob is not None and verdict(fake_prob)[0] != MIXED
    colour = [RUNG_FILL.get(label, OFF_SCALE_FILL) for label in rungs["label"]]
    rungs["fill"] = [
        c if lit and confident else UNLIT_FILL for c, lit in zip(colour, rungs["lit"], strict=True)
    ]
    rungs["stroke"] = [
        c if lit else UNLIT_FILL for c, lit in zip(colour, rungs["lit"], strict=True)
    ]
    rungs["ink"] = [
        ("#ffffff" if confident else LIT_OUTLINE_INK) if lit else UNLIT_INK for lit in rungs["lit"]
    ]
    return rungs


def ladder_chart(rungs: pd.DataFrame, height: int = 96, names: str = "all") -> alt.LayerChart:
    """The signature view: a row of rungs, the lit one taller, each showing its probability.

    ``names="lit"`` prints only the lit rung's name, for narrow ladders (Compare).
    """
    inset = 10
    named = rungs["lit"] if names == "lit" else pd.Series(True, index=rungs.index)
    rungs = rungs.assign(
        top=[1.5 if lit else inset for lit in rungs["lit"]],
        bottom=[height - 1.5 if lit else height - inset for lit in rungs["lit"]],
        stroke_width=[3 if lit else 0 for lit in rungs["lit"]],
        name_text=[name if show else "" for name, show in zip(rungs["name"], named, strict=True)],
        name_size=[19 if lit else 15 for lit in rungs["lit"]],
        name_y=height / 2 - 10,
        pct_y=[height / 2 + 13 if show else height / 2 for show in named],
    )
    x = alt.X("order:O", axis=None, scale=alt.Scale(paddingInner=0.05, paddingOuter=0))
    base = alt.Chart(rungs).encode(
        x=x,
        tooltip=[alt.Tooltip("name:N", title="Label"), alt.Tooltip("pct:N", title="Probability")],
    )
    rects = base.mark_rect(cornerRadius=8).encode(
        y=alt.Y("top:Q", scale=None),
        y2="bottom:Q",
        color=alt.Color("fill:N", scale=None),
        stroke=alt.Stroke("stroke:N", scale=None),
        strokeWidth=alt.StrokeWidth("stroke_width:Q", scale=None),
    )
    ink = alt.Color("ink:N", scale=None)
    pct = base.mark_text(fontSize=15, fontWeight=600).encode(
        text="pct:N", color=ink, y=alt.Y("pct_y:Q", scale=None)
    )
    name = base.mark_text(fontWeight=700, baseline="middle").encode(
        text="name_text:N",
        color=ink,
        size=alt.Size("name_size:Q", scale=None),
        y=alt.Y("name_y:Q", scale=None),
    )
    return alt.layer(rects, pct, name).properties(height=height).configure_view(strokeWidth=0)


def show_ladder(rungs: pd.DataFrame, height: int = 96, names: str = "all") -> None:
    lit = rungs[rungs["lit"]]
    alt_text = (
        f"Rating ladder from false to true; {lit['name'].iloc[0]} is lit at {lit['pct'].iloc[0]}."
        if len(lit)
        else "Rating ladder from false to true, waiting for a check."
    )
    st.altair_chart(ladder_chart(rungs, height, names), width="stretch", alt=alt_text)
    with st.container(horizontal=True, horizontal_alignment="distribute"):
        st.markdown(":material/arrow_back: false", width="content")
        st.markdown("true :material/arrow_forward:", width="content")


def prediction_rungs(model_dir: str, pred: serve.Prediction) -> pd.DataFrame:
    return ladder_rungs(tuple(pred.label_probs), pred.label_probs, pred.fake_prob)


def duration(seconds: float) -> str:
    return f"{seconds * 1000:.0f} ms" if seconds < 1 else f"{seconds:.2f} s"


def verdict_line(pred: serve.Prediction, rungs: pd.DataFrame) -> None:
    word, color, icon = verdict(pred.fake_prob)
    lit = rungs[rungs["lit"]].iloc[0]
    within = pred.seconds <= SPEC_S6_SECONDS
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.markdown(f"### :{color}[{icon} {word}]", width="content")
        st.badge(f"{pred.fake_prob:.0%} probability false", color=color)
        st.badge(f"Top label: {lit['name']}", color="gray")
        st.badge(
            f"{duration(pred.seconds)} encoder · {'within' if within else 'over'} the "
            f"{SPEC_S6_SECONDS * 1000:.0f} ms target (S6)",
            icon=":material/timer:",
            color="green" if within else "orange",
        )


def response_line(result: dict) -> None:
    """Spec S7: the full response is retrieval plus the LLM verdict (and the encoder)."""
    found, pred, llm = result["found"], result["pred"], result["llm"]
    parts = [("retrieval", found.seconds if found else None), ("encoder", pred.seconds)]
    parts.append(("Qwen3-8B", llm.seconds if llm else None))
    timed = [(name, s) for name, s in parts if s is not None]
    total = sum(s for _, s in timed)
    breakdown = " + ".join(f"{name} {duration(s)}" for name, s in timed)
    if llm is None:
        st.caption(
            f"Response {duration(total)} ({breakdown}). Spec S7 times retrieval plus the LLM "
            "verdict: turn on the Qwen3-8B second opinion in the sidebar to measure it."
        )
        return
    within = total <= SPEC_S7_SECONDS
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.badge(
            f"{duration(total)} full response · {'within' if within else 'over'} the "
            f"{SPEC_S7_SECONDS:.0f} s target (S7)",
            icon=":material/timer:",
            color="green" if within else "orange",
        )
        st.caption(breakdown, width="content")


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
    st.session_state.setdefault("claim", EXAMPLE_CLAIMS[0])
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("auto_check", False)
    st.session_state.setdefault("use_llm", False)


def use_example(text: str) -> None:
    st.session_state["claim"] = text
    for key in CLAIM_KEYS:
        st.session_state[key] = text
    st.session_state["auto_check"] = True
    if st.session_state.get("view") != "compare":
        st.session_state["view"] = "check"  # compare re-runs by itself; elsewhere, go and check


def claim_box(key: str, label: str = "Claim or post") -> str:
    """A claim text box per view that shares one value across views."""
    st.session_state.setdefault(key, st.session_state["claim"])
    claim = st.text_area(
        label,
        key=key,
        height=110,
        max_chars=5000,
        placeholder="Paste a claim, a post or a headline…",
    )
    st.session_state["claim"] = claim
    return claim


def waiting_panel(title: str, icon: str, body: str, note: str, command: str | None = None) -> None:
    """A panel whose content is not available here yet: what will appear and what fills it."""
    with st.container(border=True, height="stretch"):
        st.markdown(f"#### {icon} {title}")
        st.markdown(body)
        st.caption(note)
        if command:
            st.code(command, language="bash", wrap_lines=True)


def no_models_panel() -> None:
    waiting_panel(
        "No trained models yet",
        ":material/model_training:",
        "The ladder lights up once there is a trained encoder under `outputs/`. "
        "The fast DistilBERT baseline is the one the demo uses first.",
        "Models are found automatically: outputs/<run>/seed<N>/model",
        "python -m factcheck.train --config configs/encoder/distilbert_covid.yaml",
    )


def escape_markdown(text: str) -> str:
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|>~$:<])", r"\\\1", text)


def shorten(text: str, limit: int = 220) -> str:
    text = " ".join(str(text).replace('""', '"').split())  # PUBHEALTH quotes are CSV-doubled
    return text if len(text) <= limit else text[: limit - 1].rsplit(" ", 1)[0] + "…"


# --------------------------------------------------------------------------- evidence and why


def evidence_panel(result: dict, clf: serve.Classifier) -> None:
    found = result["found"]
    with st.container(border=True, height="stretch"):
        st.markdown("#### :material/library_books: Evidence")
        if found is None:
            st.markdown(
                "Similar fact-checked claims and evidence sentences appear here once the "
                "retrieval index is built (a few minutes, once per machine)."
            )
            st.code("python -m factcheck.retrieval build", language="bash")
            return
        dataset = DATASETS.get(clf.info.name, clf.info.name)
        if found.similar:
            st.markdown(f"**Already fact-checked, most similar first** · {dataset} training set")
            for hit in found.similar:
                st.markdown(
                    f":{rating_color(hit.label_name)}-badge[{rung_name(hit.label_name)}] "
                    f"{escape_markdown(shorten(hit.text))} :gray[· similarity {hit.score:.2f}]"
                )
        st.space("small")
        st.markdown("**Closest evidence sentences** · PUBHEALTH fact-check articles, Climate-FEVER")
        for sentence in found.sentences:
            st.markdown(
                f"{escape_markdown(shorten(sentence.text, 300))}  \n"
                f":gray[{serve.POOL_NAMES[sentence.source]} · similarity {sentence.score:.2f}]"
            )
        note = (
            "This model reads the claim together with these evidence sentences."
            if clf.uses_evidence
            else "Retrieved by meaning (bge-small embeddings); the verdict above reads the claim alone."
        )
        st.caption(f"{note} Retrieval took {duration(found.seconds)}.")


def highlighted(why: serve.Explanation) -> str:
    """The claim as markdown, each word tinted by its push: red towards false, green towards
    true; bold for the strongest. Words with little effect stay plain."""
    words = why.words()
    strongest = max((abs(v) for _, v in words), default=0.0) or 1e-9
    out = []
    for word, value in words:
        core = word.strip()
        trail = word[len(word.rstrip()) :]
        share = abs(value) / strongest
        text = escape_markdown(core)
        if abs(value) >= 0.02 and share >= 0.2:
            text = f":{'red' if value > 0 else 'green'}-background[{text}]"
            if share >= 0.6:
                text = f"**{text}**"
        out.append(f"{text}{trail}")
    return "".join(out)


def word_chart(why: serve.Explanation, top: int = 8) -> alt.Chart:
    pairs = why.words()
    words = pd.DataFrame({"word": [w.strip() for w, _ in pairs], "push": [v for _, v in pairs]})
    words = words[words["word"].str.contains(r"\w")]  # punctuation alone is not a reason
    words = words.reindex(words["push"].abs().sort_values(ascending=False).index).head(top)
    words["towards"] = ["false" if v > 0 else "true" for v in words["push"]]
    words["word_id"] = [f"{w} ({i})" for i, w in enumerate(words["word"])]
    return (
        alt.Chart(words)
        .mark_bar(cornerRadius=4, height=16)
        .encode(
            x=alt.X("push:Q", title="Push towards false (+) or true (−), log-odds"),
            y=alt.Y(
                "word_id:N",
                sort="-x",
                title=None,
                axis=alt.Axis(labelExpr="replace(datum.label, / \\(\\d+\\)$/, '')"),
            ),
            color=alt.Color(
                "towards:N",
                scale=alt.Scale(domain=["false", "true"], range=["#c8102e", "#2e7d3c"]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("word:N", title="Word"),
                alt.Tooltip("push:Q", title="Change in log-odds of false", format="+.2f"),
            ],
        )
        .properties(height=34 * len(words))
    )


def why_panel(result: dict, clf: serve.Classifier) -> None:
    with st.container(border=True, height="stretch"):
        st.markdown("#### :material/psychology: Why this verdict")
        if result["why"] is None:
            evidence = (
                result["found"].sentence_texts() if result["found"] and clf.uses_evidence else None
            )
            with st.spinner("Finding the words that drove the verdict (SHAP)…"):
                result["why"] = serve.explain(clf, result["claim"], evidence)
        why = result["why"]
        st.markdown(highlighted(why))
        st.caption(
            "Red words push the model towards false, green towards true; bold marks the "
            "strongest. Values are changes in the log-odds that the claim is false."
        )
        st.altair_chart(
            word_chart(why), width="stretch", alt="The words that moved the verdict most."
        )
        st.caption(
            f"SHAP partition explainer, {why.evals} model calls, {duration(why.seconds)}. "
            "Not counted in the response time."
        )


def llm_panel(result: dict, encoder_pred: serve.Prediction) -> None:
    llm, llm_dir = result["llm"], result["llm_dir"]
    with st.container(border=True):
        left, right = st.columns([3, 2], vertical_alignment="center")
        with left:
            st.markdown(f"**Second opinion** · {plate(llm_dir)}")
            show_ladder(prediction_rungs(llm_dir, llm), height=72, names="lit")
        with right:
            word, color, icon = verdict(llm.fake_prob)
            st.markdown(f"#### :{color}[{icon} {word}]")
            st.markdown(f"{llm.fake_prob:.0%} probability false · {duration(llm.seconds)}")
            same = verdict(encoder_pred.fake_prob)[0] == word
            if same:
                st.markdown(":material/handshake: Agrees with the encoder.")
            else:
                st.markdown(
                    ":orange[:material/call_split: Disagrees with the encoder]: "
                    "a claim worth a human check."
                )


# --------------------------------------------------------------------------- views


def check_view(models: dict[str, Path]) -> None:
    col_claim, col_model, col_button = st.columns([6, 4, 1.4], vertical_alignment="bottom")
    with col_claim:
        claim = claim_box("claim_check")
    with col_model:
        model_dir = st.selectbox(
            "Model",
            list(models),
            format_func=plate,
            placeholder="No trained models yet",
            disabled=not models,
            key="check_model",
        )
    with col_button:
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
        rungs = prediction_rungs(model_dir, result["pred"])
    elif model_dir:
        rungs = ladder_rungs(classifier(model_dir).info.labels)
    else:
        rungs = ladder_rungs(("fake", "mixture", "real"))
        rungs["name"] = ["Likely false", "Mixed", "Likely true"]
    show_ladder(rungs)
    if not result:
        if models:
            st.markdown("Pick an example claim in the sidebar, or paste one and press **Check**.")
        else:
            no_models_panel()
        return

    verdict_line(result["pred"], rungs)
    response_line(result)
    if result["llm"] is not None:
        llm_panel(result, result["pred"])
    st.space("small")
    clf = classifier(model_dir)
    evidence_col, why_col = st.columns(2)
    with evidence_col:
        evidence_panel(result, clf)
    with why_col:
        why_panel(result, clf)


def compare_defaults(models: dict[str, Path]) -> list[str]:
    """The best model next to the fast DistilBERT baseline of the same dataset, if there is one."""
    names = list(models)
    if not names:
        return []
    best = names[0]
    dataset = run_config(best)["data"]["dataset"]
    fast = [
        m
        for m in names[1:]
        if run_config(m)["data"]["dataset"] == dataset
        and "distilbert" in run_config(m)["model"]["name"]
    ]
    return [best, *(fast or names[1:2])]


def compare_view(models: dict[str, Path]) -> None:
    claim = claim_box("claim_compare")
    st.session_state.pop("auto_check", False)  # every comparison below runs on each rerun
    if not models:
        no_models_panel()
        return
    chosen = st.multiselect(
        "Models to compare",
        list(models),
        default=compare_defaults(models),
        format_func=plate,
        max_selections=3,
        key="compare_models",
    )
    if not (claim.strip() and chosen):
        st.markdown("Choose up to three models to see them side by side.")
        return

    results = {m: score(m, claim) for m in chosen}
    columns = st.columns(len(chosen))
    for column, (model_dir, pred) in zip(columns, results.items(), strict=False):
        with column, st.container(border=True, height="stretch"):
            st.markdown(f"**{model_label(model_dir)}**")
            st.caption(plate(model_dir))
            rungs = prediction_rungs(model_dir, pred)
            show_ladder(rungs, height=80, names="lit")
            word, color, icon = verdict(pred.fake_prob)
            st.markdown(f"#### :{color}[{icon} {word}]")
            lit = rungs[rungs["lit"]].iloc[0]
            st.markdown(
                f"{lit['name']} {lit['pct']} · {pred.fake_prob:.0%} false · "
                f"{duration(pred.seconds)}"
            )
    verdicts = {verdict(p.fake_prob)[0] for p in results.values()}
    if len(results) > 1 and len(verdicts) > 1:
        st.warning(
            "Compare their ladders: the same words lead them to different ratings.",
            title="The models disagree on this claim",
            icon=":material/call_split:",
        )
    elif len(results) > 1:
        st.markdown(f"All {len(results)} models agree: **{verdicts.pop()}**.")


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


def shortcut_view(models: dict[str, Path]) -> None:
    st.markdown(
        "Real ISOT articles almost all open with a Reuters dateline, so a model trained on raw "
        "text can learn the dateline instead of the content. Add the dateline to a fake article "
        "and watch which model flips."
    )
    shares = reuters_share()
    if shares:
        with st.container(horizontal=True):
            st.metric(
                "Real training articles with “(Reuters)”, raw text",
                f"{shares['raw']:.1%}",
                border=True,
            )
            st.metric("…after removing source shortcuts", f"{shares['clean']:.1%}", border=True)

    articles = fake_articles()
    if not articles:
        st.caption(
            "Prepare ISOT first: `python scripts/download_data.py isot && python scripts/prepare_data.py isot`."
        )
        return
    index = st.session_state.get("article_index", 0) % len(articles)
    article = articles[index]
    with st.container(horizontal=True, vertical_alignment="center"):
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

    isot = {}
    for m in models:  # the best logged model per variant (models are sorted best first)
        cfg = run_config(m)
        if cfg["data"]["dataset"] == "isot" and not serve.is_qlora(m):
            isot.setdefault(cfg["data"].get("variant", "raw"), m)
    commands = {
        "raw": "python -m factcheck.train --config configs/encoder/roberta_isot_raw.yaml",
        "clean": "python -m factcheck.train --config configs/encoder/roberta_isot_clean.yaml",
    }
    titles = {"raw": "Trained on raw text", "clean": "Trained on cleaned text"}
    for column, variant in zip(st.columns(2), ("raw", "clean"), strict=True):
        with column:
            model_dir = isot.get(variant)
            if model_dir is None:
                waiting_panel(
                    titles[variant],
                    ":material/model_training:",
                    "Needs an ISOT encoder trained on the "
                    f"{'raw' if variant == 'raw' else 'cleaned'} text.",
                    "Shortcut study, experiment E7.",
                    commands[variant],
                )
                continue
            pred = score(model_dir, text)
            with st.container(border=True, height="stretch"):
                st.markdown(f"#### {titles[variant]}")
                st.caption(plate(model_dir))
                show_ladder(prediction_rungs(model_dir, pred), height=80)
                word, color, icon = verdict(pred.fake_prob)
                st.markdown(f"#### :{color}[{icon} {word}]")
    st.markdown(
        "What to look for: the raw-trained model should flip to *real* when the dateline is added; "
        "the clean-trained model should not move."
    )


SPEC_BADGES = {
    "PASS": ("Pass", "green", ":material/check:"),
    "FAIL": ("Fail", "red", ":material/close:"),
    "no runs yet": ("No runs yet", "gray", ":material/hourglass_empty:"),
    "check by hand": ("Check by hand", "blue", ":material/back_hand:"),
}
RUN_COLUMNS = {
    "run_name": st.column_config.TextColumn("Run", pinned=True),
    "dataset": "Dataset",
    "variant": "Variant",
    "model_name": "Model",
    "method": "Method",
    "seed": st.column_config.NumberColumn("Seed", format="plain"),
    "macro_f1": st.column_config.ProgressColumn(
        "Macro-F1", min_value=0.0, max_value=1.0, format="%.3f"
    ),
    "accuracy": st.column_config.NumberColumn("Accuracy", format="%.3f"),
    "weighted_f1": st.column_config.NumberColumn("Weighted F1", format="%.3f"),
    "roc_auc_fake": st.column_config.NumberColumn("ROC-AUC (fake)", format="%.3f"),
    "ece": st.column_config.NumberColumn("ECE", format="%.3f"),
    "trainable_params": st.column_config.NumberColumn("Trainable params", format="compact"),
    "latency_ms": st.column_config.NumberColumn("Latency", format="%.1f ms"),
}


def specs_scorecard(specs: pd.DataFrame) -> None:
    passing = int((specs["status"] == "PASS").sum())
    st.subheader(f"Specifications · {passing} of {len(specs)} passing", anchor=False)
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
        label, color, icon = SPEC_BADGES.get(row.status, (row.status, "gray", None))
        with cells[4]:
            st.badge(label, color=color, icon=icon)


def results_view() -> None:
    runs = load_runs()
    specs_scorecard(check_specs(runs))

    st.space("medium")
    st.subheader("Runs on the test split", anchor=False)
    test = runs[runs["split"] == "test"] if "split" in runs else pd.DataFrame()
    if test.empty:
        waiting_panel(
            "No logged runs yet",
            ":material/table_chart:",
            "Every training run appends its scores to `results/runs.csv`; they appear here "
            "and in the specifications above, with no numbers typed by hand.",
            "Logged by factcheck.train.",
            "python -m factcheck.train --config configs/encoder/distilbert_covid.yaml",
        )
        return
    debug = test["subset"].notna() if "subset" in test else pd.Series(False, index=test.index)
    if debug.any() and not st.toggle(
        f"Include {int(debug.sum())} debug runs on data subsets", key="show_debug"
    ):
        test = test[~debug]
    if test.empty:
        st.caption("Only debug runs so far.")
        return

    # One bar per model and dataset: the run with the best MEAN over its seeds, never the
    # luckiest single seed.
    per_run = (
        test.assign(model=test["model_name"].str.split("/").str[-1])
        .groupby(["dataset", "model", "run_name"], as_index=False)
        .agg(macro_f1=("macro_f1", "mean"), sd=("macro_f1", "std"), seeds=("seed", "nunique"))
    )
    best = per_run.loc[per_run.groupby(["dataset", "model"])["macro_f1"].idxmax()]
    best["sd"] = best["sd"].fillna(0.0)
    chart = (
        alt.Chart(best)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X(
                "macro_f1:Q",
                title="Test macro-F1, mean over seeds (best run per model)",
                scale=alt.Scale(domain=[0, 1]),
            ),
            y=alt.Y("model:N", title=None, sort="-x", axis=alt.Axis(labelLimit=320)),
            color=alt.Color("dataset:N", title="Dataset"),
            yOffset="dataset:N",
            tooltip=[
                "dataset",
                "model",
                alt.Tooltip("run_name:N", title="Run"),
                alt.Tooltip("macro_f1:Q", title="Mean macro-F1", format=".3f"),
                alt.Tooltip("sd:Q", title="Std over seeds", format=".3f"),
                alt.Tooltip("seeds:Q", title="Seeds"),
            ],
        )
    )
    st.altair_chart(
        chart, width="stretch", alt="Mean test macro-F1 of the best run per model and dataset."
    )
    st.dataframe(
        test[[c for c in RUN_COLUMNS if c in test]].dropna(axis="columns", how="all"),
        column_config=RUN_COLUMNS,
        hide_index=True,
        alt="Every logged test-split run with its scores.",
    )


# --------------------------------------------------------------------------- page


def gpu_memory_caption() -> None:
    """Live GPU memory against spec S8 (everything in the app fits in 8 GB)."""
    if not torch.cuda.is_available():
        st.caption("CPU only: no GPU memory to report.")
        return
    used = torch.cuda.memory_reserved() / 2**30
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    within = used <= SPEC_S8_GB
    st.caption(
        f"GPU memory held by the app: {used:.1f} of {total:.1f} GB · "
        f"{'within' if within else 'over'} the {SPEC_S8_GB:.0f} GB target (S8)"
    )


def sidebar(models: dict[str, Path]) -> None:
    with st.sidebar:
        st.subheader("Example claims", anchor=False)
        for i, example in enumerate(EXAMPLE_CLAIMS):
            st.button(
                example, key=f"example_{i}", on_click=use_example, args=(example,), width="stretch"
            )
        st.space("medium")
        adapters = llm_adapters()
        st.toggle(
            "Qwen3-8B second opinion",
            key="use_llm",
            disabled=not adapters,
            help="Adds the QLoRA-tuned Qwen3-8B verdict for the model's dataset. Loads 4-bit "
            "weights (about 6 GB of GPU memory) the first time.",
        )
        if not adapters:
            st.caption("No QLoRA adapters under outputs/ yet.")
        st.space("medium")
        device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        st.caption(f"{len(models)} trained model{'s' if len(models) != 1 else ''} found · {device}")
        gpu_memory_caption()
        st.caption("Triage support for human fact-checkers, not a judge of truth.")


def main() -> None:
    st.set_page_config(
        page_title="Fact-Check Lab", page_icon=":material/fact_check:", layout="wide"
    )
    init_state()
    models = find_models(include_llm=st.session_state["use_llm"])

    st.title("Fact-Check Lab", anchor=False)
    st.caption(
        "Paste a claim, get an explained verdict from models trained on four misinformation "
        "datasets."
    )
    has_isot = any(run_config(m)["data"]["dataset"] == "isot" for m in models)
    views = [v for v in VIEWS if v != "shortcut" or has_isot]
    if st.session_state["view"] not in views:
        st.session_state["view"] = "check"
    view = st.segmented_control(
        "View",
        views,
        format_func=VIEWS.get,
        required=True,
        key="view",
        label_visibility="collapsed",
    )
    st.query_params["view"] = view  # ?view=results etc. opens a view directly
    st.space("small")
    if view == "compare":
        compare_view(models)
    elif view == "shortcut":
        shortcut_view(models)
    elif view == "results":
        results_view()
    else:
        check_view(models)
    sidebar(models)  # last, so the GPU memory caption counts every model this run loaded


main()
