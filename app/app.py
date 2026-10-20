"""Fact-Check Lab: paste a claim, get an explained verdict (owner M5).

    streamlit run app/app.py

Works today: the rating ladder, verdict and response time from any encoder trained with
``python -m factcheck.train`` (models are found under outputs/), side-by-side model
comparison, the shortcut demo (once raw- and clean-trained ISOT models exist) and the
results explorer (the shortcut tab appears once an ISOT model exists). Still to add:
evidence (M3), SHAP "why" highlights (M4) and the RAG LLM (M3); waiting_panel() is ready
for them.

The theme lives in app/.streamlit/config.toml; the app uses native Streamlit only.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import torch
import yaml

from factcheck.data import get_dataset
from factcheck.data.prepare import load_splits
from factcheck.evaluate import check_specs
from factcheck.paths import OUTPUT_DIR
from factcheck.results import load_runs

EXAMPLE_CLAIMS = [
    "Drinking hot water every 15 minutes kills the coronavirus.",
    "The CDC recommends that people wear masks in public settings.",
    "Says the unemployment rate has doubled since the president took office.",
    "Vitamin D supplements cure COVID-19 in 24 hours.",
]
# The app times the encoder alone, so it reports against S6 (encoder latency). S7, the full
# retrieval + LLM response, can be timed once the RAG pipeline is wired in.
SPEC_S6_SECONDS = 0.050
DATELINE = "WASHINGTON (Reuters) - "

VIEWS = {
    "check": ":material/fact_check: Check a claim",
    "compare": ":material/compare_arrows: Compare models",
    "shortcut": ":material/science: Shortcut demo",
    "results": ":material/leaderboard: Results",
}
CLAIM_KEYS = ("claim_check", "claim_compare")

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


def find_models() -> dict[str, Path]:
    """Every saved encoder: outputs/<run>/seed<N>/model with its config.yaml next to it."""
    return {
        str(p): p
        for p in sorted(OUTPUT_DIR.glob("*/seed*/model"))
        if (p / "config.json").exists() and (p.parent / "config.yaml").exists()
    }


@st.cache_data(max_entries=64)
def run_config(model_dir: str) -> dict:
    return yaml.safe_load((Path(model_dir).parent / "config.yaml").read_text(encoding="utf-8"))


def plate(model_dir: str) -> str:
    """One fixed ID plate per model, used everywhere a model is named."""
    cfg = run_config(model_dir)
    data, model = cfg["data"], cfg["model"]
    parts = [data["dataset"], model["name"].split("/")[-1]]
    if model.get("lora", {}).get("enabled"):
        parts.append("LoRA")
    if data.get("variant", "raw") != "raw":
        parts.append(data["variant"])
    parts.append(f"seed {cfg['run']['seed']}")
    return " · ".join(parts)


@st.cache_resource(max_entries=4)
def load_classifier(model_dir: str):  # noqa: ANN201
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model, tokenizer, get_dataset(run_config(model_dir)["data"]["dataset"])


@torch.no_grad()
def classify(model_dir: str, claim: str) -> dict:
    """Label probabilities, P(false) and the response time (model already loaded)."""
    model, tokenizer, info = load_classifier(model_dir)
    start = time.perf_counter()
    inputs = tokenizer(claim, truncation=True, max_length=info.max_length, return_tensors="pt")
    probs = torch.softmax(model(**inputs.to(model.device)).logits[0], dim=-1).cpu().numpy()
    return {
        "label_probs": {info.labels[i]: float(p) for i, p in enumerate(probs)},
        "fake_prob": float(probs[info.fake_label_ids].sum()),
        "elapsed": time.perf_counter() - start,
    }


def check(model_dir: str, claim: str) -> dict:
    with st.spinner("Loading the model (first check only)…"):
        load_classifier(model_dir)  # loading is not part of the timed response
    return classify(model_dir, claim)


def verdict(fake_prob: float) -> tuple[str, str, str]:
    return next((word, color, icon) for bound, word, color, icon in VERDICTS if fake_prob <= bound)


# --------------------------------------------------------------------------- rating ladder


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
    rungs["name"] = [
        RUNG_NAMES.get(x, x.replace("-", " ").replace("_", " ").capitalize())
        for x in rungs["label"]
    ]
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


def duration(seconds: float) -> str:
    return f"{seconds * 1000:.0f} ms" if seconds < 1 else f"{seconds:.2f} s"


def verdict_line(result: dict, rungs: pd.DataFrame) -> None:
    word, color, icon = verdict(result["fake_prob"])
    lit = rungs[rungs["lit"]].iloc[0]
    within = result["elapsed"] <= SPEC_S6_SECONDS
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.markdown(f"### :{color}[{icon} {word}]", width="content")
        st.badge(f"{result['fake_prob']:.0%} probability false", color=color)
        st.badge(f"Top label: {lit['name']}", color="gray")
        st.badge(
            f"{duration(result['elapsed'])} encoder · {'within' if within else 'over'} the "
            f"{SPEC_S6_SECONDS * 1000:.0f} ms target (S6)",
            icon=":material/timer:",
            color="green" if within else "orange",
        )


# --------------------------------------------------------------------------- shared pieces


def init_state() -> None:
    st.session_state.setdefault("claim", EXAMPLE_CLAIMS[0])
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("auto_check", False)
    if "view" not in st.session_state:
        linked = st.query_params.get("view")
        st.session_state["view"] = linked if linked in VIEWS else "check"


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


def waiting_panel(
    title: str,
    icon: str,
    body: str,
    owner: str,
    command: str | None = None,
    preview: Callable[[], None] | None = None,
) -> None:
    """A panel that is designed but not built yet: what will appear and who fills it.

    ``preview`` draws the panel's real structure (empty table, the plain claim) under the text.
    """
    with st.container(border=True, height="stretch"):
        st.markdown(f"#### {icon} {title}")
        st.markdown(body)
        if preview:
            preview()
        st.caption(owner)
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

    auto = st.session_state.pop("auto_check", False)
    if model_dir and claim.strip() and (pressed or auto):
        st.session_state["result"] = {"claim": claim, "model": model_dir, **check(model_dir, claim)}
    result = st.session_state["result"]
    if result and (result["claim"] != claim or result["model"] != model_dir):
        result = None  # the claim or model changed since the last check

    st.space("small")
    if model_dir:
        rungs = ladder_rungs(
            load_classifier(model_dir)[2].labels,
            result and result["label_probs"],
            result and result["fake_prob"],
        )
    else:
        rungs = ladder_rungs(("fake", "mixture", "real"))
        rungs["name"] = ["Likely false", "Mixed", "Likely true"]
    show_ladder(rungs)
    if result:
        verdict_line(result, rungs)
    elif models:
        st.markdown("Pick an example claim in the sidebar, or paste one and press **Check**.")
    else:
        no_models_panel()


def compare_view(models: dict[str, Path]) -> None:
    claim = claim_box("claim_compare")
    st.session_state.pop("auto_check", False)  # every comparison below runs on each rerun
    if not models:
        no_models_panel()
        return
    chosen = st.multiselect(
        "Encoders to compare",
        list(models),
        default=list(models)[:2],
        format_func=plate,
        max_selections=3,
        key="compare_models",
    )
    if not (claim.strip() and chosen):
        st.markdown("Choose up to three encoders to see them side by side.")
        return

    results = {m: check(m, claim) for m in chosen}
    columns = st.columns(len(chosen))
    for column, (model_dir, result) in zip(columns, results.items(), strict=False):
        with column, st.container(border=True, height="stretch"):
            st.markdown(f"**{run_config(model_dir)['run']['name']}**")
            st.caption(plate(model_dir))
            rungs = ladder_rungs(
                load_classifier(model_dir)[2].labels, result["label_probs"], result["fake_prob"]
            )
            show_ladder(rungs, height=80, names="lit")
            word, color, icon = verdict(result["fake_prob"])
            st.markdown(f"#### :{color}[{icon} {word}]")
            lit = rungs[rungs["lit"]].iloc[0]
            st.markdown(
                f"{lit['name']} {lit['pct']} · {result['fake_prob']:.0%} false · "
                f"{duration(result['elapsed'])}"
            )
    verdicts = {verdict(r["fake_prob"])[0] for r in results.values()}
    if len(results) > 1 and len(verdicts) > 1:
        st.warning(
            "Compare their ladders: the same words lead them to different ratings.",
            title="The models disagree on this claim",
            icon=":material/call_split:",
        )
    elif len(results) > 1:
        st.markdown(f"All {len(results)} encoders agree: **{verdicts.pop()}**.")


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


def escape_markdown(text: str) -> str:
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|>~$:<])", r"\\\1", text)


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

    isot = {
        run_config(m)["data"].get("variant", "raw"): m
        for m in models
        if run_config(m)["data"]["dataset"] == "isot"
    }
    commands = {
        "raw": "python -m factcheck.train --config configs/encoder/modernbert_isot_clean.yaml "
        "--set data.variant=raw --set run.name=modernbert_isot_raw",
        "clean": "python -m factcheck.train --config configs/encoder/modernbert_isot_clean.yaml",
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
                    "Shortcut study, experiment E7 (M4).",
                    commands[variant],
                )
                continue
            result = check(model_dir, text)
            with st.container(border=True, height="stretch"):
                st.markdown(f"#### {titles[variant]}")
                st.caption(plate(model_dir))
                rungs = ladder_rungs(
                    load_classifier(model_dir)[2].labels, result["label_probs"], result["fake_prob"]
                )
                show_ladder(rungs, height=80)
                word, color, icon = verdict(result["fake_prob"])
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
    try:
        runs = load_runs()
        specs = check_specs(runs)
    except NotImplementedError:
        waiting_panel(
            "Results log and specs check not built yet",
            ":material/table_chart:",
            "Every training run will append its scores to `results/runs.csv`, and the "
            "specifications S1 to S10 are checked from that file.",
            "factcheck/results.py and factcheck/evaluate.py (T04).",
        )
        return
    specs_scorecard(specs)

    st.space("medium")
    st.subheader("Runs on the test split", anchor=False)
    test = runs[runs["split"] == "test"] if "split" in runs else pd.DataFrame()
    if test.empty:
        waiting_panel(
            "No logged runs yet",
            ":material/table_chart:",
            "Every training run appends its scores to `results/runs.csv`; they appear here "
            "and in the specifications above, with no numbers typed by hand.",
            "Logged by factcheck.train (all members).",
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

    best = test.assign(model=test["model_name"].str.split("/").str[-1])
    best = best.groupby(["dataset", "model"], as_index=False)["macro_f1"].max()
    chart = (
        alt.Chart(best)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("macro_f1:Q", title="Best test macro-F1", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("model:N", title=None, sort="-x", axis=alt.Axis(labelLimit=320)),
            color=alt.Color("dataset:N", title="Dataset"),
            yOffset="dataset:N",
            tooltip=["dataset", "model", alt.Tooltip("macro_f1:Q", format=".3f")],
        )
    )
    st.altair_chart(chart, width="stretch", alt="Best test macro-F1 for each model and dataset.")
    st.dataframe(
        test[[c for c in RUN_COLUMNS if c in test]].dropna(axis="columns", how="all"),
        column_config=RUN_COLUMNS,
        hide_index=True,
        alt="Every logged test-split run with its scores.",
    )


# --------------------------------------------------------------------------- page


def sidebar(models: dict[str, Path]) -> None:
    with st.sidebar:
        st.subheader("Example claims", anchor=False)
        for i, example in enumerate(EXAMPLE_CLAIMS):
            st.button(
                example, key=f"example_{i}", on_click=use_example, args=(example,), width="stretch"
            )
        st.space("medium")
        device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        st.caption(f"{len(models)} trained model{'s' if len(models) != 1 else ''} found · {device}")
        st.caption("Triage support for human fact-checkers, not a judge of truth.")


def main() -> None:
    st.set_page_config(
        page_title="Fact-Check Lab", page_icon=":material/fact_check:", layout="wide"
    )
    init_state()
    models = find_models()
    sidebar(models)

    st.title("Fact-Check Lab", anchor=False)
    st.caption(
        "Paste a claim, get an explained verdict from encoders trained on four misinformation datasets."
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


main()
