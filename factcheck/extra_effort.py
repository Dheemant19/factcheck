"""The extra-effort table and chart (owner M3, T32; plan Section 8.5, experiment E5).

    python -m factcheck.extra_effort

How much does each improvement add? Every step is measured against the run it changes, on
the test macro-F1 averaged over seeds, from ``results/runs.csv`` alone:

    Optuna tuning              tuned DeBERTa-v3-base (T18)       vs  default DeBERTa-v3-base (T17)
    + speaker metadata         tuned + metadata (T29, LIAR)       vs  tuned
    + evidence sentences       tuned + evidence (T29, PUBHEALTH)  vs  tuned
    + class weights            tuned + class weights (T29)        vs  tuned
    domain model instead       COVID-Twitter-BERT / SciBERT (T19) vs  tuned
    top-3 ensemble instead     ensemble_top3_<dataset> (T31)      vs  tuned

T29 switched each extra on by itself, so the steps are NOT stacked: a cumulative waterfall
would claim combinations nobody trained. The chart shows each gain from the same reference
instead. Error bars are one standard error of the difference of the two means
(``sqrt(sd_a^2 / n_a + sd_b^2 / n_b)`` over seeds). If ``results/significance.csv`` exists
(T31), its Holm-corrected bootstrap p-value is added to each step it covers.

Writes ``results/e5_extra_effort.csv``, ``results/e5_extra_effort.md`` and
``results/figures/e5_extra_effort.png``. ISOT is left out: every encoder already scores
about 1.000 on it, so there is nothing to add.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from factcheck.paths import FIGURES_DIR, RESULTS_DIR  # noqa: E402
from factcheck.results import load_runs  # noqa: E402

DATASETS = {"covid": "COVID", "liar": "LIAR", "pubhealth": "PUBHEALTH"}
DOMAIN = {"covid": "covid_twitter_bert_covid", "pubhealth": "scibert_pubhealth"}
TABLE_CSV = RESULTS_DIR / "e5_extra_effort.csv"
TABLE_MD = RESULTS_DIR / "e5_extra_effort.md"
FIGURE = FIGURES_DIR / "e5_extra_effort.png"
SIGNIFICANCE_CSV = RESULTS_DIR / "significance.csv"

# Same palette as notebooks/02_results_v1.py. Gain and loss are blue and orange: checked
# for colour-blind separation (OKLab dE >= 24 under protan, deutan and tritan simulation).
INK, MUTED, GRID, SURFACE = "#13294b", "#52514e", "#e1e0d9", "#fcfcfb"
GAIN, LOSS = "#2a78d6", "#eb6834"


def steps(dataset: str) -> list[tuple[str, str | None, str]]:
    """(label, run, reference run) in display order; run None means not applicable."""
    base, tuned = f"deberta_v3_base_{dataset}", f"deberta_v3_base_{dataset}_tuned"
    return [
        ("Optuna tuning", tuned, base),
        ("+ speaker metadata", f"{tuned}_meta" if dataset == "liar" else None, tuned),
        ("+ evidence sentences", f"{tuned}_evidence" if dataset == "pubhealth" else None, tuned),
        ("+ class weights", f"{tuned}_cw", tuned),
        ("domain model instead", DOMAIN.get(dataset), tuned),
        ("top-3 ensemble instead", f"ensemble_top3_{dataset}", tuned),
    ]


def test_scores(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean, std and seed count of test macro-F1 per (dataset, run)."""
    test = runs[(runs["split"] == "test") & runs["subset"].isna()]
    return test.groupby(["dataset", "run_name"])["macro_f1"].agg(["mean", "std", "count"])


def significance() -> dict[tuple[str, str], float]:
    if not SIGNIFICANCE_CSV.exists():
        return {}
    sig = pd.read_csv(SIGNIFICANCE_CSV)
    out = {}
    for row in sig.itertuples():
        out[row.model_a, row.model_b] = out[row.model_b, row.model_a] = row.p_bootstrap_holm
    return out


def build_table(runs: pd.DataFrame) -> pd.DataFrame:
    scores, p_values = test_scores(runs), significance()
    rows = []
    for dataset in DATASETS:
        for label, run, ref in steps(dataset):
            row = {"dataset": dataset, "step": label, "run": run, "reference": ref}
            have = run is not None and (dataset, run) in scores.index
            if have and (dataset, ref) in scores.index:
                a, b = scores.loc[(dataset, run)], scores.loc[(dataset, ref)]
                se = np.sqrt(np.nan_to_num(a["std"]) ** 2 / a["count"]
                             + np.nan_to_num(b["std"]) ** 2 / b["count"])  # fmt: skip
                row |= {
                    "macro_f1": round(a["mean"], 4),
                    "macro_f1_sd": round(a["std"], 4) if a["count"] > 1 else None,
                    "seeds": int(a["count"]),
                    "reference_macro_f1": round(b["mean"], 4),
                    "gain": round(a["mean"] - b["mean"], 4),
                    "gain_se": round(se, 4),
                    "p_holm": p_values.get((run, ref)),
                    "status": "measured",
                }
            else:
                row["status"] = "not applicable" if run is None else "not run yet"
            rows.append(row)
    return pd.DataFrame(rows)


def write_markdown(table: pd.DataFrame) -> None:
    lines = [
        "# E5: what each extra adds (test macro-F1, mean over seeds)",
        "",
        "Each step is measured against the run it changes (tuning: against the default",
        "DeBERTa-v3-base; every other step: against the tuned one). The steps were trained one",
        "at a time, so the gains do not add up. ± is one standard error of the difference.",
        "Written by `python -m factcheck.extra_effort`.",
        "",
        "| Dataset | Step | Macro-F1 | Reference | Gain | p (Holm) |",
        "|---|---|---|---|---|---|",
    ]
    for r in table.itertuples():
        if r.status != "measured":
            lines.append(f"| {DATASETS[r.dataset]} | {r.step} | {r.status} | | | |")
            continue
        p = "" if pd.isna(r.p_holm) else f"{r.p_holm:.3f}"
        lines.append(
            f"| {DATASETS[r.dataset]} | {r.step} | {r.macro_f1:.3f} | {r.reference_macro_f1:.3f} "
            f"| {r.gain:+.3f} ± {r.gain_se:.3f} | {p} |"
        )
    TABLE_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(table: pd.DataFrame) -> None:
    labels = [label for label, _, _ in steps("covid")]
    y = np.arange(len(labels))[::-1]
    measured = table[table["status"] == "measured"]
    reach = (measured["gain"].abs() + measured["gain_se"]).max() * 1.9  # room for the labels
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(15, 4.8), facecolor=SURFACE, sharey=True)
    for ax, (dataset, title) in zip(axes, DATASETS.items(), strict=True):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#c3c2b7")
        ax.tick_params(colors=MUTED, labelsize=10.5, length=0)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.axvline(0, color=INK, linewidth=1.0)
        part = table[table["dataset"] == dataset].set_index("step").loc[labels]
        for yi, (_, r) in zip(y, part.iterrows(), strict=True):
            if r["status"] != "measured":
                ax.text(0, yi, f"  {r['status']}", va="center", fontsize=9.5, color=MUTED)
                continue
            color = GAIN if r["gain"] >= 0 else LOSS
            ax.barh(yi, r["gain"], height=0.56, color=color)
            ax.errorbar(r["gain"], yi, xerr=r["gain_se"], fmt="none", ecolor=INK, elinewidth=1)
            end = r["gain"] + np.sign(r["gain"] or 1) * r["gain_se"]
            note = f"{r['gain']:+.3f}  ({r['macro_f1']:.3f})"
            if r["seeds"] < 3:
                note += " *"
            ax.text(end + np.sign(r["gain"] or 1) * reach * 0.03, yi, note, va="center",
                    ha="left" if r["gain"] >= 0 else "right", fontsize=9.5, color=INK)  # fmt: skip
        tuned = part.loc["+ class weights", "reference_macro_f1"]
        ax.set_title(title, fontsize=13, color=INK, loc="left", fontweight="bold", pad=22)
        ax.text(0, 1.02, f"reference: tuned DeBERTa-v3-base, {tuned:.3f}", transform=ax.transAxes,
                fontsize=10, color=MUTED)  # fmt: skip
        ax.set_xlim(-reach, reach)
        ax.set_ylim(-0.7, len(labels) - 0.3)  # keep the bottom row off the axis line
        ax.set_yticks(y, labels, fontsize=10.5, color=INK)
        ax.set_xlabel("Change in test macro-F1 (± 1 s.e.)", fontsize=10.5, color=MUTED)
    fig.text(
        0.01, 0.01,
        "Each step is measured on its own (tuning against the default model, the rest against "
        "the tuned one); the gains do not add up. Score after the change in brackets. "
        "* fewer than 3 seeds.",
        fontsize=9, color=MUTED,
    )  # fmt: skip
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    table = build_table(load_runs())
    table.to_csv(TABLE_CSV, index=False)
    write_markdown(table)
    plot(table)
    shown = table[table["status"] == "measured"]
    print(shown[["dataset", "step", "macro_f1", "reference_macro_f1", "gain", "gain_se",
                 "seeds", "p_holm"]].to_string(index=False))  # fmt: skip
    print(f"\nwrote {TABLE_CSV}, {TABLE_MD} and {FIGURE}")


if __name__ == "__main__":
    main()
