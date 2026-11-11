"""Results freeze (task T43): check runs.csv, copy it to results/final/ and record its hash.

    python scripts/freeze_results.py --check     # completeness report only, writes nothing
    python scripts/freeze_results.py             # check, then freeze

The freeze copies results/runs.csv to results/final/runs_final.csv and writes
results/final/MANIFEST.md: the SHA-256 of the frozen file, the commit it was frozen at, row
counts per family and dataset, and the warnings below. It then prints the git commands that
tag the commit (it does not run them):

    git add results/final && git commit -m "T43: results freeze"
    git tag -a v1.0-results -m "Final results for submission" && git push origin v1.0-results

After the freeze, any change to a reported number needs the whole team's agreement (plan
Section 20). The check flags, without blocking:

* configs trained with seeds that have fewer than 3 seeds (single-seed families such as the
  deterministic TF-IDF baselines, kNN and the LLM runs are expected to have one);
* runs that predict a single class on test (a collapsed training run);
* rows logged from a "-dirty" working tree (uncommitted code changes at the time);
* duplicate (run, seed, split) rows, which would double count.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from factcheck.paths import PROJECT_ROOT, RESULTS_DIR, RUNS_CSV  # noqa: E402

FINAL_DIR = RESULTS_DIR / "final"
FROZEN = FINAL_DIR / "runs_final.csv"
MANIFEST = FINAL_DIR / "MANIFEST.md"
SINGLE_SEED_FAMILIES = {"classic", "knn", "llm", "ensemble"}  # one run by design


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    out = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True)
    return out.stdout.strip()


def check(runs: pd.DataFrame) -> list[str]:
    """Human-readable warnings about the results table."""
    warnings = []
    dupes = runs.duplicated(["run_name", "seed", "split"]).sum()
    if dupes:
        warnings.append(f"{dupes} duplicate (run, seed, split) rows")
    test = runs[(runs["split"] == "test") & runs["subset"].isna()]
    seeds = test.groupby("run_name").agg(seeds=("seed", "nunique"), family=("family", "first"))
    few = seeds[(seeds["seeds"] < 3) & ~seeds["family"].isin(SINGLE_SEED_FAMILIES)]
    few = few[~few.index.str.contains("lora")]  # LoRA runs: one seed by decision (plan v4)
    if len(few):
        warnings.append(f"{len(few)} trained configs with fewer than 3 seeds: "
                        + ", ".join(few.index[:12]))  # fmt: skip
    one_class = test[test["confusion_matrix"].map(
        lambda m: int((np.array(json.loads(m)).sum(axis=0) > 0).sum()) == 1)]  # fmt: skip
    if len(one_class):
        names = ", ".join(f"{r.run_name} seed {r.seed}" for r in one_class.itertuples())
        warnings.append(f"{len(one_class)} test runs predict a single class: {names}")
    dirty = runs["git_commit"].fillna("").str.endswith("-dirty")
    if dirty.any():
        warnings.append(f"{dirty.sum()} of {len(runs)} rows were logged with uncommitted code "
                        f"changes (-dirty), from {runs.loc[dirty, 'run_name'].nunique()} runs")  # fmt: skip
    missing = runs["git_commit"].isna() | (runs["git_commit"] == "")
    if missing.any():
        warnings.append(f"{missing.sum()} rows have no git commit")
    return warnings


def summary_table(runs: pd.DataFrame) -> str:
    test = runs[runs["split"] == "test"]
    counts = test.groupby(["family", "dataset"])["run_name"].nunique().unstack(fill_value=0)
    header = "| family | " + " | ".join(counts.columns) + " |"
    rule = "|---" * (len(counts.columns) + 1) + "|"
    rows = [f"| {fam} | " + " | ".join(str(v) for v in r) + " |" for fam, r in counts.iterrows()]
    return "\n".join([header, rule, *rows])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="report only, freeze nothing")
    args = parser.parse_args()

    runs = pd.read_csv(RUNS_CSV)
    warnings = check(runs)
    print(f"{RUNS_CSV}: {len(runs)} rows, {runs['run_name'].nunique()} runs")
    for w in warnings:
        print(f"  WARNING {w}")
    if args.check:
        return

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    # Unix line endings, so the hash is the same on every machine (Windows writes CRLF;
    # .gitattributes keeps results/final/*.csv LF on checkout).
    FROZEN.write_bytes(RUNS_CSV.read_bytes().replace(b"\r\n", b"\n"))
    commit = git("rev-parse", "--short", "HEAD")
    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    lines = [
        "# Results freeze (T43)",
        "",
        f"* Frozen file: `results/final/runs_final.csv` ({len(runs)} rows, "
        f"{runs['run_name'].nunique()} runs)",
        f"* SHA-256: `{sha256(FROZEN)}`",
        f"* Frozen on {datetime.now():%Y-%m-%d %H:%M} at commit `{commit}`"
        + (" (with uncommitted changes)" if dirty else ""),
        "",
        'Check it with `python -c "import hashlib; print(hashlib.sha256(open('
        "'results/final/runs_final.csv','rb').read()).hexdigest())\"`. After the freeze, any "
        "change to a reported number needs the whole team's agreement.",
        "",
        "## Runs per family and dataset (test split)",
        "",
        summary_table(runs),
        "",
        "## Warnings at freeze time",
        "",
        *([f"* {w}" for w in warnings] or ["* none"]),
        "",
    ]
    MANIFEST.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {FROZEN} and {MANIFEST}\nSHA-256 {sha256(FROZEN)}\n\nThen tag it:")
    print('  git add results/final && git commit -m "T43: results freeze"')
    print('  git tag -a v1.0-results -m "Final results for submission"')
    print("  git push origin main v1.0-results")


if __name__ == "__main__":
    main()
