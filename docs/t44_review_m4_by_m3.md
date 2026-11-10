# T44 code review: M4's code, reviewed by M3

Plan Section 19 makes M3 the reviewer of M4's code. This covers the files on M4's branch
`feature/finalize-eval-deliverables-t33-t42` at `7d869f5` (not yet merged into `main`):
`factcheck/evaluate_cross_dataset.py` (T36), `factcheck/plots_cross_dataset_t41.py` (T41),
`factcheck/shortcut_study_t34.py` (T34), `factcheck/specs_table_t42.py` (T42),
`notebooks/03_interpretation.py` (T33) and the results files they write. M4's earlier
`factcheck/explain.py` (T16) is on `main` and was reviewed with M1's changes to it; it is not
repeated here.

Findings come first, most serious first. Each names the file and line, what goes wrong, how it
was checked and a suggested fix. Fixes are left to M4. **Findings 1 and 2 change numbers that
would go into the slides and the report, so they need fixing before the results freeze (T43)
and before the figures are put on slides.**

## Findings

### 1. The cross-dataset matrix mixes two label conventions, so every COVID transfer cell is wrong (T36, T41)

`factcheck/evaluate_cross_dataset.py:86-118`. The docstring says the binary labels are
"0 = False/Fake, 1 = True/Real". LIAR and PUBHEALTH follow it (mapped by label name), but COVID
does not: `binary_labels.append(int(y))` keeps COVID's own class ids, where **0 = real and
1 = fake** (`get_dataset("covid").labels == ("real", "fake")`). The COVID model's predictions
(`argmax`, line 108) are in the same COVID ids. So:

* COVID -> COVID is consistent (both sides use COVID's ids) and fine;
* every cell where exactly one side is COVID compares "1 = fake" with "1 = true": COVID ->
  LIAR, COVID -> PUBHEALTH, LIAR -> COVID and PUBHEALTH -> COVID are scored with the labels
  reversed.

How it shows: `results/cross_dataset.csv` on the branch has **LIAR -> COVID accuracy 0.218**
on a two-class task with 52% / 48% classes. Below-chance accuracy that far from 0.5 is the
signature of reversed labels (flipped, it would be about 0.78). PUBHEALTH -> COVID (0.403) is
also below chance.

`factcheck/plots_cross_dataset_t41.py:32-35` then patches one of the four cells by hand: if
LIAR -> COVID is below 0.30 it is replaced by the constant `0.7710`. That hides the symptom in
one cell, leaves the other three reversed, and puts a typed-in number in the heatmap. The
write-up in `results/t41_beyond_misinformation.md` ("Models trained on COVID Twitter posts drop
significantly (~0.37 Macro-F1) on longer articles ... overfit to hashtags") is built on the
reversed cells, so its conclusion may be an artefact of the bug.

Suggested fix: use the project's mapping on both sides, `get_dataset(name).binary_label`
(1 = false/fake, 0 = true/real, -1 = left out), which is what `load_splits` already stores in
the `binary_label` column. For predictions, sum the probabilities of `info.fake_label_ids`
(the app and `compute_metrics` do the same). Drop rows with `binary_label == -1`. Then re-run
the 9 cells (inference only, on a CPU) and remove the hand patch and the fallback matrix.

### 2. The T42 figures and specs table show typed-in numbers, not results (T42)

`factcheck/specs_table_t42.py`:

* **Ladder figure (lines 28-75).** `load_dynamic_metrics` looks for `results/metrics.csv` or
  `results/eval_summary.csv` (neither exists) and then `outputs/**/eval_results.json` (the
  trainers write `train_summary.json`, not that name). With none found it falls back to a
  hard-coded table, so the figure always shows those numbers. They do not match the logged
  runs:

  | Model, dataset | Fallback in the figure | `results/runs.csv` (test macro-F1, mean over seeds) |
  |---|---|---|
  | TF-IDF + LR, COVID | 0.892 | 0.953 |
  | TF-IDF + LR, PUBHEALTH | 0.421 | 0.487 |
  | Bi-LSTM, COVID | 0.915 | 0.954 |
  | Bi-LSTM, LIAR | 0.261 | 0.242 |
  | DeBERTa-v3-base, ISOT (clean) | 0.968 | 0.9996 |
  | Bi-LSTM, ISOT (clean) | 0.912 | not run |

* **Shortcut figure (lines 107-119).** Reads `results/isot_shortcut.csv`, which nothing
  writes, and falls back to invented raw / clean accuracies ("Qwen3-8B Zero-Shot 0.985 /
  0.941", "DeBERTa-v3 Base 0.998 / 0.968"). The measured T28 result is RoBERTa raw 1.0000 vs
  clean 0.9996 (M2's contribution-log row), and no zero-shot Qwen3 run exists on raw ISOT.
* **Specs table (lines 155-184).** Every "Achieved" value is typed by hand ("0.974" for S5,
  "18.4 ms", "2.15 s", "5.80 GB", "2.2 hours", "Fully Scripted"). S5 on the logged runs is
  0.9996 to 1.000. The S3 / S4 "binary" values come from the diagonal of the cross-dataset
  file, a different metric from the spec.

Suggested fix: read `results/runs.csv` through `factcheck.results.load_runs` (as
`notebooks/02_results_v1.py` does) and build the specs table with
`factcheck.evaluate.check_specs(runs)`, which already computes S1 to S10 from the logged runs
(T04). Delete the fallback tables: a missing input should stop the script with a clear error,
not draw plausible numbers. The plan's rule (Section 16, and the header of
`02_results_v1.py`) is "no number typed by hand".

### 3. `results/shortcut_study_t34.csv` is committed empty (T34)

`factcheck/shortcut_study_t34.py:12-18` looks for `outputs/explanations/*_seed42_shap.npz`
with a matching `*_on_clean_shap.npz`. On this branch none matched, so the CSV that went into
the commit has no rows (and no header). The T34 numbers for the video therefore do not exist
yet. Suggested fix: fail when no pair is found, and point it at the T28 runs in `runs.csv`
(`roberta_isot_raw` vs `roberta_isot_clean`, `modernbert_covid` vs `modernbert_covid_clean`)
for the accuracy part; keep the SHAP files only for the "which words moved" part.

### 4. `seaborn` is used but not declared (T41, T42)

`plots_cross_dataset_t41.py:13` and `specs_table_t42.py:14` import seaborn, which is in neither
`pyproject.toml` nor `requirements.txt`, so both scripts fail on a fresh install. Add it to the
right extra (`explain` or a new `plots` one) and pin it, or plot with matplotlib only as the
other figure scripts do.

### 5. Hard-coded paths (reusability row of the rubric)

* `evaluate_cross_dataset.py:27-33`: checkpoints under
  `outputs/factcheck_checkpoints_M4_2026-10-28/...` and a staging folder
  `C:\factcheck_models` (Windows only; the script deletes it at the end, line 188).
* `evaluate_cross_dataset.py:178-180`, `shortcut_study_t34.py:8, 40`: relative `results/` and
  `outputs/` paths, which break when run from another folder.
* `specs_table_t42.py:17-19`, `plots_cross_dataset_t41.py:16-18`: their own `BASE_DIR`.

Suggested fix: `from factcheck.paths import OUTPUT_DIR, RESULTS_DIR, FIGURES_DIR` (they also
honour the `FACTCHECK_*` environment variables), and take the checkpoints from the runs'
configs (`outputs/<run>/seed<N>/model`), picking the seed by validation macro-F1.

### 6. Smaller points

* `evaluate_cross_dataset.py:13-22` sets thread environment variables and torch thread counts
  at import time, which slows down anything else that imports it; move them under
  `if __name__ == "__main__":`.
* `evaluate_cross_dataset.py:153`: `max_length=128` for every model; the PUBHEALTH model was
  trained at 384 (claims are short, so little changes, but use `info.max_length`).
* `evaluate_cross_dataset.py:114-115` (PUBHEALTH predictions): "true" only when P(true) > 0.5,
  so mixture and unproven mass counts as false, while the ground truth (line 96) also counts
  mixture and unproven as false. The project's E8 mapping leaves them out instead.
* The new scripts have no tests, and three of them write figures to `outputs/`, which git
  ignores; final figures belong in `results/figures/` so the slides and the freeze see them.
* The Climate-FEVER numbers in `notebooks/03_interpretation.py:83, 129-131` and
  `t41_beyond_misinformation.md` (0.398, 0.429, 0.479) are typed in. They do match
  `runs.csv` (Qwen3-8B zero-shot, RAG from the pool, RAG with the given evidence), and the
  38% arithmetic is right; reading them from `runs.csv` would keep them right if a run changes.

## Checked and fine

* Macro-F1 uses `average="macro"` everywhere it is computed.
* `notebooks/03_interpretation.py`'s Climate-FEVER reading (open retrieval keeps about 38% of
  the gain of the given evidence) matches the logged runs and M2's own write-up.
* The ISOT test split is not used for model choice anywhere in these files.

**Formatting.** With the repo's settings, ruff reports 25 issues in the five files (unused
imports and variables, unsorted imports, `zip()` without `strict=`, one import below code) and
black would reformat all five. `black . && ruff check --fix .` clears most; the README asks for
both before a merge.

## What this means for the freeze

`runs.csv` itself is not changed by this branch, so the freeze (T43) is not blocked by it. But
the T41 heatmap, the T42 ladder and specs table, and the T34 numbers should not go into the
slides until findings 1 to 3 are fixed. The ladder chart can be built straight from `runs.csv`
in a few lines (T21's `02_results_v1.py` already has the loading and the validation-based
selection).
