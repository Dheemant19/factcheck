# T44 code review: M1's code, reviewed by M2

Plan Section 19 pairs M1 and M2 as each other's reviewers. This covers M1's files on main at
`df2370d`: `factcheck/evaluate.py`, `results.py`, `train.py`, `tune.py`, `models/classic.py`,
`serve.py`, `style.py`, `scripts/transfer_climate_fever.py`, the logic parts of `app/app.py`
and `notebooks/02_results_v1.py`. The parts of `encoder.py`, `llm.py` and `lstm.py` that M1
wrote are left out, because M2 has worked in them throughout (T20, T26, T27, T55).

Findings come first, most serious first, then what was checked and found fine. Each finding
names the file and line, what goes wrong, how it was checked, and a suggested fix. Fixes are
left to M1. Numbers 1 and 2 change what the app tells users and what the report says, so they
are worth fixing before results are frozen (T43).

## Findings

### 1. The app says "probability that the claim is true" for what is really "probability it is not false" (PUBHEALTH, Climate-FEVER, LIAR)

`app/app.py:144-148, 404-413`. The verdict and its sentence use only `fake_prob`, the summed
probability of the false labels. For "Likely true" the app says it is `share(1 - fake_prob)`
sure the claim is true. But `1 - fake_prob` also contains every label that is neither false nor
true:

| Dataset | Counted as false | `1 - fake_prob` also includes |
|---|---|---|
| PUBHEALTH | false | mixture, unproven |
| Climate-FEVER | refutes | not_enough_info, disputed |
| LIAR | pants-fire, false, barely-true | half-true |

Example: a PUBHEALTH claim scored false 0.20, mixture 0.50, unproven 0.20, true 0.10 gets
"**Likely true**, with an **80%** probability that the claim is true, most likely rated
**Mixture**". The model gives "true" 10%. This is common, not an edge case. On the saved PUBHEALTH test
predictions of two app models, the rule gives "Likely true" for 62-63% of the 1,233 claims, and
for 29-32% of those the model's own P(true) is below 0.5 (gap up to 0.75 for
`deberta_v3_base_pubhealth_tuned_evidence` seed 42 and 0.90 for `qwen3_8b_qlora_pubhealth`).

Suggested fix: compute `true_prob` from `info.real_labels` alongside `fake_prob`. Call a claim
"Likely true" only when `true_prob` passes the threshold, and treat the rest as mixed /
uncertain. Or keep the current verdict but word the sentence as "a 20% probability that the
claim is false".

### 2. The TF-IDF SVM and GloVe SVM baselines are scored with a different model from the one the grid chose (report numbers understated)

`factcheck/models/classic.py:298-303, 347`. `select` picks C and the class weights by the
plain `LinearSVC`'s validation macro-F1. Then `calibrated` refits a `CalibratedClassifierCV`
(sigmoid, 3 folds), and that is the model scored and logged. Sigmoid calibration undoes the
`class_weight="balanced"` shift that won the grid. So the logged model predicts the majority
classes again, and its macro-F1 is well below what the grid measured. I re-ran both baselines
on gpu55 with logging switched off (runs.csv untouched). The test scores match the logged rows
exactly:

| Run | Grid's best validation macro-F1 (chosen setting) | Logged model, validation | Logged model, test |
|---|---|---|---|
| tfidf_svm_pubhealth | 0.467 (C 0.1, balanced) | 0.383 | 0.400 (= runs.csv) |
| tfidf_svm_liar | 0.248 (C 3.0, balanced) | 0.205 | 0.213 (= runs.csv) |

So the "TF-IDF + SVM" row of the results table is about 0.04 to 0.08 lower than the SVM
itself, which makes the traditional rung look weaker than it is. The GloVe SVM runs go through
the same path and are probably affected too; I did not re-run them.

Suggested fix: select and score the same model. Either put `CalibratedClassifierCV` inside the
grid, so selection uses the calibrated validation macro-F1, or keep the margins for the class
decision and use the calibrated probabilities only for ROC-AUC and ECE. Then re-run the SVM
rows.

### 3. A crash while runs.csv is being written can wipe the whole file

`factcheck/results.py:68-76`. `append_runs` takes the lock, then writes the full table straight
over `results/runs.csv` with `to_csv`. If the process is killed partway through the write (out
of memory, Ctrl-C, a node reboot, `timeout`), the file is left truncated, and with it every
team member's rows. The lock does not help, because the truncation is done by the lock holder.
The app's `load_runs` also reads without the lock, so it can read a half-written file while a
cluster run is logging.

Suggested fix: write to a temporary file in the same folder, then `os.replace` it over
`runs.csv` while still holding the lock. The replace is atomic, so readers see either the old
or the new file, never a partial one.

### 4. The app picks each run's seed by test macro-F1

`app/app.py:196-201, 249-256`. `find_models` sorts by `test_f1` and keeps the best seed of each
run. The docstring says this is only for choosing the demo model, which is fair. But any figure
quoted from the app's model (latency, an example verdict, T40's measurements) then comes from a
seed chosen on the test set. The results notebook already selects by validation
(`notebooks/02_results_v1.py:98`), and the app should do the same.

Suggested fix: read the `validation` rows in `logged_test_f1`, and rename it to match.

### 5. A trial that runs out of memory skips the memory clean-up (plausible, not reproduced)

`factcheck/tune.py:109-117`. `gc.collect()` and `torch.cuda.empty_cache()` run after a finished
or pruned trial. A `torch.cuda.OutOfMemoryError` leaves `objective` before them, and
`study.optimize(catch=...)` moves on to the next trial with the failed trial's tensors possibly
still held by the exception's traceback. The next trial can then fail on memory it would
otherwise have had. I have not checked whether this happened in T18, so treat it as a likely
issue rather than an observed one.

Suggested fix: move the clean-up into a `finally:` around `train_encoder`.

### 6. Smaller issues

* **The fake-class precision and recall use a different rule from ROC-AUC and from the app**
  (`factcheck/evaluate.py:99`). A row counts as flagged when its single most likely label is a
  false label. ROC-AUC and PR-AUC, and the app's verdict, use the summed P(false). On LIAR a
  claim at P(false) 0.6, split as 0.2 over each of the three false labels, can have "half-true"
  as its top label and so count as not flagged. Either way is defensible, but the report should
  say which rule is used, or `flagged` should be `p_fake >= 0.5` so it matches the AUCs and the
  app.
* **Calibrated SVMs report three times their size** (`classic.py:272-273`).
  `count_params` adds up the three fold models inside `CalibratedClassifierCV`. The GloVe SVM on
  COVID logs 903 parameters for a 301-weight model. Count one fold model, or note it in the
  cost plot.
* **`is_servable` lets any TF-IDF baseline through without the metadata check**
  (`serve.py:81-82`). The early `return True` for baselines skips the `use_metadata` test that
  encoders get. No baseline uses metadata today, so nothing is wrong yet. Moving the metadata
  test above the baseline branch keeps it from going wrong later.
* **`explain_linear` on a style-stripped baseline highlights the raw text, but the features come
  from the stripped text** (`serve.py:424` vs `434`). An n-gram that only exists after stripping
  (`#covid_vaccine` becomes "covid vaccine", and so does a word after a removed "RT") still
  appears in the ranked `features`, but its push never reaches a highlighted word. Matching
  pieces against `prepare([claim], ...)` would fix it. This is cosmetic.
* **Re-logging a run moves its rows to the end of runs.csv** (`results.py:73-75`). Git then
  shows untouched rows as deleted and re-added. Two `roberta_large_pubhealth` rows looked
  changed in the T55 diff, but only their position had changed. Sorting by `KEY` before
  writing, or replacing rows in place, would keep diffs to the rows that really changed.
* **Out-of-date docstring** (`train.py:7`): "llm (T07, prompted scoring, no training yet)". The
  llm family now trains as well (`method: qlora`, T27). M2 can fix this if you prefer, as it
  describes M2's feature.
* **Cost plot labels the best test point** (`notebooks/02_results_v1.py:214`). Only the
  annotation is chosen this way, and the plot itself shows every run. Labelling the run with the
  best validation score would match the table.

## Checked and fine

* `compute_metrics` and `expected_calibration_error`: the binary grouping leaves out the
  in-between labels, `labels=classes` keeps missing classes in macro-F1, and the top bin
  includes confidence 1.0.
* `measure_latency_ms`, `git_commit` (runs.csv left out of the dirty check), `check_specs`
  (debug runs on data subsets left out, the worst run used for "every run" specs).
* `append_runs` replaces rows by (run, seed, split) under a file lock. Only the write itself is
  the problem (finding 3).
* `tune.py`: the search space, the trial counting across workers (running trials included), and
  the export of the best trial as a config.
* `classic.py`: TF-IDF is fitted on the training split only (no leakage), and `top_words` and
  `class_weights_of` handle the two-label case correctly.
* `serve.py`: the evidence join matches training (`" ".join`, as in `factcheck.inputs`). fp16
  serving is safe on DeBERTa-v3, which had fp16 problems in training: on 300 PUBHEALTH test
  claims with evidence (`deberta_v3_base_pubhealth_tuned_evidence`, seed 42), fp16 and fp32
  differ by at most 0.005 in any probability, there are no NaNs, and no verdict changes.
* `transfer_climate_fever.py`: the label map and the column reordering are correct, and the
  evidence runs get each claim's own 5 sentences.
* `style.py`: both doctests pass (`python -m doctest factcheck/style.py`), and the order of the substitutions is sensible (links and
  mentions before symbols, trailing hashtags before single "#").

## Not covered

The layout and widget code in `app/app.py` (lines 473 onwards), the setup and download scripts,
and `scripts/check_env.py` / `smoke_test.py`. These were read only for logic that affects
numbers.
