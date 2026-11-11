# Ensembles, significance tests and the extra-effort chart

Notes for M3 on T31 (ensemble and significance tests) and T32 (extra-effort waterfall), for
M4 (figures, T42) and M5 (slides). Numbers below come from `results/runs.csv` as of
6 Nov 2026.

## In short

* **T32 is done**: `python -m factcheck.extra_effort` builds the E5 table and chart from
  `runs.csv` alone. The biggest gains are **evidence on PUBHEALTH (+0.059 macro-F1)**,
  class weights on PUBHEALTH (+0.032), speaker metadata on LIAR (+0.021) and the
  COVID-Twitter-BERT domain model on COVID (+0.008).
* **T31 ran on 11 Nov** with `--skip-ensemble` (significance tests only, so the frozen
  `runs.csv` is not changed; plan v4 makes the ensemble itself optional), on the per-row
  predictions gathered from the laptop, gpu21 and gpu55 (they are not in git). Of 22
  comparisons, only DeBERTa-v3-large LoRA vs full fine-tuning on PUBHEALTH (−0.096) is
  significant on macro-F1; see `docs/m3_interpretation.md`, Section 6. To re-run, put the
  predictions in `results/predictions/` and run `bash jobs/t31_ensemble.sh` (about a minute,
  no GPU).

## 1. The ensemble (T31)

| Step | What it does | Why |
|---|---|---|
| Average seeds | One probability matrix per model per split | A model is judged as a whole, not by its luckiest seed |
| Pick 3 on **validation** macro-F1 | Best first; each further slot takes the best remaining model, unless one with a backbone not yet used is within 0.01 | Choosing on test would make the ensemble look better than it is. Ensembles gain most from models that make different errors, so three DeBERTa-v3-base variants are avoided when a different model is nearly as good |
| Soft vote | Average the three probability matrices, take the most likely class | Simple and has no parameters to tune on test |
| Optional `--temperature` | Each member is calibrated on validation first | LLM label probabilities can be over-confident and would dominate the average |

Only runs on the reported variant (ISOT: cleaned text), with no `subset`, and with
predictions for the full validation and test splits are candidates. ISOT LLM runs that score
only the 2,000-article subset are left out of the full-test ensemble.

## 2. The significance tests (T31)

For each planned comparison, with each model's seeds averaged:

* **Paired bootstrap**: resample the test set with replacement 10,000 times and compute
  macro-F1(A) − macro-F1(B) on the same rows each time. The 2.5th and 97.5th percentiles give
  a 95% interval; if it excludes 0, the difference is unlikely to be test-set luck.
* **McNemar's test**: on the rows where exactly one model is right; exact binomial when fewer
  than 25 such rows.
* **Holm correction** over all comparisons, because with about 25 tests some would pass by
  chance. `significant` in `results/significance.csv` means the Holm-adjusted bootstrap
  p-value is below 0.05.

Comparisons: ensemble vs best single model; best encoder vs BERT-base; best encoder vs best
Qwen3-8B mode; Optuna-tuned vs default DeBERTa-v3-base; each extra (metadata, evidence, class
weights, domain model) vs the tuned DeBERTa-v3-base; best LoRA config vs full fine-tuning for
roberta-large and deberta-v3-large.

## 3. The extra-effort chart (T32)

T29 switched each extra on **by itself** on the T18-tuned DeBERTa-v3-base. A cumulative
waterfall (each bar floating from the previous total) would show combinations that were never
trained, so the chart shows each step's gain against the run it changes instead: tuning
against the default DeBERTa-v3-base, every other step against the tuned one. Error bars are
one standard error of the difference over seeds; T31's bootstrap adds the test-sampling
uncertainty.

| Dataset | Step | Test macro-F1 | Gain |
|---|---|---|---|
| COVID | Optuna tuning | 0.972 | +0.001 ± 0.002 |
| COVID | + class weights | 0.971 | −0.001 ± 0.000 |
| COVID | COVID-Twitter-BERT instead | 0.981 | **+0.008 ± 0.001** |
| LIAR | Optuna tuning | 0.275 | −0.003 ± 0.012 |
| LIAR | + speaker metadata | 0.296 | **+0.021 ± 0.010** |
| LIAR | + class weights | 0.277 | +0.002 ± 0.009 |
| PUBHEALTH | Optuna tuning | 0.503 | +0.039 ± 0.060 |
| PUBHEALTH | + evidence sentences | 0.562 | **+0.059 ± 0.017** |
| PUBHEALTH | + class weights | 0.535 | **+0.032 ± 0.010** |
| PUBHEALTH | SciBERT instead | 0.509 | +0.005 ± 0.013 |

What it says:

* **Evidence is the biggest single gain**, on the dataset whose claims need outside facts.
  It still leaves PUBHEALTH (0.562) below the paper's 0.705, which used evidence too.
* **Class weights help where a class is rare** (PUBHEALTH unproven 3%) and do nothing on
  balanced COVID.
* **Tuning barely matters**, except on PUBHEALTH, where the gain is mostly noise: the default
  DeBERTa-v3-base's seed 42 run is weak (0.348 against 0.50 and 0.54 for the other seeds), which
  makes the error bar ±0.060. Re-running that seed would firm this step up.
* COVID is near its ceiling (0.97), so every change there is small.
* ISOT is left out: every encoder scores about 1.000, so there is nothing to add. Part of
  that is a shortcut T13 does not remove yet: the capitalisation of the headline alone
  predicts the label with 97.8% validation accuracy (Reuters writes sentence case, the fake
  sites Title Case).

Colours: gain blue, loss orange, from the team palette; checked for colour-blind separation
(OKLab ΔE ≥ 24 under protan, deutan and tritan simulation) and 3:1 contrast. The direction of
the bar and the sign of the label carry the same information, so colour is never the only cue.

## 4. Running it

```bash
bash jobs/t31_ensemble.sh            # where results/predictions/ is complete, then commit:
git add results/significance.csv results/ensemble_candidates.csv \
        results/e5_extra_effort.* results/figures/e5_extra_effort.png
pytest tests/test_stats.py tests/test_ensemble.py tests/test_extra_effort.py
```

The statistics are checked against scikit-learn (macro-F1), a hand-worked Holm example and
the exact binomial McNemar case; the ensemble job is tested end to end on synthetic
predictions.
