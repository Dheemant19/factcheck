# Comprehensive Interpretation of Results (Tasks T33 to T42)

All numbers are test-split scores from the frozen `results/final/runs_final.csv` (`v1.0-results`),
averaged over each run's seeds (3 for encoders and RNNs; 1 for TF-IDF, GloVe and LLM runs), unless
a line says otherwise. Figures: `python -m factcheck.specs_table_t42` (ladder, shortcut chart, specs
table) and `python -m factcheck.plots_cross_dataset_t41`. Numbers corrected against the frozen
results on 13 Nov.

## 1. Model Ladder & Benchmark Performance (T42)
Each step up the ladder adds little where the dataset's own markers already decide the label:
- **COVID-19 (binary):** TF-IDF + LR 0.953, Bi-LSTM 0.954, DeBERTa-v3-base 0.971 and the domain-adapted
  COVID-Twitter-BERT 0.980 macro-F1 (0.981 weighted F1, spec S2).
- **ISOT clean (binary):** even TF-IDF + LR scores 0.988 on the cleaned text; DeBERTa-v3-base 0.9996
  and the best runs 1.000. No Bi-LSTM was trained on ISOT.
- **PUBHEALTH (4 classes):** the traditional and recurrent baselines score 0.39 to 0.49 (TF-IDF + LR
  0.487 is the best of them). DeBERTa-v3-base scores 0.465 by default and 0.504 tuned, and 0.562 with
  evidence sentences, the best run.
- **LIAR (6 classes):** baselines score 0.195 to 0.242 macro-F1; DeBERTa-v3-base 0.278, and 0.296
  with speaker and context metadata. The best 6-class accuracy is 0.299 (spec S3), above the 0.274 of
  Wang (2017).
- **Recurrent cells (M2's T55):** the plain RNN's gradients are the least stable: mean norm 1.5 to
  2.8 against 0.6 to 1.0 for GRU and LSTM, with spikes to 50 without clipping (and up to 326 before
  clipping, which cut 80% to 99% of its updates). The gated cells stay at most 13.

## 2. Cost-Performance Trade-off & Latency (T42)
Latency per claim as logged when scoring the ISOT test set (long articles) on the cluster GPUs
(RTX 3090; Qwen3-8B on an RTX A5000; TF-IDF on the CPU). The app's own latency on the laptop is in
Section 8 (S6, S7).

| Model Architecture | Latency (ms/claim) | Clean ISOT Macro-F1 | Deployment Profile |
| :--- | :--- | :--- | :--- |
| **TF-IDF + LR** | 0.7 ms | 0.988 | CPU only, explanation exact |
| **DistilBERT** | 5.1 ms | 0.998 | Smallest encoder |
| **RoBERTa-base** | 8.5 ms | 0.9996 | Base encoder |
| **DeBERTa-v3-base** | 21.5 ms | 0.9996 | Base encoder |
| **DeBERTa-v3-large** | 45.7 ms | 0.9999 | Largest encoder |
| **Qwen3-8B QLoRA** | 210.6 ms | 1.000 (1 seed) | Generative LLM, 4-bit |

On ISOT every model is within 0.012 of perfect, so the extra cost buys almost nothing here.

## 3. Extra-Effort Steps (T32 chart, from `results/e5_extra_effort.md`)
T29 switched each extra on **by itself** on the tuned DeBERTa-v3-base, so each gain is against the
run it changes (tuning against the default model). The steps were never combined, so there is no
cumulative total, and the top-3 ensemble was not run.

| Dataset | Step | Macro-F1 | Reference | Gain (± SE over seeds) |
| :--- | :--- | :--- | :--- | :--- |
| COVID | Optuna tuning | 0.972 | 0.971 | +0.001 ± 0.002 |
| COVID | Domain model (COVID-Twitter-BERT) | 0.981 | 0.972 | +0.008 ± 0.001 |
| LIAR | Optuna tuning | 0.275 | 0.278 | −0.003 ± 0.012 |
| LIAR | + speaker metadata | 0.296 | 0.275 | +0.021 ± 0.010 |
| PUBHEALTH | Optuna tuning | 0.503 | 0.465 | +0.039 ± 0.060 |
| PUBHEALTH | + evidence sentences | 0.562 | 0.503 | **+0.059 ± 0.017** (largest) |
| PUBHEALTH | + class weights | 0.535 | 0.503 | +0.032 ± 0.010 |
| PUBHEALTH | Domain model (SciBERT) | 0.509 | 0.503 | +0.005 ± 0.013 |

After the significance tests (T31, 22 comparisons, Holm-corrected) none of these gains is beyond
test-set noise on macro-F1. Evidence on PUBHEALTH does get significantly more claims right
(McNemar). See `docs/m3_interpretation.md`, Section 6.

## 4. Shortcut Analysis & Feature Attribution (T33, T34)
Exposing and removing dataset artifacts, and what is left after:
- **Dataset fingerprints (ISOT raw):** raw ISOT text carries publisher shortcuts: the Reuters
  dateline (`"reuters"`), media tags (`"via"`, `"video"`, `"image"`) and Twitter handles (`"@"`).
- **Effect of cleaning (16 named rules):** TF-IDF + LR test accuracy falls only from 0.994 to 0.988;
  RoBERTa-base from 1.000 to 0.9996; ModernBERT on COVID (links and handles masked) goes from 0.966 to 0.968
  (`results/shortcut_study_t34.csv`).
- **Why cleaning changes so little:** the cleaned real and fake articles still differ in topic (LDA
  topics alone predict the label at 0.885 macro-F1, M3's T56) and in headline style (capitalisation
  alone gives 97.9% accuracy). A near-perfect cleaned score therefore shows the models can still tell
  the two sources apart, not that they check facts.
- **SHAP attribution:** raw TF-IDF relies on tokens like `"via"` and `"video"`; cleaned TF-IDF shifts to
  political keywords (`"gop"`, `"obama"`); transformer SHAP picks reporting verbs (`"said"`, `"urged"`,
  Reuters house style) and named entities (`"trump"`, `"tillerson"`, `"u.s"`).

## 5. Model Calibration & Expected Calibration Error (ECE) (T33)
ECE on the full test sets (`ece` in `runs_final.csv`):
- **Well calibrated:** the binary datasets. ISOT runs are at most 0.029 (RoBERTa-base clean and raw both
  0.000); on COVID, COVID-Twitter-BERT 0.015 and DistilBERT 0.020.
- **Miscalibrated:** the multi-class datasets, where confident models meet hard labels. On PUBHEALTH,
  DistilBERT 0.119, the Bi-LSTM 0.162 and tuned DeBERTa-v3-base 0.162; the median run is 0.084 on
  PUBHEALTH and 0.075 on LIAR. Prompted Qwen3-8B is the worst (0.591 on PUBHEALTH), because it states
  near-certain verdicts.

## 6. Cross-Domain Generalization & Zero-Shot Retrieval Bottleneck (T41)

### Cross-Domain Transfer Matrix (Train-on-Rows, Test-on-Columns)
Scores are **binary** macro-F1 (each dataset's labels mapped to fake vs real; PUBHEALTH's mixture and
unproven left out), so the LIAR and PUBHEALTH diagonals (0.628, 0.771) are not their multi-class scores.
- **Transfer is partial at best:** LIAR and PUBHEALTH models score 0.40 to 0.59 on each other's test
  sets, against 0.63 and 0.77 in-domain.
- **COVID models do not transfer:** 0.315 on LIAR and 0.284 on PUBHEALTH, below always guessing one
  class. They learned what fake COVID posts look like (hashtags, roundup phrasing), which political and
  health claims do not share.

### Climate-FEVER Zero-Shot Extrapolation (Qwen3-8B, no Climate-FEVER training)
- **No evidence:** 0.398 macro-F1 (0.750 ROC-AUC).
- **Retrieved evidence (top 5 `bge-small` sentences from the pool of all Climate-FEVER evidence):** 0.429 (+0.032).
- **Gold evidence (each claim's own five sentences):** 0.479 (+0.082).
- **Retrieval bottleneck:** open retrieval gets only 39% of the gain that gold evidence gives, so
  finding the right evidence, not reading it, is the weak step. The classifiers trained on PUBHEALTH
  score at chance here (0.17 to 0.19 macro-F1).

## 7. Failure Modes & Representation Space Topology (T33, T32)
- **UMAP Representation Manifolds:**
  - **Binary Datasets (COVID, ISOT):** Embeddings show clean, linearly separable clusters for false vs. true claims across Transformer and LSTM backbones.
  - **Multi-Class Datasets (LIAR, PUBHEALTH):** Manifolds show substantial cluster overlap across the 6 LIAR categories and 4 PUBHEALTH categories, reflecting semantic ambiguity between adjacent truthfulness ratings (e.g., *barely-true* vs. *half-true*).
- **Dominant Failure Categories (Error Analysis):**
  1. **Source Cue Shortcuts:** Misclassifications driven by external URLs, handle patterns, or missing metadata.
  2. **Content / Semantic Ambiguity:** Errors on nuanced claims requiring external temporal verification or deep domain knowledge.

## 8. Specification & System Constraint Compliance (T42)
From `results/t42_specs_context_table.csv`: S1 to S5 and S9 from the frozen test results (each
config's seeds averaged, then the best config); S6 to S8 measured in the app on the laptop (RTX 5060
8 GB, encoder + Qwen3-8B + retrieval loaded, T40).

| Spec ID | Requirement | Target | Achieved | Status |
| :--- | :--- | :--- | :--- | :--- |
| **S1** | Algorithms explored | ≥ 2 families | 5 (classic, LSTM, encoder, LLM, kNN) | Pass |
| **S2** | COVID-19 weighted F1 | ≥ 0.95 | 0.981 (COVID-Twitter-BERT) | Pass |
| **S3** | LIAR 6-class accuracy | ≥ 0.28 | 0.299 (DeBERTa-v3-large LoRA, 1 seed) | Pass |
| **S4** | PUBHEALTH macro-F1 | ≥ 0.65 | 0.562 (tuned DeBERTa-v3-base + evidence) | **Fail**: the rare mixture and unproven labels |
| **S5** | ISOT clean accuracy | ≥ 0.95 | 1.000 | Pass, but see Section 4 |
| **S6** | Encoder latency, laptop GPU | ≤ 50 ms/claim | 18 ms median (45 ms max) | Pass |
| **S7** | Full app response, laptop | ≤ 5 s/claim | 0.16 s median (0.27 s max) | Pass |
| **S8** | App GPU memory | ≤ 8 GB | 5.6 GB | Pass |
| **S9** | Any single training run | ≤ 6 hours | 2.76 h (DeBERTa-v3-large, ISOT) | Pass |
| **S10** | Reproducibility | 1 config + 1 command | Every run from one YAML; 502 rows frozen | Pass |
