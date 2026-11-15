# factcheck: Can deep learning catch misinformation?

EE6405 Natural Language Processing group project (team of 5, NTU, November 2026).

Code: https://github.com/Dheemant19/factcheck

We compare traditional NLP, recurrent networks, transformer encoders and an 8B LLM on four
misinformation datasets. We then work out **why** each model decides what it decides: top
words, SHAP, LDA topics, shortcut removal, calibration, cross-dataset tests and significance
tests. Everything runs from one Python package (`factcheck/`). Every run is one YAML config,
every number in the report comes from one results file, and a Streamlit app (the "Fact-Check
Lab") shows the models on new claims.

| Family | Models |
|---|---|
| Traditional | TF-IDF + naive Bayes / logistic regression / linear SVM, averaged GloVe + SVM, kNN vote over sentence embeddings |
| Recurrent | Bi-LSTM on GloVe, plus a plain RNN (with and without gradient clipping), a GRU and a one-way LSTM |
| Encoders | DistilBERT, BERT, RoBERTa, DeBERTa-v3 (base and large), ModernBERT, COVID-Twitter-BERT, SciBERT; Optuna tuning; LoRA |
| LLM | Qwen3-8B zero-shot, kNN few-shot, evidence RAG, and QLoRA fine-tuning |

| Dataset | Task | Test rows |
|---|---|---|
| COVID-19 fake news (Patwa et al., 2021) | real vs fake tweets | 2,140 |
| LIAR (Wang, 2017) | 6-way truthfulness of political claims | 1,283 |
| PUBHEALTH (Kotonya & Toni, 2020) | 4-way health claims, with fact-check articles | 1,233 |
| ISOT (Ahmed et al., 2017) | real (Reuters) vs fake news articles | 3,866 |
| Climate-FEVER (Diggelmann et al., 2020) | zero-shot transfer only, no training | 1,535 |

## Main findings

All numbers are test-set scores from the frozen results (`results/final/runs_final.csv`, git
tag `v1.0-results`), averaged over seeds 13, 42 and 2026 where a run has three.

1. **Word counts are hard to beat on the easy datasets.** TF-IDF + logistic regression scores
   0.953 macro-F1 on COVID and 0.988 on cleaned ISOT, within 0.03 of the best model (0.980 and
   1.000). On LIAR (0.237 vs 0.296) and PUBHEALTH (0.487 vs 0.562) the gap is larger, but every
   model is low there.
2. **ISOT measures the news source, not truth.** Removing 16 kinds of source fingerprint (the
   Reuters dateline, image credits, embedded tweets and so on) only moves TF-IDF from 0.994 to
   0.988 accuracy. LDA topics alone still predict the label at 0.885 macro-F1, and headline
   capitalisation alone at 97.9% accuracy.
3. **The models learn a dataset's style, not falsehood.** A WHO-style tweet saying "drinking
   bleach helps stop the spread of #COVID19" is called true by TF-IDF. A true fact about the
   206 bones of the human skeleton is called fake, with full confidence, by COVID-Twitter-BERT.
4. **Evidence helps most.** Giving tuned DeBERTa-v3-base the five most relevant sentences of
   the fact-check article lifts PUBHEALTH from 0.503 to 0.562. Speaker metadata lifts LIAR from
   0.275 to 0.296, but as a prior about the speaker, not a check of the claim.
5. **Few gaps survive a significance test.** With a paired bootstrap and McNemar's test,
   Holm-corrected over 22 planned comparisons, the only macro-F1 gap beyond test-set noise is
   DeBERTa-v3-large LoRA vs full fine-tuning on PUBHEALTH (-0.096).
6. **Gates matter more than a second direction.** A GRU or one-way LSTM gets most of the way
   from a plain RNN to the Bi-LSTM. The plain RNN loses through short, unstable training, not
   through gradients vanishing.
7. **Little transfers across datasets.** COVID models score below always guessing one class on
   LIAR and PUBHEALTH.
   On Climate-FEVER, Qwen3-8B gets 0.398 with no evidence, 0.429 with retrieved evidence and
   0.479 with the gold evidence, so finding the evidence is the weak step.

### Main results (test macro-F1, mean ± std over seeds)

Within each family, the run with the best **validation** macro-F1 is shown, so the test set
never picks a model. "det." marks deterministic runs, which were run once.

| Family | COVID | LIAR | PUBHEALTH | ISOT (cleaned) |
|---|---|---|---|---|
| TF-IDF + NB / LR / SVM | 0.953 (det.) | 0.237 (det.) | 0.487 (det.) | 0.988 (det.) |
| Averaged GloVe + SVM | 0.916 (det.) | 0.221 (det.) | 0.411 (det.) | 0.975 (det.) |
| Bi-LSTM (GloVe) | 0.954 ± 0.004 | 0.242 ± 0.013 | 0.463 ± 0.021 | – |
| kNN vote (bge-small) | 0.915 (det.) | 0.241 (det.) | 0.419 (det.) | 0.925 (det.) |
| Qwen3-8B zero-shot | 0.806 (det.) | 0.179 (det.) | 0.275 (det.) | – |
| Qwen3-8B kNN few-shot | 0.937 (det.) | 0.259 (det.) | 0.457 (det.) | – |
| Base encoder | 0.966 ± 0.004 | 0.288 ± 0.007 | 0.496 ± 0.014 | 1.000 ± 0.000 |
| Large / domain encoder | 0.980 ± 0.002 | 0.296 ± 0.012 | 0.509 ± 0.017 | 1.000 ± 0.000 |
| Optuna-tuned DeBERTa-v3-base | 0.972 ± 0.000 | 0.275 ± 0.011 | 0.504 ± 0.016 | – |
| Tuned + one extra (class weights / metadata / evidence) | 0.971 ± 0.000 | 0.296 ± 0.013 | 0.562 ± 0.024 | – |
| Qwen3-8B QLoRA | 0.979 (1 seed) | 0.289 (1 seed) | 0.515 (1 seed) | 1.000 (1 seed) |

The full table, with the run behind every cell, is `results/results_v1_table.md`. The
specification targets S1 to S10 are in `results/t42_specs_context_table.csv`: all pass
except S4, the PUBHEALTH macro-F1 target of 0.65 (best 0.562; the published SciBERT result is
0.705).

## Where to find what

| Looking for | Where |
|---|---|
| Every run's scores (frozen) | `results/final/runs_final.csv`, hash and run counts in `results/final/MANIFEST.md` |
| Figures used in the report and video | `results/figures/` |
| Significance tests | `results/significance.csv` |
| Extra-effort steps (tuning, metadata, evidence, class weights) | `results/e5_extra_effort.md` |
| Shortcut study, LDA topics, recurrent-cell gradients | `results/shortcuts/`, `results/topics/`, `results/t55_gradients.csv` |
| Cross-dataset and Climate-FEVER results | `results/cross_dataset.csv`, `results/t41_beyond_misinformation.md` |
| What each finding means and what it cost | `docs/m1_interpretation.md`, `docs/m2_interpretation.md`, `docs/m3_interpretation.md`, `docs/m4_T33toT42_Note.md` |
| Data decisions (ISOT cleanup, splits, shortcut rules, retrieval) | `docs/m3_isot_and_splits.md`, `docs/m3_retrieval_and_shortcuts.md`, `docs/m3_prompts_and_inputs.md` |
| Code reviews between members | `docs/t44_review_*.md` |
| Who did what | `docs/contribution_log.md` |
| Machines the results were produced on | `docs/environments.md` |

## Setup

You need Python 3.11 or newer and `pip`. Training the encoders and the LLM needs an NVIDIA GPU.
A CPU-only machine is enough for the traditional baselines, the figures, the tests and the
app with the TF-IDF model.

**1. Create and activate a virtual environment:**

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
```

**2. Install PyTorch for your machine.** The right build depends on your GPU driver, so it is
not in the requirements. Run `nvidia-smi` and pick a build at or below the CUDA version it
shows from [pytorch.org/get-started](https://pytorch.org/get-started/locally/). RTX 50-series
GPUs need a CUDA 12.8 or newer build. For example:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130   # NVIDIA, CUDA 13.0 build
pip install torch --index-url https://download.pytorch.org/whl/cpu    # CPU only
```

**3. Install the project and check the machine:**

```bash
pip install -r requirements.txt    # pinned versions we tested, plus this package (editable)
python scripts/check_env.py        # GPU, bf16 and 4-bit support
python scripts/download_data.py covid
python scripts/smoke_test.py       # a one-minute DistilBERT run; prints PASS (--cpu without a GPU)
```

`requirements.txt` pins the direct dependencies; `requirements-lock.txt` is a full freeze of
the environment the results were produced in.

## Data

```bash
python scripts/download_data.py --list      # sources and sizes, downloads nothing
python scripts/download_data.py             # all five datasets into data/raw/
python scripts/download_data.py --glove     # GloVe 6B 300-d for the recurrent models (862 MB)
python scripts/make_splits.py --check       # your download matches the committed splits
```

**ISOT needs a Kaggle login.** Create an API token on Kaggle (Settings > API > Create New
Token) and put `kaggle.json` in `~/.kaggle/`, or set `KAGGLE_USERNAME` and `KAGGLE_KEY` (see
`.env.example`), then `pip install kagglehub`. You can also download the zip from
[Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) and put
`Fake.csv` and `True.csv` in `data/raw/isot/`. The other datasets need no login.

**Fixed splits.** Which row is in which split is committed in `data/splits/<dataset>.csv`, so
every machine trains and tests on identical rows. Loading checks the raw data against it and
fails loudly if they differ. COVID, LIAR and PUBHEALTH keep their official splits. ISOT is
split 80/10/10, stratified, with seed 42.

**Data decisions.**
- PUBHEALTH: 41 rows with a broken label are dropped.
- ISOT: 631 articles with an empty body are dropped (630 of them fake), then 5,609 exact
  duplicates.
- ISOT `clean` variant (`load_splits("isot", variant="clean")`): source fingerprints are
  removed, one named rule each in `factcheck/data/shortcuts.py`.
- Leakage: train rows that copy a validation or test row are excluded
  (`data/splits/exclusions.csv`).

```python
from factcheck.data import load_splits, get_dataset

splits = load_splits("liar")                   # {"train": df, "validation": df, "test": df}
splits = load_splits("isot", variant="clean")  # source fingerprints removed
info = get_dataset("liar")                     # labels, label2id, max_length, ...
```

## Reproducing the results

### 1. Train

One run is one config and one command. Re-running a config replaces its rows in
`results/runs.csv`.

```bash
python -m factcheck.train --config configs/classic/tfidf_lr_liar.yaml           # CPU, seconds
python -m factcheck.train --config configs/lstm/lstm_liar.yaml --seeds 13 42 2026
python -m factcheck.train --config configs/encoder/deberta_v3_base_liar.yaml --seeds 13 42 2026
python -m factcheck.tune  --config configs/encoder/deberta_v3_base_liar.yaml   # Optuna search
```

Configs are grouped by family in `configs/` (`classic/`, `lstm/`, `encoder/`, `llm/`, `knn/`),
and `configs/README.md` explains each group. The full experiment sets are job lists in
`jobs/`, one config per line, run by one worker per GPU:

```bash
CUDA_VISIBLE_DEVICES=0 bash jobs/queue.sh jobs/t17_encoders.txt
```

| Job list | Experiment |
|---|---|
| `t17_encoders.txt`, `t19_large_encoders.txt` | base, large and domain encoders, all datasets, 3 seeds |
| `t18_tune.sh`, `t18_tuned.txt` | Optuna search, then the tuned DeBERTa-v3-base |
| `t20_llm.txt`, `t27_qlora.txt` | Qwen3-8B prompting modes and QLoRA |
| `t26_lora.txt`, `t26_lora_r64_alpha.txt` | LoRA rank and target sweep |
| `t28_shortcuts.txt` | raw vs cleaned ISOT |
| `t29_ablations.txt` | the extra-effort steps (metadata, evidence, class weights) |
| `t55_rnn.txt` | plain RNN, GRU and one-way LSTM next to the Bi-LSTM |

We trained on RTX 3090 and RTX A5000 GPU servers. The longest single run (DeBERTa-v3-large on
ISOT) took 2.76 hours, and Qwen3-8B needs about 17 GB of GPU memory in bf16.

### 2. Tables, figures and tests on the results

These read `results/runs.csv` (identical to the frozen file) and run on a CPU:

```bash
python -m factcheck.evaluate                   # summary of logged runs and the specs check
python notebooks/02_results_v1.py              # main results table and v1_* figures
python -m factcheck.specs_table_t42            # model ladder, shortcut chart, specs table
python -m factcheck.plots_cross_dataset_t41    # cross-dataset heatmap, Climate-FEVER chart
python -m factcheck.extra_effort               # extra-effort table and chart
python -m factcheck.topics                     # LDA topics by label (about 2 min)
python scripts/find_shortcuts.py               # top words before and after cleaning
python scripts/find_shortcuts.py --headline    # ISOT label from headline capitalisation
python scripts/t55_gradients.py                # recurrent cells: scores and gradient sizes
```

These need the trained models in `outputs/` or the per-row predictions in
`results/predictions/`, which are too large to commit:

```bash
python -m factcheck.ensemble --skip-ensemble   # significance tests (paired bootstrap, McNemar)
python -m factcheck.evaluate_cross_dataset     # train on one dataset, test on another
python scripts/transfer_climate_fever.py <model dirs>   # PUBHEALTH models on Climate-FEVER
python -m factcheck.explain --checkpoint outputs/distilbert_covid/seed42   # SHAP
python notebooks/03_interpretation.py          # calibration, UMAP, error tagging, SHAP vs TF-IDF
```

`notebooks/01_eda.ipynb` has the dataset exploration (sizes, label balance, lengths,
duplicates, shortcuts).

### 3. Check against the frozen results

```bash
python scripts/freeze_results.py --check
python -c "import hashlib; print(hashlib.sha256(open('results/final/runs_final.csv','rb').read()).hexdigest())"
```

The hash should match `results/final/MANIFEST.md`. Every number in the report and the video
comes from that file.

## The app (Fact-Check Lab)

```bash
streamlit run app/app.py
```

Paste a claim and the app shows the verdict on the dataset's rating scale, how sure the model
is, the words that pushed it there (SHAP for a transformer, exact weights for TF-IDF), the most
similar already fact-checked claims and evidence sentences, and an optional second opinion from
Qwen3-8B. Other views compare models side by side, show the ISOT shortcut (a Reuters dateline
flips the raw-trained model) and show the results and specs.

The app loads trained models from `outputs/<run>/seed<N>/model`, which are not committed.
Training `configs/classic/tfidf_lr_covid.yaml` (seconds on a CPU) is enough to start it. Each
further config you train appears automatically. The full demo ran on a laptop RTX 5060 (8 GB)
with an encoder, Qwen3-8B in 4-bit and the retrieval index loaded together: 5.6 GB of GPU
memory and 0.16 s per claim.

## Repository layout

```
factcheck/
├── README.md
├── pyproject.toml           package metadata; pip install -e .
├── requirements.txt         pinned versions (requirements-lock.txt: full freeze)
├── configs/                 one YAML per run, by family; specs.yaml holds the targets S1-S10
├── factcheck/               the package
│   ├── data/                loaders, cleaning, fixed splits, shortcut removal, leakage checks
│   ├── models/              classic.py, lstm.py, encoder.py, llm.py (prompting, LoRA, QLoRA)
│   ├── train.py, tune.py    one run from a config; Optuna search
│   ├── evaluate.py          the one metrics function and the specs check
│   ├── stats.py, ensemble.py   paired bootstrap, McNemar, Holm correction
│   ├── retrieval.py         bge-small embedding index over train claims and evidence
│   ├── explain.py, topics.py, style.py   SHAP, LDA topics, style stripping
│   ├── serve.py             inference engine for the app
│   └── ...                  figure and table scripts (specs_table_t42.py, plots_cross_dataset_t41.py, ...)
├── scripts/                 data download, splits, environment check, analysis scripts
├── jobs/                    GPU job lists and the queue runner
├── notebooks/               EDA, results table, interpretation pipeline
├── app/                     Streamlit app (DESIGN.md and PRODUCT.md at the top level describe its design)
├── results/                 runs.csv, final/ (frozen), figures/ and analysis outputs
├── tests/                   pytest suite
└── docs/                    interpretation write-ups, data notes, reviews, contribution log
```

## Tests and code style

```bash
pytest                       # the full suite, on a CPU in a few minutes
ruff check . && black --check .
```

Settings are in `pyproject.toml`. Tuning only ever sees the validation split, and each final
model was scored on test once.

## Team

| Member | Role |
|---|---|
| M1 | Training and tuning code, encoder and Bi-LSTM runs on the GPU servers, traditional baselines, main results table, the app, style stripping, Climate-FEVER transfer |
| M2 | Qwen3-8B few-shot and RAG modes, LoRA and QLoRA, extra-effort ablations, recurrent-cell study, SHAP for QLoRA |
| M3 | Data loaders and fixed splits, shortcut removal and leakage checks, retrieval, EDA, significance tests, LDA topics, results freeze |
| M4 | Metrics, baseline comparison, SHAP, calibration, shortcut study, cross-dataset tests, report figures and specs table |
| M5 | Slide template, slides and video |

Each piece of work is logged with its author and date in `docs/contribution_log.md`.
