# factcheck: Can deep learning catch misinformation?

EE6405 group project (team of 5, due **Sun 15 Nov 2026, 11:59 pm**). We compare a Bi-LSTM,
fine-tuned transformer encoders, LoRA and a retrieval-augmented 8B LLM on four misinformation
datasets, explain why they decide what they decide, and show it in the Streamlit "Fact-Check
Lab". The full plan is `Group_Project_Plan.html` (shared separately).

Training, notebooks and the app all import this one package, every run is one YAML config,
and every number lands in `results/runs.csv`.

## State of the repo

This is the skeleton. The layout follows plan Section 16; modules that are not written yet
are stubs whose docstring names the owner and task. What works today:

* `scripts/download_data.py`: downloads all five datasets (T02). **ISOT needs a Kaggle login.**
* `scripts/check_env.py`, `scripts/smoke_test.py` and `jobs/setup_cluster.sh`: environment check, a tiny
  DistilBERT training run, and GPU-server setup (T02). Tested machines are in `docs/environments.md`.
* `app/app.py`: the Streamlit app, with designed empty states until models and results exist.
* `factcheck.data`: `load_splits(name)` for all five datasets, with fixed split files (T03); the
  `clean` variant without source shortcuts, and train rows that copy a test row excluded (T13).
* `factcheck.retrieval`: embedding search over already fact-checked training claims and over
  evidence sentences, and the kNN-vote baseline (`configs/knn/`) (T08).
* `notebooks/01_eda.ipynb`: sizes, label balance, token lengths, duplicates, shortcuts (T09).

## Who owns what

| Member | Role | Owns |
|---|---|---|
| **M1 + M2** | Training team | `factcheck/models/`, `factcheck/train.py`, `factcheck/tune.py`, `configs/`, `jobs/`; dataset download and cluster setup (T02) |
| **M3** | Data and retrieval lead | `factcheck/data/` (loaders, cleaning, fixed splits, shortcut removal), `factcheck/retrieval.py`, `notebooks/01_eda` |
| **M4** | Evaluation and interpretation lead | `factcheck/evaluate.py`, `factcheck/results.py`, `factcheck/explain.py`, `notebooks/02_results`, `notebooks/03_interpretation` |
| **M5** | Product and presentation lead | `app/`, this README, `docs/`, slides, video |

Task IDs (T01 to T51) and deadlines are in plan Section 20.

## Setup

You need Python 3.11 or newer (the pinned versions need it) and `pip`. Training needs an NVIDIA
GPU; a CPU-only machine is enough for the app, the notebooks and the tests.

**1. Create and activate a virtual environment** (or use conda, if that is what you have):

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
```

**2. Install PyTorch for your machine.** The right build depends on your GPU and driver, so it
is not in the requirements. Pick the command for your system on
[pytorch.org/get-started](https://pytorch.org/get-started/locally/). For an NVIDIA GPU, run
`nvidia-smi` first: the "CUDA Version" (newer drivers: "CUDA UMD Version") it shows is the
newest CUDA build your driver supports, so pick a PyTorch build at or below it.
Newer GPUs (RTX 50-series) need a CUDA 12.8 or newer build. For example:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130   # NVIDIA, CUDA 13.0 build
pip install torch --index-url https://download.pytorch.org/whl/cpu    # CPU only (Windows, Linux)
pip install torch                                                     # macOS (Apple silicon uses MPS)
```

**3. Install the project:**

```bash
pip install -r requirements.txt    # pinned versions we tested, plus this package (editable)
python scripts/check_env.py        # checks the GPU, bf16 and 4-bit support
```

**4. Prove the machine can train** (needs the COVID data, see Data below):

```bash
python scripts/download_data.py covid
python scripts/smoke_test.py       # tiny DistilBERT run, about a minute; --cpu without a GPU
```

It should print PASS (validation accuracy well above 0.80). Add your machine to
`docs/environments.md`.

`requirements.txt` pins the direct dependencies so runs reproduce; `requirements-lock.txt` is
a full freeze of one working environment (Windows, CUDA 13.2) for reference. To install the
latest compatible versions instead, use `pip install -e ".[all]"`, or a smaller set of extras
such as `pip install -e ".[app]"` (see `pyproject.toml`).

### On a shared GPU server

The same steps work on any Linux GPU server. Clone the repo there and download data and
models on the server itself rather than copying them over (copying over a VPN is slow):

```bash
git clone https://github.com/Dheemant19/factcheck.git && cd factcheck
bash jobs/setup_cluster.sh         # optional: steps 1-3 above, picking torch from nvidia-smi
```

Long runs go in the job lists `jobs/gpu0.sh` to `jobs/gpu3.sh`, one per GPU (instructions at
the top of each file).

## Data

```bash
python scripts/download_data.py --list      # sources and sizes, downloads nothing
python scripts/download_data.py             # all five datasets into data/raw/
python scripts/download_data.py --glove     # GloVe 6B 300-d for the Bi-LSTM (862 MB)
```

**ISOT needs a Kaggle login.** Make a Kaggle account, create an API token (Settings > API >
Create New Token), put `kaggle.json` in `~/.kaggle/` or set `KAGGLE_USERNAME` and `KAGGLE_KEY`
(see `.env.example`), and `pip install kagglehub`. Or download the zip from
[Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) and put
`Fake.csv` and `True.csv` in `data/raw/isot/`. The other four datasets need no login.

| Dataset | Source |
|---|---|
| D1 `covid` | authors' GitHub release (diptamath/covid_fake_news) |
| D2 `liar` | UCSB zip (Wang, 2017) |
| D3 `pubhealth` | Hugging Face `ImperialCollegeLondon/health_fact` |
| D4 `isot` | Kaggle `clmentbisaillon/fake-and-real-news-dataset` (login needed) |
| D5 `climate_fever` | Hugging Face `tdiggelm/climate_fever` (test only) |

`data/` is not committed, except `data/splits/` (below).

### Loading a dataset

```python
from factcheck.data import load_splits, get_dataset

splits = load_splits("liar")                   # {"train": df, "validation": df, "test": df}
splits = load_splits("isot", variant="clean")  # clean = source fingerprints removed (ISOT),
                                               # URLs and @mentions masked (COVID)
info = get_dataset("liar")                     # labels, label2id, fake_label_ids, max_length
```

Every DataFrame starts with the same columns: `id`, `text` (model input), `text_raw` (as
published), `label`, `label_name`, `binary_label` (1 = false/fake, 0 = true/real, -1 = not
used in cross-domain tests) and `in_llm_subset`. Extra columns per dataset: `meta_*` (LIAR),
`main_text` and `explanation` (PUBHEALTH), `evidence` (Climate-FEVER), `subject` and `date`
(ISOT). **Never feed `explanation`, `subject` or `date` to a model**: they give the answer away.

| Dataset | Labels (class id order) | train / validation / test |
|---|---|---|
| `covid` | real, fake | 6,420 / 2,140 / 2,140 |
| `liar` | pants-fire, false, barely-true, half-true, mostly-true, true | 10,269 / 1,284 / 1,283 |
| `pubhealth` | false, mixture, true, unproven | 9,804 / 1,214 / 1,233 |
| `isot` | real, fake | 30,926 / 3,866 / 3,866 (2,000 of test in the LLM subset) |
| `climate_fever` | supports, refutes, not_enough_info, disputed | 0 / 0 / 1,535 (test only) |

**Fixed splits.** Which row is in which split is stored in `data/splits/<dataset>.csv`
(`id, split, in_llm_subset`), committed to the repo, so the cluster and every laptop use
identical rows; loading checks the raw data against it and fails loudly if they differ.
COVID, LIAR and PUBHEALTH keep their official splits. ISOT is split 80/10/10, stratified,
seed 42, with a stratified 2,000-article test subset for the 8B LLM runs. The files were made
once with `python scripts/make_splits.py`; `--check` verifies them against your download.

**Data decisions.** PUBHEALTH: 41 rows with label -1 (broken in the source) are dropped.
ISOT: 631 articles with an empty body are dropped (630 of them fake, so keeping them would
teach "title only means fake"), then 5,609 exact duplicates; 38,658 remain. The reasoning
behind the ISOT cleanup and the 80/10/10 split is in `docs/m3_isot_and_splits.md`. COVID
ids restart at 1 in each official file, so our ids are `train-17`, `validation-17`, and so on.
Cleaning is light (encoding repair, HTML, whitespace; no lower-casing), and the prepared
tables are cached in `data/cache/datasets/`, so ISOT loads in under a second after the first
time. Tests: `pytest tests/test_data.py`.

**Shortcuts and leakage (T13).** `variant="clean"` removes ISOT's source fingerprints (the
Reuters dateline, image credits, embedded tweets, links, site boilerplate, stripped
apostrophes; one named rule each in `factcheck/data/shortcuts.py`), keeping row ids identical
to `raw`. Train rows that are copies or near-copies of a validation/test row are listed in
`data/splits/exclusions.csv` and dropped by `load_splits` (validation and test never change;
`drop_excluded=False` loads the full split). Counts per rule and per check are in
`results/leakage_report.csv`; the reasoning is in `docs/m3_retrieval_and_shortcuts.md`.
Re-run with `python scripts/find_shortcuts.py` and `python -m factcheck.data.leakage`.

## Retrieval (T08)

```bash
python -m factcheck.retrieval build      # embed and cache everything (GPU: minutes; CPU: about 2 h)
python -m factcheck.retrieval check      # example neighbours, kNN-vote accuracy, duplicate flags
python -m factcheck.train --config configs/knn/knn_liar.yaml   # kNN-vote baseline into runs.csv
```

```python
from factcheck.retrieval import load_claim_index, load_evidence

index = load_claim_index("liar")                   # TRAIN claims only
hits = index.search_texts(["Says the economy grew 5% last year."], k=8)[0]
hits[0].text, hits[0].label_name, hits[0].score    # most similar first
load_evidence("pubhealth").top_k("33456", k=5)     # best 5 sentences of that claim's article
load_evidence("climate_fever").search_pool(["Polar bears are dying out."], k=5)
```

Embeddings come from `BAAI/bge-small-en-v1.5` (unit length, so search is one matrix multiply)
and are cached as float16 in `data/cache/retrieval/`; the cache key includes the exact texts,
so a change in the data rebuilds it. Only train claims are indexed, so a prompt can never show
a validation or test verdict.

## Training and results

```bash
python -m factcheck.train --config configs/lstm/lstm_liar.yaml           # one run, seed from the config
python -m factcheck.train --config configs/lstm/lstm_liar.yaml --seeds 13 42 2026
python -m factcheck.train --config configs/encoder/bert_liar.yaml          # encoders (T06)
python -m factcheck.evaluate                                              # logged runs + specs S1-S10
```

Every run is scored by `factcheck.evaluate` and logged to `results/runs.csv` (one row per run,
seed and split; re-running a config replaces its rows). Per-row probabilities go to
`results/predictions/<run>/seed<N>/` (not committed) for ensembles, significance tests and
calibration curves. Spec targets live in `configs/specs.yaml`.

## The app

```bash
streamlit run app/app.py
```

It finds trained models under `outputs/<run>/seed<N>/model` and results in `results/runs.csv`.

## Repository layout

```
factcheck/
├── README.md
├── pyproject.toml          pip install -e .
├── requirements.txt        pinned versions for reproducibility (requirements-lock.txt: full freeze)
├── configs/                one YAML per run (template.yaml)
├── factcheck/              the reusable package
│   ├── data/               loaders behind a registry; cleaning; shortcut removal
│   ├── models/             lstm.py, encoder.py, llm.py
│   ├── retrieval.py        embedding index build, cache and search
│   ├── train.py            python -m factcheck.train --config configs/x.yaml
│   ├── tune.py             Optuna search
│   ├── evaluate.py         the one metrics function and the specs check
│   ├── explain.py          SHAP, UMAP, calibration
│   ├── results.py          runs.csv
│   └── paths.py            every path, overridable by environment variable
├── scripts/                download_data.py, check_env.py, smoke_test.py
├── jobs/                   setup_cluster.sh, gpu0.sh ... gpu3.sh
├── notebooks/              01_eda, 02_results, 03_interpretation
├── app/app.py              Streamlit Fact-Check Lab
├── results/                runs.csv and figures/
├── tests/                  loaders, cleaning and metrics
└── docs/                   contribution log, reflection, GAI form
```

## Working together

* `black` and `ruff` (settings in `pyproject.toml`); docstrings and type hints on every function.
* One branch per task, a pull request reviewed by another member, then merge.
* Tuning sees validation only; each final model is scored on test once.
* Log every piece of work in `docs/contribution_log.md`.
* New dependency: add it to `pyproject.toml` (in `dependencies` or the right extra), then pin the
  version you tested in `requirements.txt` (`pip show <package>` prints it), so everyone can
  reproduce with `pip install -r requirements.txt`. Refresh the full freeze with
  `pip freeze --exclude-editable > requirements-lock.txt`.
