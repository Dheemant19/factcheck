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

`data/` is not committed. Cleaning and the fixed splits are code (T03); the ISOT split ID
files will be committed to the repo so every machine uses identical splits.

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
