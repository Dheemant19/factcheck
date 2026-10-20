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
* `scripts/check_env.py` and `jobs/setup_cluster.sh`: laptop and cluster environment checks (T02).
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

PyTorch is installed separately because the right build depends on the GPU. We use
[uv](https://docs.astral.sh/uv/), not plain pip.

### Laptop

```powershell
uv venv --python 3.11
.venv\Scripts\Activate.ps1
uv pip install torch --index-url https://download.pytorch.org/whl/cu132   # RTX 50-series needs cu128 or newer
uv pip install -r requirements.txt                                      # pinned versions + this package
python scripts\check_env.py                                              # GPU, bf16 and 4-bit checks
```

`requirements.txt` pins the direct dependencies; `requirements-laptop.lock.txt` is the full
freeze of the working laptop environment. `uv pip install -e ".[all]"` installs unpinned.

### MLDA cluster (4x RTX 3090)

Connect with the NTU VPN on, then:

```bash
git clone https://github.com/Dheemant19/factcheck.git && cd factcheck
bash jobs/setup_cluster.sh        # uv, .venv (Python 3.11), torch for the driver's CUDA, package
source .venv/bin/activate
```

Download data and models on the server itself (scp over the VPN is slow).

## Data

```bash
python scripts/download_data.py --list      # sources and sizes, downloads nothing
python scripts/download_data.py             # all five datasets into data/raw/
python scripts/download_data.py --glove     # GloVe 6B 300-d for the Bi-LSTM (862 MB)
```

**ISOT needs a Kaggle login.** Make a Kaggle account, create an API token (Settings > API >
Create New Token), put `kaggle.json` in `~/.kaggle/` or set `KAGGLE_USERNAME` and `KAGGLE_KEY`
(see `.env.example`), and `uv pip install kagglehub`. Or download the zip from
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
├── requirements.txt        pinned versions for reproducibility
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
├── scripts/                download_data.py, check_env.py
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
