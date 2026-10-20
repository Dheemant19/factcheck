"""Download the five raw datasets (and optionally GloVe) into data/raw/ (task T02).

    python scripts/download_data.py --list          # sources and sizes, downloads nothing
    python scripts/download_data.py                 # every dataset
    python scripts/download_data.py liar covid      # just these
    python scripts/download_data.py --glove         # also GloVe 6B 300-d for the Bi-LSTM (862 MB zip)

ISOT NEEDS A KAGGLE LOGIN. It is only published on Kaggle, so before downloading it:
  1. make a free account at https://www.kaggle.com and open Settings > API > "Create New Token";
     this downloads kaggle.json;
  2. either put kaggle.json in ~/.kaggle/ (Windows: C:\\Users\\<you>\\.kaggle\\), or set
     KAGGLE_USERNAME and KAGGLE_KEY from it (see .env.example);
  3. pip install kagglehub
Without a login, download the zip by hand from
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset and put Fake.csv and
True.csv in data/raw/isot/. The other four datasets need no login.

Files land in data/raw/<dataset>/ exactly as published; cleaning and splitting are done by
the data loaders (T03), not here. Already-downloaded files are skipped, so it is safe to re-run.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import zipfile
from collections.abc import Callable
from pathlib import Path

import requests
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from factcheck.paths import RAW_DIR  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("download")

COVID_URL = "https://raw.githubusercontent.com/diptamath/covid_fake_news/main/data/"
COVID_FILES = ["Constraint_Train.csv", "Constraint_Val.csv", "english_test_with_labels.csv"]
LIAR_URL = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"
PUBHEALTH_REPO = "ImperialCollegeLondon/health_fact"
PUBHEALTH_REVISION = "refs/convert/parquet"  # the repo itself only has a loading script
PUBHEALTH_FILES = [f"default/{split}/0000.parquet" for split in ("train", "validation", "test")]
ISOT_KAGGLE = "clmentbisaillon/fake-and-real-news-dataset"
ISOT_FILES = ["Fake.csv", "True.csv"]
CLIMATE_FEVER_REPO = "tdiggelm/climate_fever"
CLIMATE_FEVER_FILES = ["data/test-00000-of-00001.parquet"]
GLOVE_URL = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"
GLOVE_FILE = "glove.6B.300d.txt"


def download_file(url: str, dest: Path, *, min_bytes: int = 1) -> Path:
    """Stream ``url`` to ``dest`` (atomic rename on success). Skips if ``dest`` exists."""
    if dest.exists() and dest.stat().st_size >= min_bytes:
        log.info("exists, skipping: %s", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0)) or None
        with (
            open(tmp, "wb") as fh,
            tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar,
        ):
            for chunk in response.iter_content(chunk_size=1 << 16):
                fh.write(chunk)
                bar.update(len(chunk))
    if tmp.stat().st_size < min_bytes:
        tmp.unlink()
        raise OSError(f"Download of {url} looks truncated (< {min_bytes} bytes)")
    tmp.replace(dest)
    return dest


def download_hf_files(
    repo_id: str, filenames: list[str], dest_dir: Path, revision: str | None = None
) -> None:
    """Fetch individual files from a Hugging Face dataset repo, keeping their repo paths.

    Parquet files are fetched directly instead of via ``datasets.load_dataset`` so the old
    script-based loaders (PUBHEALTH) are not needed.
    """
    from huggingface_hub import hf_hub_download

    for filename in filenames:
        target = dest_dir / filename
        if target.exists():
            log.info("exists, skipping: %s", target)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        cached = hf_hub_download(repo_id, filename, repo_type="dataset", revision=revision)
        target.write_bytes(Path(cached).read_bytes())
        log.info("saved %s", target)


def extract_zip(archive: Path, dest_dir: Path, members: list[str] | None = None) -> None:
    """Extract ``archive`` into ``dest_dir``, refusing paths that escape it."""
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            target = (dest_dir / member.filename).resolve()
            if not str(target).startswith(str(dest_dir.resolve())):
                raise OSError(f"Unsafe path in archive: {member.filename}")
        zf.extractall(dest_dir, members=members)


def get_covid(raw_dir: Path) -> None:
    for filename in COVID_FILES:
        download_file(COVID_URL + filename, raw_dir / filename, min_bytes=100_000)


def get_liar(raw_dir: Path) -> None:
    archive = download_file(LIAR_URL, raw_dir / "liar_dataset.zip", min_bytes=500_000)
    if not (raw_dir / "train.tsv").exists():
        extract_zip(archive, raw_dir)


def get_pubhealth(raw_dir: Path) -> None:
    download_hf_files(PUBHEALTH_REPO, PUBHEALTH_FILES, raw_dir, revision=PUBHEALTH_REVISION)


def get_isot(raw_dir: Path) -> None:
    if all((raw_dir / f).exists() for f in ISOT_FILES):
        log.info("exists, skipping: %s", raw_dir)
        return
    manual = (
        f"Or download https://www.kaggle.com/datasets/{ISOT_KAGGLE} by hand and put "
        f"{' and '.join(ISOT_FILES)} in {raw_dir}. See the top of this script."
    )
    try:
        import kagglehub
    except ImportError as err:
        raise RuntimeError(
            "ISOT is on Kaggle and needs a Kaggle login: `pip install kagglehub` and set up "
            "your Kaggle API token (~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY). " + manual
        ) from err
    try:
        cached = Path(kagglehub.dataset_download(ISOT_KAGGLE))
    except Exception as err:
        raise RuntimeError(
            "Kaggle refused the download: ISOT needs a Kaggle login. Check that your API token "
            "is in ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY are set. " + manual
        ) from err
    raw_dir.mkdir(parents=True, exist_ok=True)
    for filename in ISOT_FILES:
        shutil.copy2(next(cached.rglob(filename)), raw_dir / filename)


def get_climate_fever(raw_dir: Path) -> None:
    download_hf_files(CLIMATE_FEVER_REPO, CLIMATE_FEVER_FILES, raw_dir)


# name -> (downloader, source, approximate size)
DATASETS: dict[str, tuple[Callable[[Path], None], str, str]] = {
    "covid": (get_covid, "github.com/diptamath/covid_fake_news", "0.8 MB"),
    "liar": (get_liar, LIAR_URL, "1 MB"),
    "pubhealth": (get_pubhealth, f"huggingface.co/datasets/{PUBHEALTH_REPO}", "40 MB"),
    "isot": (get_isot, f"kaggle.com/datasets/{ISOT_KAGGLE} (KAGGLE LOGIN NEEDED)", "42 MB zip, 115 MB unpacked"),
    "climate_fever": (get_climate_fever, f"huggingface.co/datasets/{CLIMATE_FEVER_REPO}", "0.9 MB"),
}  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("datasets", nargs="*", help=f"any of {list(DATASETS)} (default: all)")
    parser.add_argument("--list", action="store_true", help="show sources and sizes, then exit")
    parser.add_argument("--glove", action="store_true", help="also download GloVe 6B (862 MB)")
    args = parser.parse_args()

    names = args.datasets or list(DATASETS)
    unknown = sorted(set(names) - set(DATASETS))
    if unknown:
        parser.error(f"unknown dataset(s) {unknown}; choose from {list(DATASETS)}")
    if args.list:
        for name in names:
            _, source, size = DATASETS[name]
            print(f"{name:14s} {size:>30s}  {source}")
        print(f"{'glove':14s} {'862 MB zip':>30s}  {GLOVE_URL}")
        return

    failed = []
    for name in names:
        downloader, source, size = DATASETS[name]
        log.info("== %s (%s, %s)", name, source, size)
        try:
            downloader(RAW_DIR / name)
        except Exception as err:  # keep going so one login problem does not block the rest
            log.error("%s failed: %s", name, err)
            failed.append(name)

    if args.glove:
        glove_dir = RAW_DIR / "glove"
        archive = download_file(GLOVE_URL, glove_dir / "glove.6B.zip", min_bytes=8 * 10**8)
        if not (glove_dir / GLOVE_FILE).exists():
            extract_zip(archive, glove_dir, members=[GLOVE_FILE])

    if failed:
        raise SystemExit(f"Failed: {failed}. See the messages above.")


if __name__ == "__main__":
    main()
