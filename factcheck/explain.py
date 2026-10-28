"""SHAP token attributions for fine-tuned encoders (owner M4, T16; T33 builds on it).

Which words pushed a verdict towards each label, on a fixed 50-example test set per dataset,
saved as interactive HTML highlights and as arrays for the T33/T34 analysis:

    python -m factcheck.explain --checkpoint outputs/distilbert_covid/seed42
    python -m factcheck.explain --checkpoint outputs/roberta_isot_clean/seed42 --variant raw

``--checkpoint`` is a run folder written by ``factcheck.train`` (``outputs/<name>/seed<N>/``,
with ``model/`` and ``config.yaml``), its ``model/`` folder, or a Hugging Face model id (then
``--dataset`` is needed). A run folder's ``config.yaml`` gives the dataset, the text variant
and the truncation the model was trained with, so the explained input is exactly what the
model saw. ``--variant`` overrides the text variant (the shortcut study, T34, explains a
raw-trained model on cleaned text and the other way round); the 50 rows stay the same.

Each run writes ``<name>_seed<N>[_on_<variant>]_shap.html`` and ``..._shap.npz`` to
``outputs/explanations/``. The npz holds one entry per example (load it with
``allow_pickle=True``, because examples have different token counts):

    ids, texts, labels      the examples, their text and true class id
    tokens                  per example, the tokens SHAP masked (str array)
    values                  per example, attributions of shape (n_tokens, n_labels)
    base_values, probs      (n, n_labels): expected and predicted probabilities
    output_names            label names, the column order of values / base_values / probs
    fake_label_ids          the labels that count as false (summed for the app's red/green)

Values are in probability units: for each label, base value + the sum of the token values is
the model's probability for that label (up to SHAP's sampling error).

``explain_claim`` explains one pasted claim for the app's "why" panel (T22).
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
import torch
import yaml
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from factcheck.data import get_dataset, load_splits
from factcheck.data.registry import DatasetInfo
from factcheck.models.encoder import encode_texts
from factcheck.models.lstm import LSTMPredictor #[cite: 17, 18]
from factcheck.paths import OUTPUT_DIR, PREDICTIONS_DIR

NUM_EXAMPLES = 50
EXAMPLE_SEED = 42
MAX_EVALS = 500  # model calls per example for the partition explainer
BATCH_SIZE = 64
EXPLANATIONS_DIR = OUTPUT_DIR / "explanations"


@dataclass
class Explainer:
    """A loaded checkpoint wrapped for SHAP, with the settings it was trained with."""

    shap: shap.Explainer
    model: Any
    tokenizer: Any
    info: DatasetInfo
    name: str  # output file stem, e.g. distilbert_covid_seed42
    variant: str  # text variant the model was trained on
    max_length: int
    head_tokens: int | None
    predictions: Path | None = None  # the run's logged test probabilities, if any


# ---------------------------------------------------------------------------------------------
# The fixed example set


def stratified_examples(df: pd.DataFrame, n: int = NUM_EXAMPLES, seed: int = EXAMPLE_SEED):
    """``n`` rows of ``df`` with its label balance; the same rows on every machine."""
    if n >= len(df):
        return df.reset_index(drop=True)
    picked, _ = train_test_split(df, train_size=n, stratify=df["label"], random_state=seed)
    return picked.reset_index(drop=True)


def fixed_examples(dataset: str, variant: str = "raw", n: int = NUM_EXAMPLES) -> pd.DataFrame:
    """The fixed test examples of one dataset, in the given text variant.

    Rows are picked from the raw variant (stratified by label) and then looked up by id, so a
    raw and a clean explanation cover the same examples. ISOT picks from its 2,000-article LLM
    subset, so the Qwen3 explanations (T30) can use the same rows.
    """
    test = load_splits(dataset, "raw")["test"]
    if test["in_llm_subset"].any():
        test = test[test["in_llm_subset"]]
    picked = stratified_examples(test, n)
    if variant == "raw":
        return picked
    other = load_splits(dataset, variant)["test"].set_index("id")
    return other.loc[picked["id"]].reset_index()


# ---------------------------------------------------------------------------------------------
# Loading a checkpoint


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()


def resolve_checkpoint(checkpoint: str) -> tuple[str, dict | None]:
    """The model folder (or hub id) and the run's config.yaml, if there is one."""
    path = Path(checkpoint)
    if (path / "model").is_dir():
        run_dir, model_dir = path, path / "model"
    elif path.is_dir():
        run_dir, model_dir = path.parent, path
    elif path.exists() or os.sep in checkpoint or checkpoint.startswith(("outputs", ".")):
        raise FileNotFoundError(f"checkpoint folder {checkpoint!r} not found")
    else:
        return checkpoint, None  # a Hugging Face model id
    config = run_dir / "config.yaml"
    if not config.exists():
        return str(model_dir), None
    with open(config, encoding="utf-8") as f:
        return str(model_dir), yaml.safe_load(f)


def get_shap_explainer(
    checkpoint: str, dataset: str | None = None, device: str | None = None
) -> Explainer:
    """Load a checkpoint and wrap it in a SHAP partition explainer over its tokens."""
    # Load weights on the main thread (transformers 5's threaded loading crashed on Windows)
    os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
    model_path, cfg = resolve_checkpoint(checkpoint)
    data = (cfg or {}).get("data", {})
    dataset = dataset or data.get("dataset")
    if dataset is None:
        raise ValueError(f"{checkpoint!r} has no config.yaml; pass the dataset")
    info = get_dataset(dataset)
    if cfg is not None:
        name = f"{cfg['run']['name']}_seed{cfg['run']['seed']}"
        predictions = (
            PREDICTIONS_DIR / cfg["run"]["name"] / f"seed{cfg['run']['seed']}" / "test.csv"
        )
    else:
        name = f"{_slug(Path(model_path).name)}_{info.name}"
        predictions = None
    max_length = int(data.get("max_length") or info.max_length)
    head_tokens = data.get("head_tokens")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------------
    # Check if this checkpoint is an LSTM model (.pt file)
    # -------------------------------------------------------------------------
    pt_path = Path(model_path) if Path(model_path).suffix == ".pt" else Path(model_path) / "model.pt"

    if pt_path.exists():
        predictor = LSTMPredictor(pt_path, device=device) #[cite: 18]
        explainer = shap.Explainer(
            predictor.predict_proba, #[cite: 17, 18]
            shap.maskers.Text(), #[cite: 16]
            output_names=list(info.labels),
        )
        return Explainer(
            shap=explainer,
            model=predictor.model, #[cite: 18]
            tokenizer=None,
            info=info,
            name=name,
            variant=data.get("variant", "raw"),
            max_length=max_length,
            head_tokens=head_tokens,
            predictions=predictions,
        )

    # -------------------------------------------------------------------------
    # Transformer models (DistilBERT, RoBERTa, DeBERTa)
    # -------------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_path) #[cite: 16]
    model = AutoModelForSequenceClassification.from_pretrained(model_path, dtype=torch.float32) #[cite: 16]
    if model.config.num_labels != info.num_labels: #[cite: 16]
        raise ValueError( #[cite: 16]
            f"{model_path} has {model.config.num_labels} labels but {info.name} has "
            f"{info.num_labels}"
        )
    model.to(device).eval() #[cite: 16]

    @torch.no_grad()
    def predict(texts) -> np.ndarray:  # noqa: ANN001  (SHAP passes an array of strings)
        out = []
        texts = [str(t) for t in texts]
        for start in range(0, len(texts), BATCH_SIZE):
            ids = encode_texts(
                texts[start : start + BATCH_SIZE], tokenizer, max_length, head_tokens
            )
            batch = tokenizer.pad({"input_ids": ids}, return_tensors="pt").to(device)
            out.append(torch.softmax(model(**batch).logits.float(), dim=-1).cpu().numpy())
        return np.concatenate(out)

    explainer = shap.Explainer(
        predict, shap.maskers.Text(tokenizer), output_names=list(info.labels) #[cite: 16]
    )
    return Explainer(
        shap=explainer,
        model=model,
        tokenizer=tokenizer,
        info=info,
        name=name,
        variant=data.get("variant", "raw"),
        max_length=max_length,
        head_tokens=head_tokens,
        predictions=predictions,
    )


# ---------------------------------------------------------------------------------------------
# Explaining


def explain_claim(claim_text: str, explainer: Explainer, max_evals: int = MAX_EVALS) -> dict:
    """One claim for the app: per-token push towards "false" (positive) or "true" (negative).

    ``towards_false`` sums the token's values over the dataset's false labels (for LIAR:
    pants-fire, false and barely-true), the same grouping as the app's P(false).
    """
    sv = explainer.shap([claim_text], max_evals=max_evals)
    values = np.asarray(sv.values[0], dtype=float)  # (n_tokens, n_labels)
    probs = np.asarray(sv.base_values[0], dtype=float) + values.sum(axis=0)
    fake = explainer.info.fake_label_ids
    towards_false = values[:, fake].sum(axis=1)
    return {
        "text": claim_text,
        "labels": list(explainer.info.labels),
        "probs": probs.tolist(),
        "p_false": float(probs[fake].sum()),
        "base_values": np.asarray(sv.base_values[0], dtype=float).tolist(),
        "token_attributions": [
            {"token": str(tok), "values": v.tolist(), "towards_false": float(f)}
            for tok, v, f in zip(sv.data[0], values, towards_false, strict=True)
        ],
    }


def explain_dataset_batch(
    examples: pd.DataFrame,
    explainer: Explainer,
    output_dir: Path = EXPLANATIONS_DIR,
    stem: str | None = None,
    max_evals: int = MAX_EVALS,
) -> tuple[Path, Path]:
    """Explain a set of examples (columns id, text, label) and save the HTML and the arrays."""
    if examples.empty:
        raise ValueError("no examples to explain")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = stem or explainer.name
    texts = examples["text"].tolist()
    print(f"{stem}: SHAP on {len(texts)} examples (max_evals {max_evals})")
    sv = explainer.shap(texts, max_evals=max_evals)

    def ragged(items) -> np.ndarray:  # noqa: ANN001
        out = np.empty(len(items), dtype=object)
        out[:] = [np.asarray(x) for x in items]
        return out

    values = ragged([np.asarray(v, dtype=np.float32) for v in sv.values])
    base_values = np.asarray(sv.base_values, dtype=np.float32)
    npz_path = output_dir / f"{stem}_shap.npz"
    np.savez_compressed(
        npz_path,
        ids=np.asarray(examples["id"].astype(str)),
        texts=np.asarray(texts, dtype=object),
        labels=examples["label"].to_numpy(),
        tokens=ragged([np.asarray(t, dtype=str) for t in sv.data]),
        values=values,
        base_values=base_values,
        probs=base_values + np.stack([v.sum(axis=0) for v in values]),
        output_names=np.asarray(explainer.info.labels),
        fake_label_ids=np.asarray(explainer.info.fake_label_ids),
    )
    html_path = output_dir / f"{stem}_shap.html"
    html_path.write_text(str(shap.plots.text(sv, display=False)), encoding="utf-8")
    return html_path, npz_path


def check_logged_predictions(explainer: Explainer, examples: pd.DataFrame) -> float | None:
    """Largest gap between this checkpoint's probabilities and the ones in results/predictions.

    A run name and seed can be trained more than once (on the laptop, then on the cluster),
    and only the latest run's predictions are kept. A gap above bf16 noise means this
    checkpoint is not the run behind runs.csv, so its explanations do not match the results.
    """
    if explainer.predictions is None or not explainer.predictions.exists():
        return None
    logged = pd.read_csv(explainer.predictions, dtype={"id": str}).set_index("id")
    ids = examples["id"].astype(str)
    if not ids.isin(logged.index).all():
        return None
    columns = [f"prob_{label}" for label in explainer.info.labels]
    ours = explainer.shap.model(np.asarray(examples["text"].tolist()))
    gap = float(np.abs(ours - logged.loc[ids, columns].to_numpy()).max())
    if gap > 0.05:
        print(
            f"  WARNING: probabilities differ by up to {gap:.2f} from {explainer.predictions}; "
            "this checkpoint is not the run logged in runs.csv (use the checkpoint from the "
            "machine that ran it last)"
        )
    return gap


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="run folder (outputs/<name>/seed<N>), model folder or hub id",
    )
    parser.add_argument("--dataset", help="needed only when the checkpoint has no config.yaml")
    parser.add_argument("--variant", help="text variant to explain (default: the training one)")
    parser.add_argument("--num_samples", type=int, default=NUM_EXAMPLES)
    parser.add_argument("--max_evals", type=int, default=MAX_EVALS)
    parser.add_argument("--output_dir", type=Path, default=EXPLANATIONS_DIR)
    parser.add_argument("--device", help="e.g. cuda, cuda:1 or cpu (default: cuda if available)")
    args = parser.parse_args(argv)

    explainer = get_shap_explainer(args.checkpoint, args.dataset, args.device)
    variant = args.variant or explainer.variant
    stem = explainer.name if variant == explainer.variant else f"{explainer.name}_on_{variant}"
    examples = fixed_examples(explainer.info.name, variant, args.num_samples)
    if variant == explainer.variant:
        check_logged_predictions(explainer, examples)
    html_path, npz_path = explain_dataset_batch(
        examples, explainer, args.output_dir, stem, args.max_evals
    )
    print(f"  {html_path}\n  {npz_path}")


if __name__ == "__main__":
    main()
