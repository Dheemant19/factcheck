"""SHAP explanations for a QLoRA Qwen3-8B run (owners M1 + M2, T30).

M4's ``factcheck.explain`` loads full-precision encoders; a QLoRA run is a 4-bit base model
plus adapters (``load_qlora``), so this wraps it in the same ``Explainer`` and writes the same
files, on the same fixed 50 test examples, as every other model:

    python -m factcheck.models.llm_explain --checkpoint outputs/qwen3_8b_qlora_liar/seed42

Outputs go to ``outputs/explanations/<run>_seed<N>_shap.{html,npz}`` (format: see
``factcheck.explain``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import shap
import torch
import yaml

from factcheck.data import get_dataset
from factcheck.explain import (
    EXPLANATIONS_DIR,
    MAX_EVALS,
    NUM_EXAMPLES,
    Explainer,
    check_logged_predictions,
    explain_dataset_batch,
    fixed_examples,
)
from factcheck.models.encoder import encode_texts, load_qlora
from factcheck.paths import PREDICTIONS_DIR

BATCH_SIZE = 32


def qlora_explainer(run_dir: str | Path, batch_size: int = BATCH_SIZE) -> Explainer:
    """A trained QLoRA run (``outputs/<name>/seed<N>``) wrapped in a SHAP partition explainer
    over its tokens, scoring texts exactly as in training (truncation, bf16)."""
    run_dir = Path(run_dir)
    with open(run_dir / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    data, run = cfg["data"], cfg["run"]
    info = get_dataset(data["dataset"])
    max_length = int(data.get("max_length") or info.max_length)
    head_tokens = data.get("head_tokens")
    model, tokenizer = load_qlora(run_dir / "model")

    @torch.no_grad()
    def predict(texts) -> np.ndarray:  # noqa: ANN001  (SHAP passes an array of strings)
        out = []
        texts = [str(t) for t in texts]
        for start in range(0, len(texts), batch_size):
            ids = encode_texts(
                texts[start : start + batch_size], tokenizer, max_length, head_tokens
            )
            batch = tokenizer.pad({"input_ids": ids}, return_tensors="pt").to(model.device)
            with torch.autocast(model.device.type, dtype=torch.bfloat16):
                logits = model(**batch).logits
            out.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        return np.concatenate(out)

    return Explainer(
        shap=shap.Explainer(predict, shap.maskers.Text(tokenizer), output_names=list(info.labels)),
        model=model,
        tokenizer=tokenizer,
        info=info,
        name=f"{run['name']}_seed{run['seed']}",
        variant=data.get("variant", "raw"),
        max_length=max_length,
        head_tokens=head_tokens,
        predictions=PREDICTIONS_DIR / run["name"] / f"seed{run['seed']}" / "test.csv",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, help="run folder, outputs/<name>/seed<N>")
    parser.add_argument("--num_samples", type=int, default=NUM_EXAMPLES)
    parser.add_argument("--max_evals", type=int, default=MAX_EVALS)
    parser.add_argument("--output_dir", type=Path, default=EXPLANATIONS_DIR)
    args = parser.parse_args(argv)

    explainer = qlora_explainer(args.checkpoint)
    examples = fixed_examples(explainer.info.name, explainer.variant, args.num_samples)
    gap = check_logged_predictions(explainer, examples)
    if gap is not None:
        print(f"  largest gap to the logged test probabilities: {gap:.4f}")
    html_path, npz_path = explain_dataset_batch(
        examples, explainer, args.output_dir, max_evals=args.max_evals
    )
    print(f"  {html_path}\n  {npz_path}")


if __name__ == "__main__":
    main()
