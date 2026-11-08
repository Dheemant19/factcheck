"""Score PUBHEALTH-trained models on Climate-FEVER, with no Climate-FEVER training (T38, E8,
owner M1).

    python scripts/transfer_climate_fever.py outputs/deberta_v3_base_pubhealth_tuned_evidence/seed13/model \\
        outputs/scibert_pubhealth/seed42/model outputs/tfidf_lr_pubhealth/seed42/model

Climate-FEVER has only a test split, so no model can be trained on it; this is the extrapolation
test of plan E8. PUBHEALTH is the closest source (claims checked against evidence) and its four
labels map one to one onto Climate-FEVER's:

    true -> supports    false -> refutes    mixture -> disputed    unproven -> not_enough_info

A run trained with evidence reads each claim with Climate-FEVER's own 5 evidence sentences
(most similar first, as in the RAG run); the others read the claim alone. Each source is logged
to results/runs.csv as ``<source run>_to_climate_fever`` (method ``transfer``), so it sits next
to the Qwen3-8B zero-shot and RAG rows with the same metrics.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from factcheck import serve
from factcheck.data import get_dataset
from factcheck.evaluate import log_run
from factcheck.inputs import load_inputs

TARGET = "climate_fever"
LABEL_MAP = {  # PUBHEALTH label -> Climate-FEVER label
    "true": "supports",
    "false": "refutes",
    "mixture": "disputed",
    "unproven": "not_enough_info",
}


def transfer(model_dir: str) -> dict:
    model = serve.load(model_dir)
    if model.info.name != "pubhealth":
        raise ValueError(f"{model_dir} is a {model.info.name} run; the label map is PUBHEALTH's")
    target = get_dataset(TARGET)
    columns = [model.info.labels.index(src) for src in LABEL_MAP]  # source column per target label
    order = [target.labels.index(dst) for dst in LABEL_MAP.values()]
    cfg = {
        "data": {"dataset": TARGET, "variant": "raw"},
        "features": {
            "use_evidence": model.uses_evidence,
            "evidence_k": 5,
            "evidence_source": "own",
        },
    }
    test = load_inputs(cfg)["test"]
    start = time.perf_counter()
    if model.uses_evidence:  # each claim with its own evidence
        source = np.vstack(
            [model.proba([c], [e]) for c, e in zip(test["text"], test["text_pair"], strict=True)]
        )
    else:
        source = np.vstack(
            [model.proba(list(test["text"][i : i + 64])) for i in range(0, len(test), 64)]
        )
    seconds = time.perf_counter() - start
    probs = np.zeros((len(test), target.num_labels))
    probs[:, order] = source[:, columns]
    src_cfg = model.cfg
    run = {
        "run": {
            "name": f"{src_cfg['run']['name']}_to_{TARGET}",
            "seed": int(src_cfg["run"]["seed"]),
            "config": f"scripts/transfer_climate_fever.py {Path(model_dir).as_posix()}",
        },
        "data": {"dataset": TARGET, "variant": "raw"},
        "model": {
            "family": src_cfg["model"]["family"],
            "name": src_cfg["model"]["name"],
            "method": "transfer",
        },
    }
    print(f"{run['run']['name']}: {len(test):,} claims in {seconds:.0f} s")
    rows = log_run(run, target, {"test": (test, probs)}, {"latency_ms": seconds / len(test) * 1000})
    return rows[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "models", nargs="+", help="PUBHEALTH run folders: outputs/<run>/seed<N>/model"
    )
    for model_dir in parser.parse_args().models:
        transfer(model_dir)


if __name__ == "__main__":
    main()
