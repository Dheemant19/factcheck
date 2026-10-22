"""One training entry point: python -m factcheck.train --config configs/x.yaml (owners M1 + M2, T06).

    python -m factcheck.train --config configs/lstm/lstm_liar.yaml
    python -m factcheck.train --config configs/lstm/lstm_liar.yaml --seeds 13 42 2026

The config's ``model.family`` picks the trainer: ``lstm`` (T05), ``encoder`` (T06) or ``llm``
(T07, prompted scoring, no training yet). Every run logs its scores to results/runs.csv.
"""

from __future__ import annotations

import argparse
import copy

import yaml


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run(cfg: dict) -> dict:
    """Train one run (one seed) and return its summary."""
    family = cfg["model"]["family"]
    if family == "lstm":
        from factcheck.models.lstm import train_lstm

        return train_lstm(cfg)
    if family == "encoder":
        from factcheck.models.encoder import train_encoder

        return train_encoder(cfg)
    if family == "llm":
        from factcheck.models.llm import score_llm

        return score_llm(cfg)
    raise ValueError(f"model.family must be lstm, encoder or llm, not {family!r}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config", required=True, help="YAML run config, e.g. configs/lstm/lstm_liar.yaml"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", help="run once per seed (overrides run.seed)"
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    for seed in args.seeds or [cfg["run"]["seed"]]:
        seed_cfg = copy.deepcopy(cfg)
        seed_cfg["run"]["seed"] = seed
        seed_cfg["run"]["config"] = args.config  # logged in runs.csv (spec S10)
        run(seed_cfg)


if __name__ == "__main__":
    main()
