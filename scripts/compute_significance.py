"""CLI: compute nonparametric significance tests for the paper's comparison claims.

Reads ``eval_results.csv`` under the configured results dir and writes
``significance.csv`` (one row per comparison x scope: source asymmetry, CNN vs ViT,
direct vs two-stage, and the training-time / post-hoc interventions, each pooled
and per transfer direction). Comparisons whose rows are absent are skipped.

Run locally (``python scripts/compute_significance.py``) or, on Colab, import and
call ``run(cfg)`` with the resolved config:

    from src.config import load_config
    from scripts.compute_significance import run
    run(load_config(paths=paths))
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from src.config import Paths, load_config
from src.evaluation.significance import pairwise_tests


def run(cfg) -> None:
    results = cfg.paths.results_dir
    eval_csv = results / "eval_results.csv"
    if not eval_csv.exists():
        raise SystemExit(f"missing {eval_csv}; run the 'evaluate' experiment first")
    df = pairwise_tests(pd.read_csv(eval_csv))
    if df.empty:
        logging.info("no comparisons could be computed from %s", eval_csv)
        return
    out = results / "significance.csv"
    df.to_csv(out, index=False)
    logging.info("wrote %s (%d rows)", out, len(df))
    for _, r in df.iterrows():
        logging.info("  %-50s [%-5s] p=%.4g  effect=%.3f",
                     r["comparison"], r.get("scope", ""), r["p_value"], r.get("effect_size", float("nan")))


def main() -> None:
    parser = argparse.ArgumentParser(description="Significance tests for the paper's comparison claims.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--results-dir", default=None, help="override results location (CSVs in, CSV out)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    paths = Paths.default().with_overrides(results_dir=args.results_dir)
    cfg = load_config(args.config, paths=paths) if args.config else load_config(paths=paths)
    run(cfg)


if __name__ == "__main__":
    main()
