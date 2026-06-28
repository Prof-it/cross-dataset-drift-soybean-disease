"""CLI: aggregate evaluation results into the paper's headline tables.

Reads ``eval_results.csv`` (written by the ``evaluate`` experiment) and produces
the transfer-gap summary: within- minus cross-dataset macro F1 per model, with
mean / std / n over the training and split seeds (the variance bands).

Example
-------
    python scripts/aggregate_results.py
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from src.config import Paths, load_config
from src.evaluation.bootstrap import aggregate_over_seeds


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate evaluation results into summary tables.")
    parser.add_argument("--config", default=None, help="optional experiment YAML override")
    parser.add_argument("--results-dir", default=None, help="override results location")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    paths = Paths.default().with_overrides(results_dir=args.results_dir)
    cfg = load_config(args.config, paths=paths) if args.config else load_config(paths=paths)
    results_dir = cfg.paths.results_dir
    eval_csv = results_dir / "eval_results.csv"
    if not eval_csv.exists():
        raise SystemExit(f"missing {eval_csv}; run the 'evaluate' experiment first")

    df = pd.read_csv(eval_csv)
    # The baseline and the training-time ablations all live in finetune-family
    # namespaces (finetune, finetune_unweighted, finetune_label_smoothing); keep
    # the namespace in the grouping so their transfer gaps are reported side by side.
    finetune = df[df["experiment"].astype(str).str.startswith("finetune")]
    if finetune.empty:
        raise SystemExit("eval_results.csv has no finetune-family rows; cannot form the gap")
    index = ["experiment", "arch", "path", "train_dataset", "seed", "split_seed"]
    pivot = finetune.pivot_table(index=index, columns="direction", values="macro_f1").reset_index()
    if "within" not in pivot or "cross" not in pivot:
        raise SystemExit("eval_results.csv lacks both within and cross rows; cannot form the gap")
    pivot["transfer_gap"] = pivot["within"] - pivot["cross"]

    summary = aggregate_over_seeds(
        pivot, ["experiment", "arch", "path", "train_dataset"], ["within", "cross", "transfer_gap"]
    )
    out = results_dir / "summary_transfer_gap.csv"
    summary.to_csv(out, index=False)
    logging.info("wrote %s", out)


if __name__ == "__main__":
    main()
