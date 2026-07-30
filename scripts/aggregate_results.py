"""CLI: aggregate evaluation results into the paper's headline tables.

Reads ``eval_results.csv`` (written by the ``evaluate`` experiment) and produces the
transfer-gap summaries with mean / std / n and a 95% BCa bootstrap confidence
interval over the runs, at the two levels the paper quotes:

- ``summary_transfer_gap.csv``            per configuration (architecture x path x
  source dataset), i.e. over the nine seed runs;
- ``summary_transfer_gap_by_source.csv``  pooled per transfer direction, which is
  the level the headline asymmetry (ASDID versus MH) is stated at.

Resampling is over runs, so an interval expresses variation across initialization
seeds and data partitions rather than sampling error inside one test split. The
pooled table aggregates configurations that share a backbone family and pretraining
corpus, so its runs are not mutually independent and its interval should be read as
descriptive spread rather than as a strict test; the formal comparison is the
Mann-Whitney test in ``compute_significance.py``.

Example
-------
    python scripts/aggregate_results.py
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from src.config import Paths, load_config
from src.evaluation.bootstrap import aggregate_with_ci

GAP_METRICS = ["within", "cross", "transfer_gap"]


def _gap_per_run(df: pd.DataFrame) -> pd.DataFrame:
    """Within- minus cross-dataset macro F1, one row per configuration and run."""
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
    return pivot


def _control_gap_per_run(df: pd.DataFrame) -> "pd.DataFrame | None":
    """Matched-control gap per run: within (matched subsample) minus cross (target)."""
    control = df[df["experiment"].astype(str) == "control_study"]
    if control.empty:
        return None
    index = ["experiment", "arch", "path", "seed", "split_seed"]
    pivot = control.pivot_table(index=index, columns="direction", values="macro_f1").reset_index()
    if "within" not in pivot or "cross" not in pivot:
        return None
    pivot["transfer_gap"] = pivot["within"] - pivot["cross"]
    return pivot


def run(cfg) -> None:
    results_dir = cfg.paths.results_dir
    eval_csv = results_dir / "eval_results.csv"
    if not eval_csv.exists():
        raise SystemExit(f"missing {eval_csv}; run the 'evaluate' experiment first")

    per_run = _gap_per_run(pd.read_csv(eval_csv))
    iterations = cfg.evaluation.bootstrap_iterations
    seed = cfg.split_seeds[0]  # fixed so the resampling is reproducible

    outputs = {
        "summary_transfer_gap.csv": ["experiment", "arch", "path", "train_dataset"],
        "summary_transfer_gap_by_source.csv": ["experiment", "train_dataset"],
    }
    for name, group_cols in outputs.items():
        table = aggregate_with_ci(
            per_run, group_cols, GAP_METRICS, n_iterations=iterations, seed=seed
        )
        table.to_csv(results_dir / name, index=False)
        logging.info("wrote %s (%d rows)", results_dir / name, len(table))

    # The control study lives in its own namespace and its within-dataset side is
    # the matched subsample, so it cannot be folded into the finetune pivot above.
    # The paper quotes its gap and interval, so it gets its own table.
    control = _control_gap_per_run(pd.read_csv(eval_csv))
    if control is not None:
        table = aggregate_with_ci(
            control, ["experiment"], GAP_METRICS, n_iterations=iterations, seed=seed
        )
        table.to_csv(results_dir / "summary_control_gap.csv", index=False)
        logging.info("wrote %s (%d rows)", results_dir / "summary_control_gap.csv", len(table))

    headline = aggregate_with_ci(
        per_run[per_run["experiment"] == "finetune"], ["train_dataset"], GAP_METRICS,
        n_iterations=iterations, seed=seed,
    )
    show = ["train_dataset", "transfer_gap_n", "transfer_gap_mean",
            "transfer_gap_ci_low", "transfer_gap_ci_high"]
    logging.info(
        "\nheadline transfer gap by source dataset (95%% BCa over runs):\n%s",
        headline[show].round(4).to_string(index=False),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate evaluation results into summary tables."
    )
    parser.add_argument("--config", default=None, help="optional experiment YAML override")
    parser.add_argument("--results-dir", default=None, help="override results location")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    paths = Paths.default().with_overrides(results_dir=args.results_dir)
    cfg = load_config(args.config, paths=paths) if args.config else load_config(paths=paths)
    run(cfg)


if __name__ == "__main__":
    main()
