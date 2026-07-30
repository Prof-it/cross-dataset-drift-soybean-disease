"""CLI: paper-facing summary and tests for the input-level interventions.

Reads ``background_intervention.csv`` and ``frequency_intervention.csv`` (written
by the corresponding experiments) and produces the two numbers the manuscript
reports, with the tests behind them.

Background. For each condition, the mean change in macro F1 from the unedited
image, split by transfer direction, plus a Wilcoxon signed-rank test over the
per-run changes -- paired because the same checkpoint is scored before and after
the edit. Reported per source-dataset block and pooled, mirroring the thesis
table.

Frequency. For each sigma, the within- and cross-dataset drop in macro F1, and a
paired test of the *difference* between them. The question is not whether blur
hurts (it does, mildly) but whether it hurts the cross-dataset direction more --
that is what would make high-frequency texture a non-transferring cue.

``--eligibility`` selects which images the metrics were computed over (see the
``eligibility`` column written by the experiments). ``heldout`` is the default and
the one to report: it excludes images that were in a given run's training split,
which matters for the background experiment because the annotated subset predates
this repository's split seeds.

Usage
-----
    python scripts/compute_intervention_stats.py
    python scripts/compute_intervention_stats.py --results-dir /path/to/results --eligibility test
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.data.interventions import ORIGINAL, sigma_of
from src.evaluation.significance import wilcoxon_signed_rank

logger = logging.getLogger(__name__)

# One row per checkpoint run; the unit the paired tests operate on.
RUN_KEYS = ["arch", "path", "train_dataset", "split_seed", "seed"]


def _load(results_dir: Path, stem: str, eligibility: str) -> pd.DataFrame | None:
    path = results_dir / f"{stem}.csv"
    if not path.exists():
        logger.warning("missing %s; skipping", path)
        return None
    frame = pd.read_csv(path)
    frame = frame[frame["eligibility"] == eligibility]
    if frame.empty:
        logger.warning("%s has no rows with eligibility=%r; skipping", path, eligibility)
        return None
    return frame


def _paired_change(frame: pd.DataFrame, condition: str, direction: str) -> pd.DataFrame:
    """Per-run macro F1 before and after ``condition`` for one transfer direction."""
    subset = frame[frame["direction"] == direction]
    base = subset[subset["condition"] == ORIGINAL][RUN_KEYS + ["macro_f1", "n_images"]]
    edited = subset[subset["condition"] == condition][RUN_KEYS + ["macro_f1"]]
    merged = base.merge(edited, on=RUN_KEYS, suffixes=("_original", "_edited"))
    return merged.assign(change=merged["macro_f1_edited"] - merged["macro_f1_original"])


def background_stats(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean change in macro F1 per condition x direction, pooled and per block."""
    conditions = [c for c in frame["condition"].unique() if c != ORIGINAL]
    rows: list[dict] = []
    for condition in sorted(conditions):
        for direction in ("within", "cross"):
            changes = _paired_change(frame, condition, direction)
            if changes.empty:
                continue
            for block in ("all", *sorted(changes["train_dataset"].unique())):
                sub = changes if block == "all" else changes[changes["train_dataset"] == block]
                test = wilcoxon_signed_rank(
                    sub["macro_f1_edited"].to_numpy(), sub["macro_f1_original"].to_numpy()
                )
                rows.append({
                    "condition": condition, "direction": direction, "block": block,
                    "n_runs": int(len(sub)),
                    "mean_n_images": float(sub["n_images"].mean()),
                    "macro_f1_original": float(sub["macro_f1_original"].mean()),
                    "macro_f1_edited": float(sub["macro_f1_edited"].mean()),
                    "mean_change": float(sub["change"].mean()),
                    "median_change": float(sub["change"].median()),
                    "p_value": test["p_value"],
                    "effect_size": test["effect_size"],
                })
    return pd.DataFrame(rows)


def frequency_stats(frame: pd.DataFrame) -> pd.DataFrame:
    """Within/cross drop per sigma, and a paired test of the difference between them."""
    conditions = [c for c in frame["condition"].unique() if c != ORIGINAL]
    rows: list[dict] = []
    for condition in sorted(conditions, key=sigma_of):
        within = _paired_change(frame, condition, "within")
        cross = _paired_change(frame, condition, "cross")
        if within.empty or cross.empty:
            continue
        # A drop is a fall in macro F1, so negate the signed change.
        paired = within.merge(cross, on=RUN_KEYS, suffixes=("_within", "_cross"))
        for block in ("all", *sorted(paired["train_dataset"].unique())):
            sub = paired if block == "all" else paired[paired["train_dataset"] == block]
            test = wilcoxon_signed_rank(
                (-sub["change_cross"]).to_numpy(), (-sub["change_within"]).to_numpy()
            )
            rows.append({
                "condition": condition, "sigma": sigma_of(condition), "block": block,
                "n_runs": int(len(sub)),
                "within_drop": float(-sub["change_within"].mean()),
                "cross_drop": float(-sub["change_cross"].mean()),
                "excess_cross_drop": float((sub["change_within"] - sub["change_cross"]).mean()),
                "p_value": test["p_value"],
                "effect_size": test["effect_size"],
            })
    return pd.DataFrame(rows)


def run(results_dir: Path, eligibility: str = "heldout") -> None:
    outputs: dict[str, pd.DataFrame] = {}

    background = _load(results_dir, "background_intervention", eligibility)
    if background is not None:
        outputs["background_intervention_stats"] = background_stats(background)

    # Blur uses the full test split, which is held out by construction, so the
    # eligibility policies coincide; read 'test' whichever was requested.
    frequency = _load(results_dir, "frequency_intervention", "test")
    if frequency is not None:
        outputs["frequency_intervention_stats"] = frequency_stats(frequency)

    if not outputs:
        raise SystemExit(
            f"no intervention results under {results_dir}; run the "
            f"background_intervention / frequency_intervention experiments first"
        )
    for stem, table in outputs.items():
        path = results_dir / f"{stem}.csv"
        table.to_csv(path, index=False)
        logger.info(
            "wrote %s (%d rows)\n%s", path, len(table), table.round(4).to_string(index=False)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the input-level interventions.")
    parser.add_argument("--results-dir", default=None, help="override results location")
    parser.add_argument("--eligibility", default="heldout", choices=("all", "heldout", "test"),
                        help="which images the background metrics are computed over")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from src.config import Paths

    results_dir = Path(args.results_dir) if args.results_dir else Paths.default().results_dir
    run(results_dir, args.eligibility)


if __name__ == "__main__":
    main()
