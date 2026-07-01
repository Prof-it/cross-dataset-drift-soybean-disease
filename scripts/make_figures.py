"""CLI: render the standard paper figure set from the evaluation CSVs.

Reads the tables written by the ``evaluate`` experiment under ``results_dir`` and
writes styled PDFs to ``results_dir/figures/``. Figures whose inputs are missing
are skipped. Not every figure goes into the paper, but all share one style.

Example
-------
    python scripts/make_figures.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.config import Paths, load_config
from src.viz import figures
from src.viz.style import set_style


def _read(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the standard figure set from result CSVs.")
    parser.add_argument("--config", default=None, help="optional experiment YAML override")
    parser.add_argument("--results-dir", default=None, help="override results location (CSVs in, figures out)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    paths = Paths.default().with_overrides(results_dir=args.results_dir)
    cfg = load_config(args.config, paths=paths) if args.config else load_config(paths=paths)
    set_style(cfg)
    results = cfg.paths.results_dir
    eval_results = _read(results / "eval_results.csv")
    if eval_results is None:
        raise SystemExit(f"missing {results / 'eval_results.csv'}; run the 'evaluate' experiment first")
    reliability = _read(results / "reliability.csv")
    decomp = _read(results / "decomposition.csv")
    confusions = _read(results / "confusions.csv")
    out = results / "figures"

    produced = [
        figures.transfer_dumbbell(cfg, eval_results, out),
        figures.calibration_by_dataset(cfg, eval_results, reliability, out),
        figures.intervention_recovery(cfg, eval_results, out),
        figures.per_class_f1(cfg, eval_results, out),
        figures.decomposition(cfg, decomp, out),
        figures.confusion_matrices(cfg, confusions, out),
    ]
    for path in produced:
        if path is not None:
            logging.info("wrote %s", path)


if __name__ == "__main__":
    main()
