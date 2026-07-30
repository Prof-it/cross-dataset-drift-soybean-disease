"""CLI: run a named experiment from a config.

The single entry point for training and evaluation. It reads an experiment YAML
under ``configs/experiments/``, which declares which experiment to run via an
``experiment:`` key, merges it onto the base config, and dispatches.

Example
-------
    python scripts/run_experiment.py configs/experiments/full_finetune.yaml
"""

from __future__ import annotations

import argparse
import importlib
import logging

import yaml

from src.config import Paths, load_config

EXPERIMENTS = {
    "finetune": "src.experiments.finetune",
    "control_study": "src.experiments.robustness.control_study",
    "linear_probe": "src.experiments.robustness.linear_probe",
    "linear_solvability": "src.experiments.robustness.linear_solvability",
    "background_intervention": "src.experiments.robustness.background_intervention",
    "frequency_intervention": "src.experiments.robustness.frequency_intervention",
    "evaluate": "src.experiments.robustness.evaluate",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a named experiment from an experiment YAML.")
    parser.add_argument("config", help="path to an experiment YAML under configs/experiments/")
    # Path overrides, e.g. to evaluate Colab-trained checkpoints locally:
    #   run_experiment.py configs/experiments/evaluate.yaml --checkpoints-dir <drive>/checkpoints
    parser.add_argument("--checkpoints-dir", default=None, help="override checkpoint location")
    parser.add_argument("--data-root", default=None, help="override raw-data location")
    parser.add_argument("--results-dir", default=None, help="override results output location")
    parser.add_argument("--logs-dir", default=None, help="override logs location")
    parser.add_argument("--masks-dir", default=None,
                        help="override foreground-mask location (background intervention)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    raw = yaml.safe_load(open(args.config, encoding="utf-8")) or {}
    name = raw.get("experiment")
    if name not in EXPERIMENTS:
        raise SystemExit(
            f"config {args.config!r} must set 'experiment' to one of {sorted(EXPERIMENTS)}"
        )

    paths = Paths.default().with_overrides(
        checkpoints_dir=args.checkpoints_dir,
        data_root=args.data_root,
        results_dir=args.results_dir,
        logs_dir=args.logs_dir,
        masks_dir=args.masks_dir,
    )
    cfg = load_config(args.config, paths=paths)
    module = importlib.import_module(EXPERIMENTS[name])
    if not hasattr(module, "run"):
        raise SystemExit(f"experiment '{name}' is not implemented yet")
    module.run(cfg)


if __name__ == "__main__":
    main()
