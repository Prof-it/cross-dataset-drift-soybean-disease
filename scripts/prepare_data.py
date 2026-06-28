"""CLI: build the data-partition artifacts.

A one-shot, deterministic step that turns the raw folders into the inputs the
experiments consume: stratified split CSVs (per dataset, per split seed), the
class-weights JSON, and optionally the matched ASDID subsample for the control
study. Idempotent: existing CSVs are reused unless ``--overwrite`` is passed.

Examples
--------
    python scripts/prepare_data.py --matched
    python scripts/prepare_data.py --datasets asdid mh --overwrite
"""

from __future__ import annotations

import argparse
import logging

from src.config import load_config
from src.data import splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Build split CSVs, class weights, control subsample.")
    parser.add_argument("--config", default=None, help="optional experiment YAML override")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="datasets to build (default: source datasets + plantvillage)")
    parser.add_argument("--matched", action="store_true",
                        help="also build the ASDID->MH matched subsample (control study)")
    parser.add_argument("--overwrite", action="store_true", help="rebuild existing artifacts")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(args.config) if args.config else load_config()
    soy_datasets = args.datasets or list(cfg.source_datasets)

    for split_seed in cfg.split_seeds:
        for dataset in soy_datasets:
            path = splits.build_splits(cfg, dataset, split_seed, overwrite=args.overwrite)
            logging.info("splits  seed %-4s %-12s -> %s", split_seed, dataset, path)
        if args.matched:
            path = splits.build_matched_subsample(cfg, split_seed, overwrite=args.overwrite)
            logging.info("matched seed %-4s asdid_matched -> %s", split_seed, path)

    # PlantVillage uses one fixed split (two-stage pretraining), reused across soybean splits.
    if args.datasets is None and "twostage" in cfg.training_paths:
        pv = splits.build_splits(cfg, "plantvillage", cfg.plantvillage_split_seed, overwrite=args.overwrite)
        logging.info("splits  pv      %-4s plantvillage -> %s", cfg.plantvillage_split_seed, pv)

    primary = cfg.split_seeds[0]
    weights = {
        dataset: splits.compute_class_weights(splits.load_split(cfg, dataset, primary), cfg.data.class_names)
        for dataset in cfg.source_datasets
    }
    splits.write_class_weights(cfg, weights)
    logging.info("class weights -> %s", cfg.paths.class_weights_path)


if __name__ == "__main__":
    main()
