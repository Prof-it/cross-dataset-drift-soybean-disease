"""CLI: ECE robustness across bin counts for the finetune models.

Loads each finetune checkpoint, collects predictions on the within- and
cross-dataset test sets, and computes ECE at several equal-width bin counts, to
confirm the calibration asymmetry is not an artefact of the 10-bin default. Writes
``ece_robustness.csv``.

Needs the trained checkpoints, so run on Colab/GPU. In the notebook, after the
``evaluate`` run, call ``run(cfg)`` with the resolved config:

    from src.config import load_config
    from scripts.compute_ece_robustness import run
    run(load_config(paths=paths))
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd
import torch

from src.config import Paths, get_device, load_config, model_id
from src.data.loaders import make_dataloaders
from src.data.splits import build_splits
from src.evaluation.metrics import collect_predictions, ece_across_bins
from src.experiments._common import run_dir, seed_pairs
from src.models.factory import create_model

BIN_COUNTS = (5, 10, 15, 20)


def _load(cfg, arch, ckpt, device):
    model = create_model(arch, cfg.data.num_classes, cfg.architectures, pretrained=False)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    return model.to(device)


def run(cfg) -> None:
    device = get_device()
    for split_seed in cfg.split_seeds:  # splits are generated on demand (as the experiments do)
        for dataset in cfg.source_datasets:
            build_splits(cfg, dataset, split_seed)
    rows = []
    for split_seed, seed in seed_pairs(cfg):
        for arch in cfg.architectures:
            for path in cfg.training_paths:
                for train_ds in cfg.source_datasets:
                    ckpt = run_dir(cfg, "finetune", model_id(arch, path, train_ds), split_seed, seed) / "best_model.pt"
                    if not ckpt.exists():
                        continue
                    model = _load(cfg, arch, ckpt, device)
                    for eval_ds in cfg.source_datasets:
                        direction = "within" if eval_ds == train_ds else "cross"
                        preds = collect_predictions(model, make_dataloaders(cfg, eval_ds, split_seed, seed)["test"], device)
                        rows.append({
                            "arch": arch, "path": path, "train_dataset": train_ds, "eval_dataset": eval_ds,
                            "direction": direction, "split_seed": split_seed, "seed": seed,
                            **ece_across_bins(preds.y_true, preds.probs, BIN_COUNTS),
                        })
    if not rows:
        raise SystemExit("no finetune checkpoints found; run the finetune experiment first")

    out = pd.DataFrame(rows)
    out.to_csv(cfg.paths.results_dir / "ece_robustness.csv", index=False)
    logging.info("wrote %s (%d rows)", cfg.paths.results_dir / "ece_robustness.csv", len(out))
    cols = [f"ece_b{n}" for n in BIN_COUNTS]
    logging.info("\nmean ECE by source x direction and bin count:\n%s",
                 out.groupby(["train_dataset", "direction"])[cols].mean().to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="ECE robustness across bin counts.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    paths = Paths.default().with_overrides(results_dir=args.results_dir)
    cfg = load_config(args.config, paths=paths) if args.config else load_config(paths=paths)
    run(cfg)


if __name__ == "__main__":
    main()
