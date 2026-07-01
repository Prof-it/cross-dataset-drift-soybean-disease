"""CLI: confusion matrices for the finetune models (within and cross).

Loads each finetune checkpoint, collects predictions on the within- and
cross-dataset test sets, and accumulates the 3x3 confusion counts (summed over
the nine runs) per configuration. This makes the per-class story concrete --
in particular which class frogeye leaf spot is mistaken for under cross-dataset
transfer. Writes ``confusions.csv`` in long format:
``arch, path, train_dataset, eval_dataset, direction, true_class, pred_class, count``.

Needs the trained checkpoints, so run on Colab/GPU. In the notebook, after the
``evaluate`` run, call ``run(cfg)`` with the resolved config:

    from src.config import load_config
    from scripts.compute_confusions import run
    run(load_config(paths=paths))
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from src.config import Paths, get_device, load_config, model_id
from src.data.loaders import make_dataloaders
from src.data.splits import build_splits
from src.evaluation.metrics import collect_predictions, confusion
from src.experiments._common import run_dir, seed_pairs
from src.models.factory import create_model


def _load(cfg, arch, ckpt, device):
    model = create_model(arch, cfg.data.num_classes, cfg.architectures, pretrained=False)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    return model.to(device)


def run(cfg) -> None:
    device = get_device()
    for split_seed in cfg.split_seeds:  # splits are generated on demand (as the experiments do)
        for dataset in cfg.source_datasets:
            build_splits(cfg, dataset, split_seed)
    names = cfg.data.class_names
    k = len(names)
    totals: dict[tuple, np.ndarray] = defaultdict(lambda: np.zeros((k, k), dtype=int))
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
                        raw, _ = confusion(preds.y_true, preds.y_pred, k)
                        totals[(arch, path, train_ds, eval_ds, direction)] += raw
    if not totals:
        raise SystemExit("no finetune checkpoints found; run the finetune experiment first")

    rows = []
    for (arch, path, train_ds, eval_ds, direction), mat in totals.items():
        for i, true_c in enumerate(names):
            for j, pred_c in enumerate(names):
                rows.append({
                    "arch": arch, "path": path, "train_dataset": train_ds, "eval_dataset": eval_ds,
                    "direction": direction, "true_class": true_c, "pred_class": pred_c,
                    "count": int(mat[i, j]),
                })
    out = pd.DataFrame(rows)
    out.to_csv(cfg.paths.results_dir / "confusions.csv", index=False)
    logging.info("wrote %s (%d rows)", cfg.paths.results_dir / "confusions.csv", len(out))

    # headline: cross-direction, row-normalized, summed over arch and path
    cross = out[out.direction == "cross"]
    for train_ds in cfg.source_datasets:
        sub = cross[cross.train_dataset == train_ds]
        if sub.empty:
            continue
        mat = (sub.pivot_table(index="true_class", columns="pred_class", values="count", aggfunc="sum")
               .reindex(index=list(names), columns=list(names)).fillna(0.0))
        norm = mat.div(mat.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0).round(3)
        logging.info("\n%s -> other (row-normalized confusion):\n%s", train_ds, norm.to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Confusion matrices (within and cross).")
    parser.add_argument("--config", default=None)
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    paths = Paths.default().with_overrides(results_dir=args.results_dir)
    cfg = load_config(args.config, paths=paths) if args.config else load_config(paths=paths)
    run(cfg)


if __name__ == "__main__":
    main()
