"""CLI: few-shot target-adaptation curve (data efficiency of the head refit).

For each architecture and transfer direction, takes the source-fine-tuned (direct)
backbone, caches the target train and test features once, then refits a linear head
on ``n`` labeled target images per class for ``n`` in a grid, and records
target-test macro F1. This turns the paper's head-refit "upper bound" into a curve
that answers how little target data is needed. Writes ``few_shot.csv``.

Needs the trained checkpoints, so run on Colab/GPU. In the notebook, call ``run(cfg)``:

    from src.config import load_config
    from scripts.compute_few_shot import run
    run(load_config(paths=paths))
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
import torch

from src.config import Paths, get_device, load_config, model_id
from src.data.loaders import make_eval_loader
from src.data.splits import build_splits
from src.evaluation.few_shot import fit_eval_linear, subsample_indices
from src.experiments._common import run_dir, seed_pairs
from src.models.factory import backbone_features, create_model

N_GRID = (5, 10, 25, 50, None)  # None = all available target training data


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
            for source in cfg.source_datasets:
                ckpt = run_dir(cfg, "finetune", model_id(arch, "direct", source), split_seed, seed) / "best_model.pt"
                if not ckpt.exists():
                    continue
                model = _load(cfg, arch, ckpt, device)
                for target in (d for d in cfg.source_datasets if d != source):
                    tr_x, tr_y = backbone_features(model, arch, make_eval_loader(cfg, target, split_seed, "train"), device)
                    te_x, te_y = backbone_features(model, arch, make_eval_loader(cfg, target, split_seed, "test"), device)
                    tr_x, tr_y, te_x, te_y = tr_x.numpy(), tr_y.numpy(), te_x.numpy(), te_y.numpy()
                    for n in N_GRID:
                        idx = subsample_indices(tr_y, n, np.random.default_rng(seed))
                        f1 = fit_eval_linear(tr_x[idx], tr_y[idx], te_x, te_y)
                        rows.append({
                            "arch": arch, "source": source, "target": target,
                            "n_per_class": "all" if n is None else n,
                            "split_seed": split_seed, "seed": seed, "macro_f1": f1,
                        })
    if not rows:
        raise SystemExit("no direct finetune checkpoints found; run the finetune experiment first")

    out = pd.DataFrame(rows)
    out.to_csv(cfg.paths.results_dir / "few_shot.csv", index=False)
    logging.info("wrote %s (%d rows)", cfg.paths.results_dir / "few_shot.csv", len(out))
    logging.info("\nmean target-test macro F1 by direction and n_per_class:\n%s",
                 out.groupby(["source", "target", "n_per_class"])["macro_f1"].mean().to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Few-shot target-adaptation curve.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    paths = Paths.default().with_overrides(results_dir=args.results_dir)
    cfg = load_config(args.config, paths=paths) if args.config else load_config(paths=paths)
    run(cfg)


if __name__ == "__main__":
    main()
