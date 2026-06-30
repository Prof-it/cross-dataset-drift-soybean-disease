"""CLI: class-vs-dataset feature silhouettes for the finetune models.

For each fine-tuned model, caches penultimate-layer features for a balanced
diagnostic sample drawn from both datasets' test splits, projects them to 2-D with
UMAP, and computes class and dataset silhouettes. Strong class clustering with weak
dataset clustering supports the head-misalignment reading. Writes
``feature_geometry.csv``.

Needs the trained checkpoints and ``umap-learn``, so run on Colab/GPU
(``pip install umap-learn``). In the notebook, call ``run(cfg)``:

    from src.config import load_config
    from scripts.compute_feature_geometry import run
    run(load_config(paths=paths), per_class=25)
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
from src.evaluation.feature_geometry import silhouettes
from src.experiments._common import run_dir, seed_pairs
from src.models.factory import backbone_features, create_model


def _load(cfg, arch, ckpt, device):
    model = create_model(arch, cfg.data.num_classes, cfg.architectures, pretrained=False)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    return model.to(device)


def _diagnostic(cfg, model, arch, split_seed, per_class, device, rng):
    """Balanced diagnostic sample of penultimate features from both datasets."""
    feats, classes, datasets = [], [], []
    for ds_index, ds in enumerate(cfg.source_datasets):
        f, lab = backbone_features(model, arch, make_eval_loader(cfg, ds, split_seed, "test"), device)
        f = f.numpy()
        lab = lab.numpy()
        for c in np.unique(lab):
            idx = np.where(lab == c)[0]
            if len(idx) > per_class:
                idx = rng.choice(idx, size=per_class, replace=False)
            feats.append(f[idx])
            classes.append(lab[idx])
            datasets.append(np.full(len(idx), ds_index))
    return np.concatenate(feats), np.concatenate(classes), np.concatenate(datasets)


def run(cfg, per_class: int = 25) -> None:
    import umap  # local import: optional dependency, only needed here

    device = get_device()
    for split_seed in cfg.split_seeds:  # splits are generated on demand (as the experiments do)
        for dataset in cfg.source_datasets:
            build_splits(cfg, dataset, split_seed)
    rng = np.random.default_rng(0)
    rows = []
    for split_seed, seed in seed_pairs(cfg):
        for arch in cfg.architectures:
            for path in cfg.training_paths:
                for train_ds in cfg.source_datasets:
                    ckpt = run_dir(cfg, "finetune", model_id(arch, path, train_ds), split_seed, seed) / "best_model.pt"
                    if not ckpt.exists():
                        continue
                    model = _load(cfg, arch, ckpt, device)
                    x, cl, dl = _diagnostic(cfg, model, arch, split_seed, per_class, device, rng)
                    emb = umap.UMAP(n_components=2, random_state=seed).fit_transform(x)
                    rows.append({
                        "arch": arch, "path": path, "train_dataset": train_ds,
                        "split_seed": split_seed, "seed": seed, **silhouettes(emb, cl, dl),
                    })
    if not rows:
        raise SystemExit("no finetune checkpoints found; run the finetune experiment first")

    out = pd.DataFrame(rows)
    out.to_csv(cfg.paths.results_dir / "feature_geometry.csv", index=False)
    logging.info("wrote %s (%d rows)", cfg.paths.results_dir / "feature_geometry.csv", len(out))
    logging.info("\nmean silhouettes by source (class | dataset):\n%s",
                 out.groupby("train_dataset")[["class_silhouette", "dataset_silhouette"]].mean().to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Class vs dataset feature silhouettes.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--per-class", type=int, default=25, help="diagnostic images per class per dataset")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    paths = Paths.default().with_overrides(results_dir=args.results_dir)
    cfg = load_config(args.config, paths=paths) if args.config else load_config(paths=paths)
    run(cfg, per_class=args.per_class)


if __name__ == "__main__":
    main()
