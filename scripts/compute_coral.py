"""CLI: CORAL zero-label domain-adaptation baseline for the target-adaptation ladder.

For each architecture and transfer direction, takes the source-fine-tuned (direct)
backbone and its trained head, aligns the target features to the source with CORAL
(unsupervised, no target labels), and evaluates the unchanged head on the aligned
target features. The CORAL covariances are estimated on the larger unlabeled
source-train and target-train splits (with shrinkage) and then applied to the
target-test features, so the alignment is not undersampled. Reports target-test
macro F1 without adaptation and with CORAL, so it sits beside the few-shot
head-refit curve as the zero-label rung. The no-adaptation column reproduces the
cross-dataset macro F1 as a sanity check. Writes ``coral.csv``.

Needs the trained checkpoints, so run on Colab/GPU. In the notebook, call ``run(cfg)``.
Default is the full 3x3 grid; pass ``quick=True`` for the thesis subset.
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd
import torch
from sklearn.metrics import f1_score

from src.config import Paths, get_device, load_config, model_id
from src.data.loaders import make_eval_loader
from src.data.splits import build_splits
from src.evaluation.coral import coral_apply, coral_fit
from src.experiments._common import run_dir, seed_pairs
from src.models.factory import backbone_features, classifier, create_model


def _load(cfg, arch, ckpt, device):
    model = create_model(arch, cfg.data.num_classes, cfg.architectures, pretrained=False)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    return model.to(device)


def _head(model, arch):
    head = classifier(model, arch)
    return head.weight.detach().cpu().numpy(), head.bias.detach().cpu().numpy()


def run(cfg, quick: bool = False) -> None:
    device = get_device()
    pairs = [(cfg.split_seeds[0], s) for s in cfg.seeds] if quick else seed_pairs(cfg)
    for split_seed in sorted({sp for sp, _ in pairs}):  # splits generated on demand
        for dataset in cfg.source_datasets:
            build_splits(cfg, dataset, split_seed)

    rows = []
    for split_seed, seed in pairs:
        for arch in cfg.architectures:
            for source in cfg.source_datasets:
                ckpt = run_dir(cfg, "finetune", model_id(arch, "direct", source), split_seed, seed) / "best_model.pt"
                if not ckpt.exists():
                    continue
                model = _load(cfg, arch, ckpt, device)
                weight, bias = _head(model, arch)
                # Covariance from the larger unlabeled train splits (shrinkage handles conditioning).
                s_feat, _ = backbone_features(model, arch, make_eval_loader(cfg, source, split_seed, "train"), device)
                s_feat = s_feat.numpy()
                for target in (d for d in cfg.source_datasets if d != source):
                    logging.info("CORAL  %s  %s->%s  split%s seed%s", arch, source, target, split_seed, seed)
                    t_train, _ = backbone_features(model, arch, make_eval_loader(cfg, target, split_seed, "train"), device)
                    t_test, t_lab = backbone_features(model, arch, make_eval_loader(cfg, target, split_seed, "test"), device)
                    t_train, t_test, t_lab = t_train.numpy(), t_test.numpy(), t_lab.numpy()
                    params = coral_fit(s_feat, t_train)            # fit on unlabeled train features
                    raw_pred = (t_test @ weight.T + bias).argmax(axis=1)
                    coral_pred = (coral_apply(params, t_test) @ weight.T + bias).argmax(axis=1)
                    rows.append({
                        "arch": arch, "source": source, "target": target,
                        "split_seed": split_seed, "seed": seed,
                        "macro_f1_cross": float(f1_score(t_lab, raw_pred, average="macro")),
                        "macro_f1_coral": float(f1_score(t_lab, coral_pred, average="macro")),
                    })
    if not rows:
        raise SystemExit("no direct finetune checkpoints found; run the finetune experiment first")

    out = pd.DataFrame(rows)
    out.to_csv(cfg.paths.results_dir / "coral.csv", index=False)
    logging.info("wrote %s (%d rows)", cfg.paths.results_dir / "coral.csv", len(out))
    logging.info("\nmean macro F1 by direction (no adaptation vs CORAL):\n%s",
                 out.groupby(["source", "target"])[["macro_f1_cross", "macro_f1_coral"]].mean().to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="CORAL zero-label domain-adaptation baseline.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--quick", action="store_true", help="thesis subset (fixed split seed x init seeds); default is the full 3x3 grid")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    paths = Paths.default().with_overrides(results_dir=args.results_dir)
    cfg = load_config(args.config, paths=paths) if args.config else load_config(paths=paths)
    run(cfg, quick=args.quick)


if __name__ == "__main__":
    main()
