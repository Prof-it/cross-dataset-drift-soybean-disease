"""Train the model space.

Produces the trained checkpoints the analyses consume, so this is
experiment-agnostic and lives at the ``experiments/`` level. Full fine-tuning of the
model space (architectures x training_paths x source_datasets) over the
configured training and split seeds. The unweighted-MH ablation and the
label-smoothing variant are config overrides of the same driver
(``training.class_weighting`` / ``training.label_smoothing``).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.config import get_device, model_id, set_seed
from src.data.loaders import make_dataloaders
from src.data.splits import build_splits
from src.experiments._common import class_weights, pv_dir, run_dir, run_id, seed_pairs
from src.models.factory import create_model
from src.training.engine import fit, fit_two_stage
from src.training.losses import build_loss

if TYPE_CHECKING:
    from src.config import Config

logger = logging.getLogger(__name__)


def run(cfg: "Config") -> None:
    """Train every (architecture, path, source dataset) over all seeds and splits.

    Target checkpoints are written under ``cfg.run_name`` (default ``finetune``).
    Variants set a distinct ``run_name`` (e.g. ``finetune_unweighted``,
    ``finetune_label_smoothing``) so they do not collide with the baseline; the
    shared PlantVillage stage-1 always lives in the ``finetune`` namespace and is
    reused across all of them.
    """
    device = get_device()
    pairs = seed_pairs(cfg)
    logger.info("finetune (run_name=%s, %s) on %s | pairs=%s",
                cfg.run_name, cfg.seed_design, device, pairs)
    for split_seed, seed in pairs:
        _ensure_splits(cfg, split_seed)
        for arch in cfg.architectures:
            for path in cfg.training_paths:
                for dataset in cfg.source_datasets:
                    set_seed(seed)
                    _train(cfg, arch, path, dataset, split_seed, seed, device)


def _ensure_splits(cfg: "Config", split_seed: int) -> None:
    for dataset in cfg.source_datasets:
        build_splits(cfg, dataset, split_seed)
    if "twostage" in cfg.training_paths:
        build_splits(cfg, "plantvillage", cfg.plantvillage_split_seed)


def _train(cfg, arch, path, dataset, split_seed, seed, device) -> None:
    mid = model_id(arch, path, dataset)
    target_loaders = make_dataloaders(cfg, dataset, split_seed, seed)
    target_criterion = build_loss(
        cfg, class_weights(cfg, dataset, split_seed, cfg.data.num_classes), device
    )
    save_dir = run_dir(cfg, cfg.run_name, mid, split_seed, seed)
    rid = run_id(mid, split_seed, seed)

    if path == "direct":
        model = create_model(arch, cfg.data.num_classes, cfg.architectures, pretrained=True)
        fit(model, target_loaders["train"], target_loaders["val"], save_dir,
            target_criterion, cfg, device, rid)
    elif path == "twostage":
        pv_split = cfg.plantvillage_split_seed
        pv_loaders = make_dataloaders(cfg, "plantvillage", pv_split, pv_split)
        pv_criterion = build_loss(
            cfg, class_weights(cfg, "plantvillage", pv_split, cfg.data.num_classes_plantvillage), device
        )
        stage1_dir = pv_dir(cfg, arch)
        stage1_rid = run_id(f"{arch}_twostage_pv", pv_split, pv_split)
        fit_two_stage(arch, pv_loaders, target_loaders, stage1_dir, save_dir,
                      pv_criterion, target_criterion, cfg, device, stage1_rid, rid,
                      pv_seed=pv_split, target_seed=seed)
    else:
        raise ValueError(f"Unknown training path: {path}")
