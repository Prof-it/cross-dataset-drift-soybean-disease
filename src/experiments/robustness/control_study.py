"""Experiment: the matched-subsample control study.

Tests whether class distribution and sample size, rather than acquisition context,
account for any cross-dataset change in performance. ASDID is resampled to MH's
per-class, per-split counts (``asdid_matched``), then the model space is trained on
it exactly as in ``finetune``. One-directional (ASDID -> MH): MH's per-class support
is too small to subsample in reverse.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.config import get_device, model_id, set_seed
from src.data.loaders import make_dataloaders
from src.data.splits import build_matched_subsample, build_splits
from src.experiments._common import class_weights, pv_dir, run_dir, run_id, seed_pairs
from src.models.factory import create_model
from src.training.engine import fit, fit_two_stage
from src.training.losses import build_loss

if TYPE_CHECKING:
    from src.config import Config

logger = logging.getLogger(__name__)

EXPERIMENT = "control_study"
SOURCE = "asdid_matched"


def run(cfg: "Config") -> None:
    """Build the matched subsample, then train the model space on it."""
    device = get_device()
    pairs = seed_pairs(cfg)
    logger.info("control study (%s, %s) on %s | pairs=%s", SOURCE, cfg.seed_design, device, pairs)
    for split_seed, seed in pairs:
        build_splits(cfg, "asdid", split_seed)
        build_splits(cfg, "mh", split_seed)
        build_matched_subsample(cfg, split_seed)
        if "twostage" in cfg.training_paths:
            build_splits(cfg, "plantvillage", cfg.plantvillage_split_seed)
        for arch in cfg.architectures:
            for path in cfg.training_paths:
                set_seed(seed)
                _train(cfg, arch, path, split_seed, seed, device)


def _train(cfg, arch, path, split_seed, seed, device) -> None:
    mid = model_id(arch, path, SOURCE)
    loaders = make_dataloaders(cfg, SOURCE, split_seed, seed)
    criterion = build_loss(
        cfg, class_weights(cfg, SOURCE, split_seed, cfg.data.num_classes), device
    )
    save_dir = run_dir(cfg, EXPERIMENT, mid, split_seed, seed)
    rid = run_id(mid, split_seed, seed)

    if path == "direct":
        model = create_model(arch, cfg.data.num_classes, cfg.architectures, pretrained=True)
        fit(model, loaders["train"], loaders["val"], save_dir, criterion, cfg, device, rid)
    elif path == "twostage":
        pv_split = cfg.plantvillage_split_seed
        pv_loaders = make_dataloaders(cfg, "plantvillage", pv_split, pv_split)
        pv_criterion = build_loss(
            cfg, class_weights(cfg, "plantvillage", pv_split, cfg.data.num_classes_plantvillage), device
        )
        # Reuse the shared PlantVillage stage-1 checkpoint (same backbone as finetune).
        stage1_dir = pv_dir(cfg, arch)
        stage1_rid = run_id(f"{arch}_twostage_pv", pv_split, pv_split)
        fit_two_stage(arch, pv_loaders, loaders, stage1_dir, save_dir,
                      pv_criterion, criterion, cfg, device, stage1_rid, rid,
                      pv_seed=pv_split, target_seed=seed)
    else:
        raise ValueError(f"Unknown training path: {path}")
