"""Experiment: classifier-head refit / linear probing (Paper 1).

The accuracy-oriented intervention and the "linearly solvable" baselines:

- ``head_refit``: take a model fully fine-tuned on a source dataset, freeze its
  backbone, and refit only the head on the *other* (target) dataset. This is the
  recovery intervention reported in the paper.
- ``imagenet``: freeze the raw ImageNet backbone and fit a head on each dataset
  (tests how linearly separable the task is; feedback #2, light touch).
- ``plantvillage``: same, with the PlantVillage stage-1 backbone.

Which probes to run is set in the experiment YAML (``linear_probe.probes``); the
default runs all three.

Each probe freezes a backbone and fits only a linear head. Rather than re-running
the frozen backbone every epoch, the backbone is forwarded **once** over the train
and val splits (using the eval transform, so no per-epoch augmentation) and the
feature vectors are cached; the head then trains on those cached features. The
full model (frozen backbone + trained head) is saved so the evaluator loads it
like any other checkpoint.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.config import get_device, model_id, set_seed
from src.data.loaders import make_eval_loader
from src.data.splits import build_splits
from src.experiments._common import class_weights, pv_dir, run_dir, run_id
from src.models.factory import backbone_features, classifier, create_model, reinitialize_classifier
from src.training.engine import fit
from src.training.losses import build_loss

if TYPE_CHECKING:
    from src.config import Config

logger = logging.getLogger(__name__)

EXPERIMENT = "linear_probe"
DEFAULT_PROBES = ("head_refit", "imagenet", "plantvillage")


class _FeatureDataset(Dataset):
    """Cached backbone features, yielding ``(feature, label, index)`` to match the
    training engine's batch signature."""

    def __init__(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.features[index], self.labels[index], index


def run(cfg: "Config") -> None:
    probes = tuple(cfg.extras.get("linear_probe", {}).get("probes", DEFAULT_PROBES))
    device = get_device()
    logger.info("linear probe %s on %s | seeds=%s | splits=%s", probes, device, cfg.seeds, cfg.split_seeds)
    for split_seed in cfg.split_seeds:
        for dataset in cfg.source_datasets:
            build_splits(cfg, dataset, split_seed)
        for seed in cfg.seeds:
            for arch in cfg.architectures:
                if "head_refit" in probes:
                    _head_refit(cfg, arch, split_seed, seed, device)
                if "imagenet" in probes:
                    for dataset in cfg.source_datasets:
                        _fit_head(cfg, arch, dataset, "lp_imagenet", split_seed, seed, device)
                if "plantvillage" in probes:
                    _plantvillage(cfg, arch, split_seed, seed, device)


def _head_refit(cfg, arch, split_seed, seed, device) -> None:
    """Source-fine-tuned backbone, head refit on the other (target) dataset."""
    for source in cfg.source_datasets:
        ckpt = run_dir(cfg, "finetune", model_id(arch, "direct", source), split_seed, seed) / "best_model.pt"
        if not ckpt.exists():
            logger.warning("head_refit: missing %s (run finetune first); skipping", ckpt)
            continue
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        for target in (d for d in cfg.source_datasets if d != source):
            _fit_head(cfg, arch, target, f"headrefit_from_{source}", split_seed, seed, device,
                      init_state=state, init_num_classes=cfg.data.num_classes)


def _plantvillage(cfg, arch, split_seed, seed, device) -> None:
    """PlantVillage stage-1 backbone, head fit on each soybean dataset."""
    ckpt = pv_dir(cfg, arch) / "best_model.pt"
    if not ckpt.exists():
        logger.warning("lp_pv: missing %s (run a two-stage finetune first); skipping", ckpt)
        return
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    for dataset in cfg.source_datasets:
        _fit_head(cfg, arch, dataset, "lp_pv", split_seed, seed, device,
                  init_state=state, init_num_classes=cfg.data.num_classes_plantvillage)


def _fit_head(
    cfg, arch, target, tag, split_seed, seed, device,
    init_state: dict | None = None,
    init_num_classes: int | None = None,
) -> None:
    """Freeze a backbone, cache its features once, and fit only the head on ``target``.

    If ``init_state`` is given, the model is built with ``init_num_classes`` (to
    match the checkpoint's label space), loaded, then its head is reset to the
    soybean classes; otherwise the backbone is ImageNet-pretrained. The backbone is
    forwarded once to cache features, the head trains on those, and the full model
    is saved as ``best_model.pt``.
    """
    mid = f"{arch}_{tag}_{target}"
    save_dir = run_dir(cfg, EXPERIMENT, mid, split_seed, seed)
    ckpt = save_dir / "best_model.pt"
    log_path = save_dir / "training_log.json"
    rid = run_id(mid, split_seed, seed)
    if ckpt.exists() and log_path.exists():  # idempotent (Colab-reconnect safe)
        logger.info("[%s] checkpoint found, skipping.", rid)
        return

    set_seed(seed)
    build_classes = init_num_classes if init_state is not None else cfg.data.num_classes
    model = create_model(arch, build_classes, cfg.architectures, pretrained=init_state is None)
    if init_state is not None:
        model.load_state_dict(init_state)
    reinitialize_classifier(model, arch, cfg.data.num_classes, cfg.architectures)

    # Cache frozen-backbone features once (eval transform on both splits).
    train_features, train_labels = backbone_features(
        model, arch, make_eval_loader(cfg, target, split_seed, "train"), device
    )
    val_features, val_labels = backbone_features(
        model, arch, make_eval_loader(cfg, target, split_seed, "val"), device
    )
    model.to("cpu")  # backbone no longer needed on the GPU; only the head trains

    # Train the linear head on cached features (no backbone in the loop -> fast).
    # The head trains in a sub-directory so the final, full-model checkpoint only
    # appears once grafting succeeds (keeps the idempotency signal consistent).
    feature_dim = cfg.architectures[arch]["feature_dim"]
    head = nn.Linear(feature_dim, cfg.data.num_classes)
    criterion = build_loss(cfg, class_weights(cfg, target, split_seed, cfg.data.num_classes), device)
    batch_size = cfg.training.batch_size
    train_loader = DataLoader(_FeatureDataset(train_features, train_labels), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(_FeatureDataset(val_features, val_labels), batch_size=batch_size)
    head_dir = save_dir / "head"
    log = fit(head, train_loader, val_loader, head_dir, criterion, cfg, device, rid)

    # Graft the best head back into the full model and write the final checkpoint.
    head.load_state_dict(torch.load(head_dir / "best_model.pt", map_location="cpu", weights_only=True))
    classifier(model, arch).load_state_dict(head.state_dict())
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt)
    log_path.write_text(json.dumps(log, indent=2))
