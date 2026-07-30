"""Image transforms for training and evaluation.

Builds the torchvision pipelines from the typed config, so input size,
normalization, and augmentation are never hard-coded at the call site. The train
pipeline adds the augmentation from ``cfg.data.augmentation``; val and test are
deterministic (resize, center crop, normalize).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torchvision import transforms

if TYPE_CHECKING:
    from src.config import Config


def eval_geometry(cfg: "Config") -> transforms.Compose:
    """The deterministic geometry of the eval pipeline: resize then center crop.

    This is the leading half of ``build_transforms(cfg, train=False)``, split out
    so the input-level interventions can cache uint8 crops and normalize only at
    batch time (see :mod:`src.data.interventions`).
    """
    return transforms.Compose([
        transforms.Resize(cfg.data.resize_size),
        transforms.CenterCrop(cfg.data.input_size),
    ])


def build_transforms(cfg: "Config", train: bool) -> transforms.Compose:
    """Return the transform pipeline for the train (augmented) or eval split."""
    data = cfg.data
    steps: list = [
        transforms.Resize(data.resize_size),
        transforms.CenterCrop(data.input_size),
    ]
    if train:
        aug = data.augmentation
        steps += [
            transforms.RandomAffine(degrees=aug.rotation_degrees, translate=tuple(aug.translate)),
            transforms.ColorJitter(brightness=tuple(aug.brightness)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ]
    steps += [
        transforms.ToTensor(),
        transforms.Normalize(mean=list(data.mean), std=list(data.std)),
    ]
    return transforms.Compose(steps)
