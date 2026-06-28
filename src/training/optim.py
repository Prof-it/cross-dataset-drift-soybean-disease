"""Optimizer and learning-rate scheduler builders.

Constructs both from the typed config in one place, so the training engine stays
agnostic to the specific choices.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from src.config import Config


def build_optimizer(parameters: Iterable[torch.nn.Parameter], cfg: "Config") -> torch.optim.Optimizer:
    """Adam with the configured learning rate, weight decay, and betas."""
    t = cfg.training
    return torch.optim.Adam(
        parameters,
        lr=t.learning_rate,
        weight_decay=t.weight_decay,
        betas=tuple(t.adam_betas),
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: "Config"
) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    """ReduceLROnPlateau stepped on validation loss."""
    s = cfg.training.schedule
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=s.factor,
        patience=s.patience,
        min_lr=s.min_lr,
    )
