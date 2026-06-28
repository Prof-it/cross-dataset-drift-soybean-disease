"""Loss construction.

Builds the classification loss from config so weighting and label smoothing are
explicit and configurable. Both are experiment variables: inverse-frequency class
weights handle MH's imbalance, and label smoothing is one of the calibration
interventions. The experiment decides whether to pass ``class_weights`` (it omits
them for the unweighted-MH ablation, ``cfg.training.class_weighting = false``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from src.config import Config


def build_loss(
    cfg: "Config",
    class_weights: Sequence[float] | torch.Tensor | None = None,
    device: torch.device | None = None,
) -> nn.Module:
    """Cross-entropy with optional inverse-frequency weights and label smoothing.

    ``class_weights=None`` yields the unweighted variant; ``label_smoothing`` comes
    from the config.
    """
    weight = None
    if class_weights is not None:
        weight = torch.as_tensor(class_weights, dtype=torch.float)
        if device is not None:
            weight = weight.to(device)
    return nn.CrossEntropyLoss(weight=weight, label_smoothing=cfg.training.label_smoothing)
