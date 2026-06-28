"""Post-hoc calibration: temperature scaling (Guo et al., 2017).

The temperature-scaling intervention: fit a single scalar temperature on
validation logits by minimizing NLL (LBFGS), then divide held-out logits by it.
Argmax-invariant, so it changes calibration but not predictions. Operates on
collected logits (see :func:`src.evaluation.metrics.collect_predictions`), so it
needs no model or dataloader. Label smoothing, the training-time calibration
intervention, lives in :mod:`src.training.losses`.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    max_iter: int = 50,
    lr: float = 0.01,
) -> float:
    """Fit the optimal temperature on validation logits/labels; return the scalar."""
    logits_t = torch.as_tensor(np.asarray(logits), dtype=torch.float)
    labels_t = torch.as_tensor(np.asarray(labels), dtype=torch.long)
    temperature = nn.Parameter(torch.ones(1))
    optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iter)
    criterion = nn.CrossEntropyLoss()

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = criterion(logits_t / temperature, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        temperature.clamp_(0.01, 100.0)
    return float(temperature.item())


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Divide logits by the fitted temperature (recompute softmax for probabilities)."""
    return np.asarray(logits, dtype=float) / float(temperature)
