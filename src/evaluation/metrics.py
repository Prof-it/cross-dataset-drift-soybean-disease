"""Classification and calibration metrics.

Computes the metrics the paper reports from model outputs. Macro F1 is the
primary metric (MH is rust-dominated); accuracy and ECE are reported alongside.
``collect_predictions`` returns raw logits so calibration (temperature scaling)
and probability-based metrics can be derived without a second forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax (numerically stable)."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


@dataclass
class Predictions:
    """Model outputs for one evaluation pass over a loader."""

    image_ids: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    logits: np.ndarray  # shape (N, C)

    @property
    def probs(self) -> np.ndarray:
        return softmax(self.logits)


def collect_predictions(model, dataloader, device) -> Predictions:
    """Run inference over a loader, returning ids, labels, predictions, and logits."""
    import torch  # local import keeps the metric helpers torch-free

    model.eval()
    ids: list[str] = []
    labels_all: list[np.ndarray] = []
    logits_all: list[np.ndarray] = []
    with torch.no_grad():
        for images, labels, image_ids in dataloader:
            logits = model(images.to(device)).cpu().numpy()
            logits_all.append(logits)
            labels_all.append(labels.numpy())
            ids.extend(str(i) for i in image_ids)
    logits = np.concatenate(logits_all)
    y_true = np.concatenate(labels_all)
    return Predictions(np.asarray(ids), y_true, logits.argmax(axis=1), logits)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: tuple[str, ...]) -> dict:
    """Accuracy, macro F1, and per-class precision/recall/F1/support."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(class_names)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n)), zero_division=0
    )
    out: dict = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    for i, name in enumerate(class_names):
        out[f"{name}_precision"] = float(precision[i])
        out[f"{name}_recall"] = float(recall[i])
        out[f"{name}_f1"] = float(f1[i])
        out[f"{name}_support"] = int(support[i])
    return out


def confusion(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> tuple[np.ndarray, np.ndarray]:
    """Raw and row-normalized (true-label) confusion matrices."""
    labels = list(range(num_classes))
    raw = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = raw.sum(axis=1, keepdims=True)
    normalized = np.divide(
        raw.astype(float), row_sums, out=np.zeros(raw.shape, dtype=float), where=row_sums != 0
    )
    return raw, normalized


def _bin_stats(y_true: np.ndarray, probs: np.ndarray, n_bins: int):
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    acc = np.zeros(n_bins)
    conf = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (confidence >= lo) & (confidence < hi) if b < n_bins - 1 else (
            (confidence >= lo) & (confidence <= hi)
        )
        count = int(mask.sum())
        counts[b] = count
        if count:
            acc[b] = correct[mask].mean()
            conf[b] = confidence[mask].mean()
    return acc, conf, counts


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    """Equal-width-bin ECE (Guo et al., 2017): weighted mean |accuracy - confidence|."""
    acc, conf, counts = _bin_stats(y_true, probs, n_bins)
    total = int(counts.sum())
    if total == 0:
        return 0.0
    return float(np.sum(counts * np.abs(acc - conf)) / total)


def reliability_curve(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> dict:
    """Per-bin accuracy, confidence, and counts for a reliability diagram."""
    acc, conf, counts = _bin_stats(y_true, probs, n_bins)
    return {"bin_accuracies": acc, "bin_confidences": conf, "bin_counts": counts}
