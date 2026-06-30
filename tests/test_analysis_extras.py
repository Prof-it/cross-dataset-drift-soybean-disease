"""Tests: ECE bin robustness, feature-space silhouettes, and few-shot helpers."""

from __future__ import annotations

import numpy as np

from src.evaluation.feature_geometry import silhouettes
from src.evaluation.few_shot import fit_eval_linear, subsample_indices
from src.evaluation.metrics import ece_across_bins, expected_calibration_error


def test_ece_across_bins_keys_and_consistency():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet([1, 1, 1], size=200)
    y = probs.argmax(axis=1)
    out = ece_across_bins(y, probs, (5, 10, 15))
    assert set(out) == {"ece_b5", "ece_b10", "ece_b15"}
    assert out["ece_b10"] == expected_calibration_error(y, probs, 10)
    assert all(0.0 <= v <= 1.0 for v in out.values())


def test_silhouettes_class_separated_dataset_mixed():
    rng = np.random.default_rng(0)
    a = rng.normal([0, 0], 0.05, size=(20, 2))
    b = rng.normal([10, 10], 0.05, size=(20, 2))
    emb = np.vstack([a, b])
    cls = np.array([0] * 20 + [1] * 20)
    ds = np.tile([0, 1], 20)  # datasets interleaved within each class cluster
    s = silhouettes(emb, cls, ds)
    assert s["class_silhouette"] > 0.8
    assert s["dataset_silhouette"] < 0.3
    assert abs(s["class_minus_dataset"] - (s["class_silhouette"] - s["dataset_silhouette"])) < 1e-9


def test_subsample_indices_caps_and_reproducible():
    labels = np.array([0] * 10 + [1] * 3)
    idx = subsample_indices(labels, 5, np.random.default_rng(1))
    assert (labels[idx] == 0).sum() == 5
    assert (labels[idx] == 1).sum() == 3  # only three available
    all_idx = subsample_indices(labels, None, np.random.default_rng(1))
    assert len(all_idx) == 13


def test_fit_eval_linear_separable_is_perfect():
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(-5, 0.5, size=(30, 4)), rng.normal(5, 0.5, size=(30, 4))])
    y = np.array([0] * 30 + [1] * 30)
    assert fit_eval_linear(x, y, x, y) == 1.0
