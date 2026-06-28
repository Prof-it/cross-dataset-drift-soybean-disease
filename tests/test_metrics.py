"""Tests: metrics and calibration (need scikit-learn)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

from src.evaluation.metrics import compute_metrics, expected_calibration_error

CLASSES = ("healthy", "rust", "frogeye_leaf_spot")


def test_perfect_predictions():
    y = np.array([0, 1, 2, 0, 1, 2])
    m = compute_metrics(y, y, CLASSES)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0


def test_macro_f1_weights_classes_equally():
    # All rust (majority) correct, both rare-class items wrong -> high accuracy, low macro F1.
    y_true = np.array([1, 1, 1, 1, 0, 2])
    y_pred = np.array([1, 1, 1, 1, 1, 1])
    m = compute_metrics(y_true, y_pred, CLASSES)
    assert m["accuracy"] > m["macro_f1"]


def test_ece_low_when_calibrated_and_high_when_overconfident():
    probs = np.array([[0.99, 0.005, 0.005], [0.005, 0.99, 0.005]])
    assert expected_calibration_error(np.array([0, 1]), probs, 10) < 0.05   # confident + correct
    assert expected_calibration_error(np.array([1, 0]), probs, 10) > 0.5    # confident + wrong
