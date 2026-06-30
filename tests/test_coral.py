"""Tests: CORAL feature alignment (unsupervised zero-label adaptation)."""

from __future__ import annotations

import numpy as np

from src.evaluation.coral import coral_apply, coral_fit, coral_transform


def test_coral_aligns_mean_and_covariance_to_source():
    rng = np.random.default_rng(0)
    d = 5
    a_s = rng.normal(size=(d, d))
    a_t = rng.normal(size=(d, d))
    cov_s = a_s @ a_s.T + np.eye(d)
    cov_t = a_t @ a_t.T + np.eye(d)
    source = rng.multivariate_normal(np.full(d, 2.0), cov_s, size=8000)
    target = rng.multivariate_normal(np.zeros(d), cov_t, size=8000)

    aligned = coral_transform(source, target)

    # After alignment the target mean and covariance should match the source.
    assert np.allclose(aligned.mean(axis=0), source.mean(axis=0), atol=0.15)
    assert np.allclose(np.cov(aligned, rowvar=False), np.cov(source, rowvar=False), atol=0.3)
    assert aligned.shape == target.shape


def test_coral_identity_when_distributions_match():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2000, 4))
    # Source and target drawn the same way: alignment should be close to a no-op.
    aligned = coral_transform(x, x.copy())
    assert np.allclose(aligned, x, atol=1e-6)


def test_coral_fit_apply_matches_transform_and_generalises():
    rng = np.random.default_rng(2)
    source = rng.normal(0.0, 1.0, size=(500, 4))
    target = rng.normal(1.0, 2.0, size=(600, 4))
    params = coral_fit(source, target)
    # fit-then-apply on the fitting target equals the convenience transform
    assert np.allclose(coral_apply(params, target), coral_transform(source, target))
    # the fitted transform applies to unseen target rows (e.g. the test split)
    held_out = rng.normal(1.0, 2.0, size=(50, 4))
    assert coral_apply(params, held_out).shape == held_out.shape
