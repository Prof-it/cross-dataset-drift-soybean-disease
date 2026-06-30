"""CORAL feature alignment for zero-label (unsupervised) domain adaptation.

Closed-form CORrelation ALignment (Sun and Saenko, 2016): align the second-order
statistics of the target features to the source, so a source-trained classifier
head applies to the target without any target labels and without retraining. This
is the zero-label rung of the target-adaptation ladder; the few-label rung is the
head refit.

Covariances are estimated with Ledoit-Wolf shrinkage, which keeps the alignment
well-conditioned when the feature dimension is large relative to the sample size.
``coral_fit`` estimates the transform from (unlabeled) source and target feature
samples and ``coral_apply`` applies it, so the covariance can be estimated on the
larger unlabeled target *train* split and then applied to the target test split.
``coral_transform`` is the fit-and-apply convenience used in the tests. The feature
extraction that feeds these lives in ``scripts/compute_coral.py`` because it needs
the trained checkpoints and a GPU.
"""

from __future__ import annotations

import numpy as np


def _sym_power(cov: np.ndarray, power: float, eps: float) -> np.ndarray:
    """Symmetric-matrix power via eigendecomposition (cov must be symmetric PSD)."""
    w, vecs = np.linalg.eigh(cov)
    w = np.clip(w, eps, None)
    return (vecs * (w ** power)) @ vecs.T


def _shrunk_cov(x: np.ndarray, eps: float) -> np.ndarray:
    """Ledoit-Wolf shrinkage covariance, well-conditioned for small n / large d."""
    from sklearn.covariance import ledoit_wolf

    cov, _ = ledoit_wolf(np.asarray(x, dtype=float))
    return cov + eps * np.eye(cov.shape[0])


def coral_fit(source: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> dict:
    """Estimate the CORAL transform that aligns target features to the source.

    Uses shrinkage covariance on both feature sets. Returns the mean shift and the
    whiten-recolour matrix; apply it with :func:`coral_apply`. ``source`` and
    ``target`` are unlabeled ``(n, d)`` feature arrays.
    """
    s = np.asarray(source, dtype=float)
    t = np.asarray(target, dtype=float)
    transform = _sym_power(_shrunk_cov(t, eps), -0.5, eps) @ _sym_power(_shrunk_cov(s, eps), 0.5, eps)
    return {"mean_source": s.mean(axis=0), "mean_target": t.mean(axis=0), "transform": transform}


def coral_apply(params: dict, x: np.ndarray) -> np.ndarray:
    """Apply a fitted CORAL transform to ``x`` (whiten by target, recolour to source)."""
    x = np.asarray(x, dtype=float)
    return (x - params["mean_target"]) @ params["transform"] + params["mean_source"]


def coral_transform(source: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Fit on (source, target) and apply to target. Convenience for the transductive case."""
    return coral_apply(coral_fit(source, target, eps), target)
