"""Few-shot target adaptation: how little labeled target data the head refit needs.

Pure helpers for the data-efficiency curve. :func:`subsample_indices` picks up to
``n`` examples per class; :func:`fit_eval_linear` fits a linear classifier on cached
features and returns target-test macro F1. Both are unit tested. The orchestration
that loads a fine-tuned backbone and caches its features lives in
``scripts/compute_few_shot.py`` because it needs the trained checkpoints and a GPU.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


def subsample_indices(labels, n_per_class, rng) -> np.ndarray:
    """Indices of up to ``n_per_class`` examples per class (all of a class if fewer).

    ``n_per_class=None`` returns every index. ``rng`` is a ``numpy.random.Generator``
    so the subsample is reproducible.
    """
    labels = np.asarray(labels)
    chosen: list[int] = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if n_per_class is None or n_per_class >= len(idx):
            chosen.extend(idx.tolist())
        else:
            chosen.extend(rng.choice(idx, size=n_per_class, replace=False).tolist())
    return np.array(sorted(chosen))


def fit_eval_linear(train_x, train_y, test_x, test_y, *, max_iter: int = 2000) -> float:
    """Fit a multinomial logistic head on cached features; return target-test macro F1."""
    clf = LogisticRegression(max_iter=max_iter)
    clf.fit(np.asarray(train_x), np.asarray(train_y))
    pred = clf.predict(np.asarray(test_x))
    return float(f1_score(np.asarray(test_y), pred, average="macro"))
