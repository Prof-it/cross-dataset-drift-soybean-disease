"""Feature-space geometry: class vs dataset silhouettes.

Tests whether a model's penultimate-layer representations cluster more by disease
class or by source dataset. Strong class clustering with weak dataset clustering
supports the head-misalignment reading (the features stay usable across datasets;
the limiting step is the decision rule).

:func:`silhouettes` is a pure helper over a low-dimensional embedding and is unit
tested. The feature extraction itself (loading a backbone, caching features for a
diagnostic image set, UMAP projection) lives in ``scripts/compute_feature_geometry.py``
because it needs the trained checkpoints and a GPU.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import silhouette_score


def silhouettes(embedding, class_labels, dataset_labels) -> dict:
    """Class and dataset silhouette scores for a (2-D) embedding.

    Returns ``class_silhouette``, ``dataset_silhouette`` and their difference. A
    label set with fewer than two groups (or fewer than three points) yields NaN,
    since the silhouette is undefined there.
    """
    emb = np.asarray(embedding, dtype=float)
    cl = np.asarray(class_labels)
    dl = np.asarray(dataset_labels)

    def _safe(labels) -> float:
        labels = np.asarray(labels)
        if len(labels) < 3 or len(np.unique(labels)) < 2:
            return float("nan")
        return float(silhouette_score(emb, labels))

    cs = _safe(cl)
    ds = _safe(dl)
    return {
        "class_silhouette": cs,
        "dataset_silhouette": ds,
        "class_minus_dataset": cs - ds,
    }
