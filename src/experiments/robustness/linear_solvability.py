"""Experiment: linear-solvability spectrum (addresses Prof. Lu feedback #2).

Answers the question "is the cross-dataset task merely *linearly solvable* with
generic features?" by fitting the **same** linear classifier on a spectrum of
*fixed* (untrained) feature extractors and comparing within- and cross-dataset
macro F1:

- ``raw_pixel``   : flattened, downsampled RGB pixels (``pixel_size`` per side).
- ``random_proj`` : a fixed Gaussian random projection of those pixels.
- ``random_init`` : features from a frozen, **randomly initialised** backbone
  (same architecture, no pretraining, no fine-tuning).
- ``imagenet``    : features from a frozen **ImageNet-pretrained** backbone
  (reproduces the thesis LP-on-ImageNet condition).

Interpretation
--------------
- If ``random_init`` / ``raw_pixel`` reach ``imagenet`` accuracy, the task is
  linearly solvable with almost any features and the disease-specific
  representation claim must be tempered.
- If ``imagenet`` >> ``random_init`` >> ``raw_pixel``, the ImageNet features carry
  genuine, disease-relevant signal that is not a trivial linear artefact, which
  answers the reviewer's "linearly solvable" objection.

Design note (comparability): the classifier is a single multinomial logistic
regression fit on standardised, cached features, with ``class_weight="balanced"``
to mirror the inverse-frequency weighted cross-entropy used elsewhere. The head
fitting is therefore *identical* across all feature types, so the comparison is
apples-to-apples. This is deliberately a *fixed-feature* probe: it does not
fine-tune, and it does not use the fine-tuned checkpoints (those are reported by
the finetune/head-refit experiments). Compare these numbers against the
fine-tuned within/cross macro F1 from ``eval_results.csv``.

Output: one tidy CSV, ``results/linear_solvability.csv``, one row per
(feature, arch, train_dataset, eval_dataset, direction, split_seed, seed).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.config import get_device, set_seed
from src.data.loaders import make_eval_loader
from src.data.splits import build_splits, load_split
from src.evaluation.metrics import compute_metrics, expected_calibration_error
from src.models.factory import backbone_features, create_model
from src.utils.io import ensure_dir

if TYPE_CHECKING:
    from src.config import Config

logger = logging.getLogger(__name__)

EXPERIMENT = "linear_solvability"
DEFAULT_FEATURES = ("raw_pixel", "random_proj", "random_init", "imagenet")
BACKBONE_FEATURES = {"random_init": False, "imagenet": True}  # -> pretrained flag


# --------------------------------------------------------------------------- #
# Feature extractors (all return (X, y) as float32 / int numpy arrays)        #
# --------------------------------------------------------------------------- #
def _pixel_features(cfg: "Config", dataset: str, split: str, split_seed: int, size: int):
    """Downsampled, flattened RGB pixels in [0, 1] for one split (no augmentation)."""
    df = load_split(cfg, dataset, split_seed)
    df = df[df["split"] == split]
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for rel_path, class_index in zip(df["relative_path"], df["class_index"]):
        img = Image.open(cfg.paths.data_root / rel_path).convert("RGB").resize((size, size))
        xs.append(np.asarray(img, dtype=np.float32).reshape(-1) / 255.0)
        ys.append(int(class_index))
    return np.stack(xs), np.asarray(ys, dtype=int)


def _backbone_features(cfg, arch, dataset, split, split_seed, device, pretrained, seed):
    """Frozen-backbone features for one split. ``set_seed`` fixes the random init."""
    set_seed(seed)
    model = create_model(arch, cfg.data.num_classes, cfg.architectures, pretrained=pretrained)
    loader = make_eval_loader(cfg, dataset, split_seed, split)
    feats, labels = backbone_features(model, arch, loader, device)
    return feats.numpy().astype(np.float32), labels.numpy().astype(int)


# --------------------------------------------------------------------------- #
# Linear head: fit once on the source, score within and cross                  #
# --------------------------------------------------------------------------- #
def _fit_and_score(x_train, y_train, eval_sets, cfg, seed, projection=None):
    """Fit a standardised multinomial logistic head on the source train features
    and score each eval set. ``eval_sets`` maps direction -> (X, y). Returns a list
    of ``(direction, metrics_dict)``."""
    if projection is not None:
        x_train = x_train @ projection
    scaler = StandardScaler().fit(x_train)
    clf = LogisticRegression(
        max_iter=2000, C=1.0, class_weight="balanced", solver="lbfgs", random_state=seed
    )
    clf.fit(scaler.transform(x_train), y_train)

    out = []
    for direction, (x_eval, y_eval) in eval_sets.items():
        xe = x_eval @ projection if projection is not None else x_eval
        proba = clf.predict_proba(scaler.transform(xe))
        # align probability columns to 0..K-1 in case a class is absent from train
        full = np.zeros((proba.shape[0], cfg.data.num_classes), dtype=float)
        full[:, clf.classes_] = proba
        pred = full.argmax(axis=1)
        metrics = compute_metrics(y_eval, pred, cfg.data.class_names)
        metrics["ece"] = expected_calibration_error(y_eval, full, cfg.evaluation.ece_bins)
        out.append((direction, metrics))
    return out


def _other(cfg, dataset: str) -> str:
    return next(d for d in cfg.source_datasets if d != dataset)


def _row(cfg, feature, arch, train_ds, eval_ds, direction, seed, split_seed, metrics) -> dict:
    return {
        "experiment": EXPERIMENT, "feature": feature, "arch": arch,
        "train_dataset": train_ds, "eval_dataset": eval_ds, "direction": direction,
        "seed": seed, "split_seed": split_seed,
        "model_id": f"{arch}_{feature}_{train_ds}", **metrics,
    }


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #
def run(cfg: "Config") -> None:
    opts = cfg.extras.get(EXPERIMENT, {})
    features = tuple(opts.get("feature_types", DEFAULT_FEATURES))
    pixel_size = int(opts.get("pixel_size", 32))
    proj_dim = int(opts.get("proj_dim", 512))
    device = get_device()
    logger.info(
        "linear_solvability features=%s on %s | pixel_size=%d proj_dim=%d | splits=%s seeds=%s",
        features, device, pixel_size, proj_dim, cfg.split_seeds, cfg.seeds,
    )
    rows: list[dict] = []

    for split_seed in cfg.split_seeds:
        for dataset in cfg.source_datasets:
            build_splits(cfg, dataset, split_seed)

        # Pixel features are seed-independent: cache once per split_seed.
        need_pixels = "raw_pixel" in features or "random_proj" in features
        pixels = {
            (ds, sp): _pixel_features(cfg, ds, sp, split_seed, pixel_size)
            for ds in cfg.source_datasets for sp in ("train", "test")
        } if need_pixels else {}

        # ImageNet features are also seed-independent (fixed weights): cache per
        # (arch, dataset, split) once per split_seed. Random-init features depend
        # on the seed, so they are extracted inside the seed loop.
        imagenet = {}
        if "imagenet" in features:
            for arch in cfg.architectures:
                for ds in cfg.source_datasets:
                    for sp in ("train", "test"):
                        imagenet[(arch, ds, sp)] = _backbone_features(
                            cfg, arch, ds, sp, split_seed, device, pretrained=True, seed=0
                        )

        for seed in cfg.seeds:
            # ---- pixel-level floors ----
            if need_pixels:
                rng = np.random.default_rng(seed)
                proj = rng.standard_normal((pixel_size * pixel_size * 3, proj_dim)).astype(np.float32)
                proj /= np.sqrt(proj_dim)
                for src in cfg.source_datasets:
                    other = _other(cfg, src)
                    xtr, ytr = pixels[(src, "train")]
                    eval_sets = {"within": pixels[(src, "test")], "cross": pixels[(other, "test")]}
                    if "raw_pixel" in features:
                        for direction, m in _fit_and_score(xtr, ytr, eval_sets, cfg, seed):
                            rows.append(_row(cfg, "raw_pixel", "raw_pixel", src,
                                             src if direction == "within" else other, direction, seed, split_seed, m))
                    if "random_proj" in features:
                        for direction, m in _fit_and_score(xtr, ytr, eval_sets, cfg, seed, projection=proj):
                            rows.append(_row(cfg, "random_proj", "random_proj", src,
                                             src if direction == "within" else other, direction, seed, split_seed, m))

            # ---- backbone feature probes ----
            for arch in cfg.architectures:
                for kind, pretrained in BACKBONE_FEATURES.items():
                    if kind not in features:
                        continue
                    if kind == "imagenet":
                        feats = {(ds, sp): imagenet[(arch, ds, sp)]
                                 for ds in cfg.source_datasets for sp in ("train", "test")}
                    else:  # random_init: extract per seed
                        feats = {(ds, sp): _backbone_features(cfg, arch, ds, sp, split_seed, device, pretrained, seed)
                                 for ds in cfg.source_datasets for sp in ("train", "test")}
                    for src in cfg.source_datasets:
                        other = _other(cfg, src)
                        xtr, ytr = feats[(src, "train")]
                        eval_sets = {"within": feats[(src, "test")], "cross": feats[(other, "test")]}
                        for direction, m in _fit_and_score(xtr, ytr, eval_sets, cfg, seed):
                            rows.append(_row(cfg, kind, arch, src,
                                             src if direction == "within" else other, direction, seed, split_seed, m))
            logger.info("linear_solvability: split %s seed %s done (%d rows so far)", split_seed, seed, len(rows))

    out_dir = ensure_dir(cfg.paths.results_dir)
    df = pd.DataFrame(rows)
    out_path = out_dir / "linear_solvability.csv"
    df.to_csv(out_path, index=False)
    logger.info("wrote %d rows to %s", len(df), out_path)

    # Console summary: mean macro F1 per (feature, direction), averaged over archs/seeds/splits.
    if not df.empty:
        summary = df.groupby(["feature", "direction"])["macro_f1"].mean().round(3)
        logger.info("macro F1 by feature x direction (mean over arch/seed/split):\n%s", summary.to_string())
