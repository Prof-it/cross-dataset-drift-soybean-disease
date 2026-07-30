"""Shared driver for the input-level intervention experiments.

Both input-level experiments have the same shape: take the trained fine-tune
checkpoints, run them over a set of edited copies of some evaluation images, and
record one row per (checkpoint run, image, condition). Only the image set and the
edits differ, so that shape lives here and
:mod:`~src.experiments.robustness.background_intervention` /
:mod:`~src.experiments.robustness.frequency_intervention` supply the specifics.

Two properties matter for correctness and are enforced here rather than left to
the callers:

- **The edited inputs are identical across checkpoints.** The condition cache is
  built once per split seed and reused for every architecture, path, source
  dataset and init seed, so a difference between two models is a difference in
  the model, never in its inputs.
- **Split membership travels with every prediction.** Each row records the
  evaluated image's split (``train`` / ``val`` / ``test``) *under that run's split
  seed*. The annotated mask subset was not drawn per split seed, so some of those
  images are in a given run's training data; carrying the split label lets the
  summary restrict to held-out images instead of silently scoring on train.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.config import model_id
from src.data.interventions import ORIGINAL, ConditionCache
from src.data.splits import load_split
from src.experiments._common import run_dir, seed_pairs
from src.utils.io import ensure_dir

if TYPE_CHECKING:
    import torch

    from src.config import Config

logger = logging.getLogger(__name__)

# Split membership that counts as held out from a run's training data. Both val
# and test qualify: neither contributes gradients, and val is only used for early
# stopping and the temperature fit.
HELD_OUT_SPLITS = ("val", "test")


def load_checkpoint(cfg: "Config", arch: str, ckpt, device) -> "torch.nn.Module | None":
    """Load a fine-tuned checkpoint, or ``None`` if it is missing or unreadable."""
    import torch

    from src.models.factory import create_model

    if not ckpt.exists():
        return None
    try:
        model = create_model(arch, cfg.data.num_classes, cfg.architectures, pretrained=False)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        return model.to(device).eval()
    except Exception as exc:
        logger.error("failed to load %s (%s: %s); skipping", ckpt, type(exc).__name__, exc)
        return None


def predict(model, cache: ConditionCache, condition: str, batch_size: int, device):
    """Predicted index and confidence for one condition, in cache order."""
    import torch

    predictions: list[np.ndarray] = []
    confidences: list[np.ndarray] = []
    with torch.no_grad():
        for images, _, _ in cache.batches(condition, batch_size):
            probabilities = model(images.to(device)).softmax(dim=1)
            confidence, predicted = probabilities.max(dim=1)
            predictions.append(predicted.cpu().numpy())
            confidences.append(confidence.cpu().numpy())
    return np.concatenate(predictions), np.concatenate(confidences)


def score_model_space(
    cfg: "Config",
    experiment: str,
    make_caches: "Callable[[int], dict[str, ConditionCache]]",
    device,
    *,
    source_run: str,
    batch_size: int,
) -> pd.DataFrame:
    """Run every fine-tuned checkpoint over every cached condition.

    ``make_caches(split_seed)`` returns ``{evaluated dataset -> ConditionCache}``
    for that split seed. It is called once per split seed and the result is held
    only while that seed's runs are scored, so an experiment whose image set does
    depend on the split seed (the full test split) never holds more than one
    seed's crops in memory. When the image set is split-seed independent, return
    the same caches every time and they are built once. ``seed_pairs`` groups by
    split seed, so at most one rebuild per seed happens either way.

    Returns one row per (run, image, condition).
    """
    rows: list[dict] = []
    caches: dict[str, ConditionCache] = {}
    current_split_seed: int | None = None
    for split_seed, seed in seed_pairs(cfg):
        if split_seed != current_split_seed:
            caches = {}  # drop the previous seed's crops before building the next
            _free(device)
            caches = make_caches(split_seed)
            current_split_seed = split_seed
            logger.info(
                "%s: cache for split%s holds %.0f MB",
                experiment, split_seed, sum(c.megabytes() for c in caches.values()),
            )
        splits = {
            dataset: load_split(cfg, dataset, split_seed).set_index("image_id")["split"]
            for dataset in cfg.source_datasets
        }
        for arch in cfg.architectures:
            for path in cfg.training_paths:
                for train_dataset in cfg.source_datasets:
                    mid = model_id(arch, path, train_dataset)
                    ckpt = run_dir(cfg, source_run, mid, split_seed, seed) / "best_model.pt"
                    model = load_checkpoint(cfg, arch, ckpt, device)
                    if model is None:
                        logger.warning("%s: missing or unreadable %s", experiment, ckpt)
                        continue
                    logger.info(
                        "%s: %s split%s seed%s", experiment, mid, split_seed, seed
                    )
                    for eval_dataset, cache in caches.items():
                        rows += _score_one(
                            model, cache, splits[eval_dataset], device,
                            experiment=experiment, arch=arch, path=path,
                            train_dataset=train_dataset, eval_dataset=eval_dataset,
                            seed=seed, split_seed=split_seed, mid=mid,
                            batch_size=batch_size,
                        )
                    del model
                    _free(device)
    return pd.DataFrame(rows)


def _score_one(
    model, cache, split_lookup, device, *, experiment, arch, path,
    train_dataset, eval_dataset, seed, split_seed, mid, batch_size,
) -> list[dict]:
    frame = cache.frame
    image_ids = frame["image_id"].to_numpy()
    labels = frame["class_index"].to_numpy()
    class_labels = frame["class_label"].to_numpy()
    membership = frame["image_id"].map(split_lookup).to_numpy()
    direction = "within" if eval_dataset == train_dataset else "cross"

    rows: list[dict] = []
    for condition in cache.conditions:
        predicted, confidence = predict(model, cache, condition, batch_size, device)
        for i, image_id in enumerate(image_ids):
            rows.append({
                "experiment": experiment, "arch": arch, "path": path,
                "train_dataset": train_dataset, "eval_dataset": eval_dataset,
                "direction": direction, "seed": seed, "split_seed": split_seed,
                "model_id": mid, "image_id": image_id,
                "class_label": class_labels[i], "class_index": int(labels[i]),
                "eval_split": membership[i], "condition": condition,
                "predicted_index": int(predicted[i]),
                "confidence": float(confidence[i]),
                "correct": bool(predicted[i] == labels[i]),
            })
    return rows


def summarize(predictions: pd.DataFrame, class_names: tuple[str, ...]) -> pd.DataFrame:
    """Per-run macro F1 and accuracy for every condition and eligibility policy.

    ``eligibility`` records which images the row was computed over:

    - ``all``      every annotated/evaluated image, regardless of split membership;
    - ``heldout``  images in the run's val or test split (no training leakage);
    - ``test``     images in the run's test split only (strictest).

    For a cross-dataset row the model never trained on the evaluated dataset at
    all, so all three are leakage-free there and differ only in composition; for a
    within-dataset row ``all`` includes images the model was trained on and is
    reported for completeness rather than for the paper.
    """
    from sklearn.metrics import f1_score

    keys = [
        "experiment", "arch", "path", "train_dataset", "eval_dataset",
        "direction", "seed", "split_seed", "model_id", "condition",
    ]
    policies = {
        "all": lambda f: f,
        "heldout": lambda f: f[f["eval_split"].isin(HELD_OUT_SPLITS)],
        "test": lambda f: f[f["eval_split"] == "test"],
    }
    labels = list(range(len(class_names)))
    rows: list[dict] = []
    for eligibility, select in policies.items():
        subset = select(predictions)
        if subset.empty:
            continue
        for key, group in subset.groupby(keys, sort=False):
            rows.append({
                **dict(zip(keys, key, strict=True)),
                "eligibility": eligibility,
                "n_images": int(len(group)),
                "accuracy": float(group["correct"].mean()),
                "macro_f1": float(
                    f1_score(
                        group["class_index"], group["predicted_index"],
                        labels=labels, average="macro", zero_division=0,
                    )
                ),
            })
    return pd.DataFrame(rows)


def deltas(summary: pd.DataFrame) -> pd.DataFrame:
    """Per-run change in macro F1 and accuracy from ``original`` to each condition."""
    keys = [
        "experiment", "arch", "path", "train_dataset", "eval_dataset",
        "direction", "seed", "split_seed", "model_id", "eligibility",
    ]
    baseline = summary[summary["condition"] == ORIGINAL].set_index(keys)
    edited = summary[summary["condition"] != ORIGINAL]
    joined = edited.join(
        baseline[["macro_f1", "accuracy", "n_images"]].add_suffix("_original"), on=keys
    )
    joined = joined.assign(
        delta_macro_f1=joined["macro_f1"] - joined["macro_f1_original"],
        delta_accuracy=joined["accuracy"] - joined["accuracy_original"],
    )
    return joined.reset_index(drop=True)


def write(cfg: "Config", stem: str, predictions: pd.DataFrame, *, per_image: bool) -> pd.DataFrame:
    """Write the summary (and optionally the per-image predictions); return the summary."""
    results_dir = ensure_dir(cfg.paths.results_dir)
    if predictions.empty:
        logger.warning("%s: no checkpoints scored; nothing written", stem)
        return predictions
    summary = summarize(predictions, cfg.data.class_names)
    summary.to_csv(results_dir / f"{stem}.csv", index=False)
    logger.info("wrote %s (%d rows)", results_dir / f"{stem}.csv", len(summary))
    if per_image:
        path = results_dir / f"{stem}_per_image.csv"
        predictions.to_csv(path, index=False)
        logger.info("wrote %s (%d rows)", path, len(predictions))
    return summary


def log_headline(summary: pd.DataFrame, eligibility: str = "heldout") -> None:
    """Log mean macro F1 by direction and condition, for a quick read of the run."""
    subset = summary[summary["eligibility"] == eligibility]
    if subset.empty:
        return
    table = subset.pivot_table(
        index=["train_dataset", "direction"], columns="condition", values="macro_f1"
    )
    counts = subset.groupby(["train_dataset", "direction"])["n_images"].mean().round(1)
    logger.info(
        "mean macro F1 by condition (eligibility=%s, mean n=%s):\n%s",
        eligibility, counts.to_dict(), table.round(4).to_string(),
    )


def _free(device) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
