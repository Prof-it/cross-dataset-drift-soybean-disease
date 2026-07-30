"""Experiment: background-replacement intervention.

Asks whether the cross-dataset decision rule leans on background context that does
not transfer. Every fine-tuned checkpoint is re-scored on the hand-annotated
subset of both datasets under three conditions: the unedited image, and the image
with all background pixels (mask == 0) replaced by a uniform grey or black fill.
Foreground masks come from ``paths.masks_dir`` (see
``data/masks/ANNOTATION_PROTOCOL.md``).

Reading the result. Replacing the background is itself an out-of-distribution
edit -- no training image contains a uniform grey field -- so a model that gained
nothing from background context should get *worse*, not better. If cross-dataset
macro F1 nevertheless rises while within-dataset macro F1 falls, the models were
using dataset-specific background cues, and the cost of the artefact is smaller
than the benefit of removing them. The frequency low-pass experiment is the
control for the alternative reading that any input edit would have moved the
numbers.

Partition. An annotated image is usable for the within-dataset arm only if it was
held out of that run's training data, and "held out" is a property of the (image,
split seed) pair rather than of the image. A set held out under every split seed
is too small to be usable, so the experiment YAML pins ``split_seeds`` to one
partition and varies only the initialization seed; ``scripts/select_annotation_batch.py``
picks the images to annotate for that partition. The cross-dataset arm is
unaffected by this either way, since a model never trains on the other dataset.

As a guard rather than an assumption, every prediction still carries the evaluated
image's split membership, and the summary reports each metric under three
eligibility policies (``all``, ``heldout``, ``test``). Use ``heldout`` for
within-dataset claims: if the annotated set and the configured partition ever drift
apart, the difference between ``all`` and ``heldout`` makes that visible instead of
silently scoring on training images.

Outputs (under ``paths.results_dir``):
- ``background_intervention.csv``            per (run, condition, eligibility) metrics
- ``background_intervention_per_image.csv``  per-image predictions (paired tests)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.config import get_device
from src.data.interventions import ORIGINAL, ConditionCache, background_edits, mask_table
from src.data.splits import build_splits
from src.experiments.robustness import _intervention_driver as driver

if TYPE_CHECKING:
    from src.config import Config

logger = logging.getLogger(__name__)

EXPERIMENT = "background_intervention"


def run(cfg: "Config") -> None:
    options = cfg.extras.get(EXPERIMENT, {})
    fills = {name: tuple(value) for name, value in options["fills"].items()}
    source_run = str(options.get("source_run_name", "finetune"))
    batch_size = int(options.get("batch_size", cfg.training.batch_size))
    per_image = bool(options.get("write_per_image", True))
    device = get_device()

    for split_seed in cfg.split_seeds:
        for dataset in cfg.source_datasets:
            build_splits(cfg, dataset, split_seed)

    # The annotated image set does not depend on the split seed -- only the split
    # label attached to each image does -- so the (expensive) edited crops are
    # built once and shared by every split seed.
    reference_seed = cfg.split_seeds[0]
    edits = background_edits(fills)
    caches = {}
    for dataset in cfg.source_datasets:
        frame = mask_table(cfg, dataset, reference_seed)
        logger.info(
            "%s: %s has %d annotated images (%s)",
            EXPERIMENT, dataset, len(frame), frame["class_label"].value_counts().to_dict(),
        )
        caches[dataset] = ConditionCache(cfg, frame, edits)
    total_mb = sum(cache.megabytes() for cache in caches.values())
    logger.info(
        "%s: conditions=%s | cache %.0f MB | device=%s",
        EXPERIMENT, (ORIGINAL, *fills), total_mb, device,
    )

    predictions = driver.score_model_space(
        cfg, EXPERIMENT, lambda _split_seed: caches, device,
        source_run=source_run, batch_size=batch_size,
    )
    summary = driver.write(cfg, EXPERIMENT, predictions, per_image=per_image)
    if not summary.empty:
        driver.log_headline(summary)
