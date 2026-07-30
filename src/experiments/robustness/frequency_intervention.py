"""Experiment: frequency low-pass intervention.

The control for the background result. Every fine-tuned checkpoint is re-scored on
the *full* test split of both datasets under the unedited image and under a
Gaussian low-pass filter at each configured sigma. Blurring removes fine texture
without removing any semantic region and needs no annotation, so it separates two
readings of the background finding:

- if cross-dataset performance degrades no faster than within-dataset performance
  under blur, the models are not leaning on fine high-frequency texture that the
  change of acquisition context destroys, and the background effect is specific to
  *what* was removed;
- if it degrades faster, high-frequency texture is itself a non-transferring cue
  and the background result needs that qualification.

Unlike the background experiment this one uses the full test split, so the image
set *does* depend on the split seed and the edited crops are rebuilt (and
released) one split seed at a time. Per-image predictions are large here
(hundreds of thousands of rows) and are off by default; enable them only if you
need the paired per-image tests.

Outputs (under ``paths.results_dir``):
- ``frequency_intervention.csv``            per (run, condition, eligibility) metrics
- ``frequency_intervention_per_image.csv``  per-image predictions (opt-in)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.config import get_device
from src.data.interventions import ConditionCache, frequency_edits
from src.data.splits import build_splits, load_split
from src.experiments.robustness import _intervention_driver as driver

if TYPE_CHECKING:
    from src.config import Config

logger = logging.getLogger(__name__)

EXPERIMENT = "frequency_intervention"


def run(cfg: "Config") -> None:
    options = cfg.extras.get(EXPERIMENT, {})
    sigmas = [float(s) for s in options["sigmas"]]
    split = str(options.get("split", "test"))
    source_run = str(options.get("source_run_name", "finetune"))
    batch_size = int(options.get("batch_size", cfg.training.batch_size))
    per_image = bool(options.get("write_per_image", False))
    device = get_device()

    for split_seed in cfg.split_seeds:
        for dataset in cfg.source_datasets:
            build_splits(cfg, dataset, split_seed)

    edits = frequency_edits(sigmas)
    logger.info(
        "%s: sigmas=%s | split=%s | device=%s", EXPERIMENT, sigmas, split, device
    )

    def make_caches(split_seed: int) -> dict[str, ConditionCache]:
        caches: dict[str, ConditionCache] = {}
        for dataset in cfg.source_datasets:
            frame = load_split(cfg, dataset, split_seed)
            frame = frame[frame["split"] == split].assign(source_dataset=dataset)
            logger.info(
                "%s: building %s %s crops for split%s (%d images x %d conditions)",
                EXPERIMENT, dataset, split, split_seed, len(frame), len(edits) + 1,
            )
            caches[dataset] = ConditionCache(cfg, frame, edits)
        return caches

    predictions = driver.score_model_space(
        cfg, EXPERIMENT, make_caches, device,
        source_run=source_run, batch_size=batch_size,
    )
    summary = driver.write(cfg, EXPERIMENT, predictions, per_image=per_image)
    if not summary.empty:
        driver.log_headline(summary, eligibility="test")
