"""Shared helpers for experiment drivers: checkpoint paths, run ids, class weights.

Keeps the checkpoint-naming policy and the class-weight lookup in one place so the
individual experiment drivers stay short and consistent.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from src.data.splits import class_weights_by_index, load_split

if TYPE_CHECKING:
    from src.config import Config


def run_dir(cfg: "Config", experiment: str, model_id: str, split_seed: int, seed: int) -> Path:
    """Checkpoint directory for one run: ``<ckpt>/<experiment>/<model_id>/split<S>/seed<s>``."""
    return cfg.paths.checkpoints_dir / experiment / model_id / f"split{split_seed}" / f"seed{seed}"


def pv_dir(cfg: "Config", arch: str) -> Path:
    """Canonical, shared PlantVillage stage-1 checkpoint (one per architecture).

    PlantVillage pretraining uses a fixed split and a fixed init seed
    (``cfg.plantvillage_split_seed``) and does not depend on the soybean target,
    split, or run seed, so stage-1 is trained once per architecture and reused by
    every two-stage run and downstream experiment.
    """
    pv = cfg.plantvillage_split_seed
    return run_dir(cfg, "finetune", f"{arch}_twostage_pv", pv, pv)


def seed_pairs(cfg: "Config") -> list[tuple[int, int]]:
    """(split_seed, init_seed) pairs to run, per ``cfg.seed_design``.

    ``diagonal`` pairs ``split_seeds[i]`` with ``seeds[i]`` (requires equal length);
    ``cross`` is the full Cartesian product.
    """
    if cfg.seed_design == "diagonal":
        if len(cfg.split_seeds) != len(cfg.seeds):
            raise ValueError("diagonal seed_design requires equal-length split_seeds and seeds")
        return list(zip(cfg.split_seeds, cfg.seeds))
    if cfg.seed_design == "cross":
        return [(sp, se) for sp in cfg.split_seeds for se in cfg.seeds]
    raise ValueError(f"unknown seed_design: {cfg.seed_design!r}")


def run_id(model_id: str, split_seed: int, seed: int) -> str:
    """Identifier for TensorBoard logs and console messages."""
    return f"{model_id}_split{split_seed}_seed{seed}"


def class_weights(cfg: "Config", dataset: str, split_seed: int, num_classes: int) -> list[float] | None:
    """Inverse-frequency weights for ``dataset``, or ``None`` if weighting is off."""
    if not cfg.training.class_weighting:
        return None
    return class_weights_by_index(load_split(cfg, dataset, split_seed), num_classes)
