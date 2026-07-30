"""Configuration loading and runtime helpers.

YAML files under ``configs/`` are the single source of truth for all values.
This module loads and merges them into typed, frozen dataclasses that define only
the *shape* of the configuration (no default values live here, so there is
nothing to keep in sync with the YAML). It also provides the small runtime
helpers shared across experiments: seeding, device selection, and paths.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:  # avoid importing torch at config-load time
    import torch

# Repository root: this file is src/config.py, so two levels up is the repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"


# --------------------------------------------------------------------------- #
# Typed configuration (schema only; values come from YAML)                    #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Schedule:
    """ReduceLROnPlateau settings, stepped on validation loss."""

    factor: float
    patience: int
    min_lr: float


@dataclass(frozen=True)
class EarlyStopping:
    """Early-stopping settings, monitored on validation loss."""

    patience: int
    min_delta: float


@dataclass(frozen=True)
class Training:
    """Optimization and training-loop hyperparameters."""

    batch_size: int
    learning_rate: float
    weight_decay: float
    adam_betas: tuple[float, float]
    max_epochs: int
    label_smoothing: float
    class_weighting: bool
    amp: bool
    num_workers: int
    schedule: Schedule
    early_stopping: EarlyStopping


@dataclass(frozen=True)
class Evaluation:
    """Evaluation and uncertainty-estimation settings."""

    bootstrap_iterations: int
    ece_bins: int


@dataclass(frozen=True)
class Figures:
    """Shared plotting style (applied via ``src.viz``)."""

    dpi: int
    format: str
    font_size: float
    palette: dict[str, str]


@dataclass(frozen=True)
class Augmentation:
    """Train-time augmentation (val/test are deterministic)."""

    rotation_degrees: float
    translate: tuple[float, float]
    brightness: tuple[float, float]


@dataclass(frozen=True)
class Data:
    """Input preprocessing, label space, and the dataset registry."""

    input_size: int
    resize_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    augmentation: Augmentation
    class_names: tuple[str, ...]
    num_classes_plantvillage: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    datasets: dict[str, Any]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def class_to_idx(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.class_names)}


@dataclass
class Paths:
    """Filesystem locations. Mutable so they can be retargeted on Colab.

    Paths are derived from the repository root (or, on Colab, from local SSD and
    Drive roots via ``colab.resolve_paths``); they are not stored in YAML.
    """

    data_root: Path
    splits_dir: Path
    class_weights_path: Path
    checkpoints_dir: Path
    results_dir: Path
    logs_dir: Path
    # Foreground masks for the background intervention. Not version-controlled
    # (see .gitignore) and read by that one experiment, so the field is optional:
    # left unset it lands beside the split CSVs, which keeps a retargeted ``Paths``
    # (Colab, tests) self-consistent instead of silently pointing back at the repo.
    masks_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.masks_dir is None:
            self.masks_dir = Path(self.splits_dir).parent / "masks"

    @classmethod
    def default(cls, root: Path = REPO_ROOT) -> "Paths":
        return cls(
            data_root=root / "data" / "raw",
            splits_dir=root / "data" / "splits",
            class_weights_path=root / "data" / "class_weights.json",
            checkpoints_dir=root / "checkpoints",
            results_dir=root / "results",
            logs_dir=root / "logs",
            masks_dir=root / "data" / "masks",
        )

    def with_overrides(self, **overrides: "str | Path | None") -> "Paths":
        """Return a copy with the given path fields replaced (ignoring ``None``)."""
        from dataclasses import replace

        changes = {key: Path(value) for key, value in overrides.items() if value is not None}
        return replace(self, **changes)


@dataclass(frozen=True)
class Config:
    """Top-level configuration assembled from the YAML files."""

    seeds: tuple[int, ...]
    split_seeds: tuple[int, ...]
    plantvillage_split_seed: int
    seed_design: str
    run_name: str
    architectures: dict[str, Any]
    training_paths: tuple[str, ...]
    source_datasets: tuple[str, ...]
    training: Training
    evaluation: Evaluation
    figures: Figures
    data: Data
    paths: Paths
    extras: dict[str, Any]  # experiment-specific keys not part of the typed core


# --------------------------------------------------------------------------- #
# Loading                                                                     #
# --------------------------------------------------------------------------- #
def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (override wins)."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(
    *overrides: str | Path | dict[str, Any],
    configs_dir: Path = CONFIGS_DIR,
    paths: Paths | None = None,
) -> Config:
    """Load and merge configuration into a typed :class:`Config`.

    Reads ``default.yaml`` and ``datasets.yaml`` from ``configs_dir``, then
    applies any number of overrides in order (each a path to an experiment YAML
    or a plain dict). Later overrides win. Missing keys raise ``KeyError`` rather
    than falling back to a hidden default, so the YAML stays authoritative.
    """
    merged = _read_yaml(configs_dir / "default.yaml")
    merged = _deep_merge(merged, _read_yaml(configs_dir / "datasets.yaml"))
    for override in overrides:
        extra = override if isinstance(override, dict) else _read_yaml(Path(override))
        merged = _deep_merge(merged, extra)
    return _build_config(merged, paths or Paths.default())


_CORE_KEYS = frozenset({
    "seeds", "split_seeds", "plantvillage_split_seed", "seed_design", "run_name",
    "architectures", "training_paths", "source_datasets", "preprocessing", "training",
    "evaluation", "figures", "classes", "num_classes_plantvillage", "split", "datasets",
})


def _build_config(cfg: dict[str, Any], paths: Paths) -> Config:
    train = cfg["training"]
    preprocessing = cfg["preprocessing"]
    split = cfg["split"]
    evaluation = cfg["evaluation"]
    figures = cfg["figures"]

    return Config(
        seeds=tuple(cfg["seeds"]),
        split_seeds=tuple(cfg["split_seeds"]),
        plantvillage_split_seed=int(cfg["plantvillage_split_seed"]),
        seed_design=str(cfg["seed_design"]),
        run_name=str(cfg["run_name"]),
        architectures=cfg["architectures"],
        training_paths=tuple(cfg["training_paths"]),
        source_datasets=tuple(cfg["source_datasets"]),
        training=Training(
            batch_size=int(train["batch_size"]),
            learning_rate=float(train["learning_rate"]),
            weight_decay=float(train["weight_decay"]),
            adam_betas=tuple(train["adam_betas"]),  # type: ignore[arg-type]
            max_epochs=int(train["max_epochs"]),
            label_smoothing=float(train["label_smoothing"]),
            class_weighting=bool(train["class_weighting"]),
            amp=bool(train["amp"]),
            num_workers=int(train["num_workers"]),
            schedule=Schedule(**train["schedule"]),
            early_stopping=EarlyStopping(**train["early_stopping"]),
        ),
        evaluation=Evaluation(
            bootstrap_iterations=int(evaluation["bootstrap_iterations"]),
            ece_bins=int(evaluation["ece_bins"]),
        ),
        figures=Figures(
            dpi=int(figures["dpi"]),
            format=str(figures["format"]),
            font_size=float(figures["font_size"]),
            palette=dict(figures["palette"]),
        ),
        data=Data(
            input_size=int(preprocessing["input_size"]),
            resize_size=int(preprocessing["resize_size"]),
            mean=tuple(preprocessing["mean"]),  # type: ignore[arg-type]
            std=tuple(preprocessing["std"]),  # type: ignore[arg-type]
            augmentation=Augmentation(
                rotation_degrees=float(preprocessing["augmentation"]["rotation_degrees"]),
                translate=tuple(preprocessing["augmentation"]["translate"]),  # type: ignore[arg-type]
                brightness=tuple(preprocessing["augmentation"]["brightness"]),  # type: ignore[arg-type]
            ),
            class_names=tuple(cfg["classes"]),
            num_classes_plantvillage=int(cfg["num_classes_plantvillage"]),
            train_ratio=float(split["train"]),
            val_ratio=float(split["val"]),
            test_ratio=float(split["test"]),
            datasets=cfg["datasets"],
        ),
        paths=paths,
        extras={k: v for k, v in cfg.items() if k not in _CORE_KEYS},
    )


# --------------------------------------------------------------------------- #
# Runtime helpers                                                             #
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for deterministic runs."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> "torch.device":
    """Return the best available device: CUDA, then Apple MPS, then CPU."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def model_id(arch: str, path: str, dataset: str) -> str:
    """Canonical identifier for a trained model, e.g. ``resnet50_direct_asdid``."""
    return f"{arch}_{path}_{dataset}"
