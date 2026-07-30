"""Stratified splits, class weights, and the control-study subsample.

Every data-partition artifact is derived deterministically from the raw folders
so results are reproducible and the partition can be varied by split seed. The
split CSVs and class-weights JSON these functions write are the inputs the rest
of the pipeline consumes; raw images themselves are not version-controlled.

CSV columns: ``image_id``, ``relative_path`` (relative to ``data_root``),
``class_label``, ``class_index``, ``split``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.config import Config, Paths

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def split_csv_path(paths: "Paths", dataset: str, split_seed: int) -> Path:
    """Canonical location of a dataset's split CSV for a given split seed."""
    return paths.splits_dir / f"{dataset}_splits_seed{split_seed}.csv"


def _list_images(directory: Path) -> list[Path]:
    """All image files under ``directory``, sorted for determinism."""
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def _dataset_records(cfg: "Config", dataset: str) -> list[dict]:
    """Scan the registry folder for ``dataset`` and return per-image records.

    Soybean datasets map their on-disk subfolders to the canonical class names
    and fixed indices. PlantVillage (no subfolder list, no class map) uses its
    folder names as labels and indexes them in sorted order.
    """
    registry = cfg.data.datasets[dataset]
    base = cfg.paths.data_root / registry["folder"]
    subfolders = registry["subfolders"]
    class_map = registry["class_map"]
    records: list[dict] = []

    if subfolders is None:  # PlantVillage: folder names are the labels
        class_dirs = sorted(p for p in base.iterdir() if p.is_dir())
        index = {d.name: i for i, d in enumerate(class_dirs)}
        for class_dir in class_dirs:
            for image in _list_images(class_dir):
                records.append(_record(cfg, image, class_dir.name, index[class_dir.name]))
    else:  # soybean: explicit subfolder -> canonical class map
        for sub in subfolders:
            label = class_map[sub]
            class_index = cfg.data.class_to_idx[label]
            for image in _list_images(base / sub):
                records.append(_record(cfg, image, label, class_index))
    return records


def _record(cfg: "Config", image: Path, label: str, class_index: int) -> dict:
    relative_path = image.relative_to(cfg.paths.data_root).as_posix()
    return {
        "image_id": f"{label}/{image.stem}",
        "relative_path": relative_path,
        "class_label": label,
        "class_index": class_index,
    }


def _assign_splits(records: list[dict], cfg: "Config", split_seed: int) -> pd.DataFrame:
    """Per-class stratified train/val/test assignment under ``split_seed``."""
    df = pd.DataFrame.from_records(records)
    rng = np.random.default_rng(split_seed)
    df["split"] = ""
    for _, group in df.groupby("class_label", sort=True):
        idx = group.index.to_numpy()
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(n * cfg.data.train_ratio))
        n_val = int(round(n * cfg.data.val_ratio))
        df.loc[idx[:n_train], "split"] = "train"
        df.loc[idx[n_train : n_train + n_val], "split"] = "val"
        df.loc[idx[n_train + n_val :], "split"] = "test"
    return df.sort_values(["class_label", "split", "image_id"]).reset_index(drop=True)


def build_splits(cfg: "Config", dataset: str, split_seed: int, overwrite: bool = False) -> Path:
    """Build (or reuse) the stratified split CSV for ``dataset`` and ``split_seed``."""
    out = split_csv_path(cfg.paths, dataset, split_seed)
    if out.exists() and not overwrite:
        return out
    df = _assign_splits(_dataset_records(cfg, dataset), cfg, split_seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


def load_split(cfg: "Config", dataset: str, split_seed: int) -> pd.DataFrame:
    """Read a previously built split CSV."""
    return pd.read_csv(split_csv_path(cfg.paths, dataset, split_seed))


def class_weights_by_index(df: pd.DataFrame, num_classes: int) -> list[float]:
    """Inverse-frequency class weights from the train split, indexed 0..K-1.

    weight(c) = N / (K * n_c), with N the train size, K the number of classes, and
    n_c the train count of class c. Works for any label space (soybean or
    PlantVillage) since it keys on ``class_index``.
    """
    train = df[df["split"] == "train"]
    counts = train["class_index"].value_counts()
    freqs = np.array([counts.get(i, 0) for i in range(num_classes)], dtype=float)
    return (freqs.sum() / (num_classes * np.maximum(freqs, 1.0))).tolist()


def compute_class_weights(df: pd.DataFrame, class_names: tuple[str, ...]) -> list[float]:
    """Inverse-frequency class weights in ``class_names`` (index) order."""
    return class_weights_by_index(df, len(class_names))


def write_class_weights(cfg: "Config", weights_by_dataset: dict[str, list[float]]) -> Path:
    """Persist class weights for all datasets to the configured JSON path."""
    path = cfg.paths.class_weights_path
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {ds: dict(zip(cfg.data.class_names, w)) for ds, w in weights_by_dataset.items()}
    path.write_text(json.dumps(payload, indent=2))
    return path


def build_matched_subsample(
    cfg: "Config",
    split_seed: int,
    source: str = "asdid",
    reference: str = "mh",
    overwrite: bool = False,
) -> Path:
    """Resample ``source`` to match ``reference``'s per-class, per-split counts.

    Produces the control-study dataset (one-directional, ASDID matched to MH): for
    each split and class, draw the reference's count of images from the source's
    own split, so there is no train/test leakage. Written as ``<source>_matched``.
    """
    out = split_csv_path(cfg.paths, f"{source}_matched", split_seed)
    if out.exists() and not overwrite:
        return out
    src = load_split(cfg, source, split_seed)
    ref = load_split(cfg, reference, split_seed)
    rng = np.random.default_rng(split_seed)
    chunks: list[pd.DataFrame] = []
    for split in ("train", "val", "test"):
        for label in cfg.data.class_names:
            n_ref = int(((ref["split"] == split) & (ref["class_label"] == label)).sum())
            pool = src[(src["split"] == split) & (src["class_label"] == label)]
            take = min(n_ref, len(pool))
            chunks.append(pool.sample(n=take, random_state=int(rng.integers(0, 2**31 - 1))))
    matched = pd.concat(chunks).reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    matched.to_csv(out, index=False)
    return out
