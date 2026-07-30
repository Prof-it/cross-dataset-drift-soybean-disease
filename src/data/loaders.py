"""DataLoader construction.

Assembles seeded, reproducible DataLoaders from the split CSVs and transforms,
centralizing the Colab-relevant knobs (worker count, pinned memory, deterministic
shuffling) in one place. The training loader shuffles with a generator seeded by
the training ``seed``; pinned memory is disabled on Apple MPS.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader

from src.data.datasets import CsvImageDataset
from src.data.splits import split_csv_path
from src.data.transforms import build_transforms

if TYPE_CHECKING:
    from src.config import Config


def make_dataloaders(
    cfg: "Config",
    dataset: str,
    split_seed: int,
    seed: int,
) -> dict[str, DataLoader]:
    """Build train/val/test DataLoaders for ``dataset`` under the given seeds.

    ``dataset`` is a registry key (``asdid``, ``mh``, ``plantvillage``) or a
    derived split such as ``asdid_matched`` (control study); the latter reuses the
    soybean dataset class and label space.
    """
    csv = split_csv_path(cfg.paths, dataset, split_seed)
    pin_memory = not torch.backends.mps.is_available()

    loaders: dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        is_train = split == "train"
        ds = CsvImageDataset(
            csv,
            split=split,
            data_root=cfg.paths.data_root,
            transform=build_transforms(cfg, train=is_train),
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=cfg.training.batch_size,
            shuffle=is_train,
            num_workers=cfg.training.num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            generator=torch.Generator().manual_seed(seed) if is_train else None,
        )
    return loaders


def make_eval_loader(
    cfg: "Config",
    dataset: str,
    split_seed: int,
    split: str,
) -> DataLoader:
    """A single, non-shuffled DataLoader for one ``split`` using the eval transform.

    Unlike :func:`make_dataloaders`, the train split is read *without* augmentation,
    so it is suitable for deterministic feature caching in linear probing.
    """
    csv = split_csv_path(cfg.paths, dataset, split_seed)
    ds = CsvImageDataset(
        csv,
        split=split,
        data_root=cfg.paths.data_root,
        transform=build_transforms(cfg, train=False),
    )
    return DataLoader(
        ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=not torch.backends.mps.is_available(),
        drop_last=False,
    )
