"""Shared pytest fixtures.

Cheap, dependency-light fixtures so tests run fast on CPU. The synthetic loaders
yield random tensors with the real shapes, exercising the training/evaluation
paths without the actual datasets. Fixtures needing torch skip if it is absent.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def tmp_paths(tmp_path):
    from src.config import Paths

    return Paths(
        data_root=tmp_path / "data" / "raw",
        splits_dir=tmp_path / "data" / "splits",
        class_weights_path=tmp_path / "data" / "class_weights.json",
        checkpoints_dir=tmp_path / "checkpoints",
        results_dir=tmp_path / "results",
        logs_dir=tmp_path / "logs",
    )


@pytest.fixture
def train_cfg(tmp_paths):
    """A tiny config (1 epoch, batch 4, no workers, no AMP) writing into tmp paths."""
    from src.config import load_config

    return load_config(
        {"training": {"max_epochs": 1, "batch_size": 4, "num_workers": 0, "amp": False}},
        paths=tmp_paths,
    )


@pytest.fixture
def synthetic_loaders(train_cfg):
    """Train/val/test DataLoaders over random images with the real shapes."""
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader, Dataset

    num_classes = train_cfg.data.num_classes
    size = train_cfg.data.input_size

    class _Synthetic(Dataset):
        def __init__(self, n: int) -> None:
            self.n = n

        def __len__(self) -> int:
            return self.n

        def __getitem__(self, idx: int):
            return torch.randn(3, size, size), idx % num_classes, f"img{idx}"

    def loader(n: int, shuffle: bool) -> DataLoader:
        return DataLoader(_Synthetic(n), batch_size=train_cfg.training.batch_size, shuffle=shuffle)

    return {"train": loader(8, True), "val": loader(4, False), "test": loader(4, False)}
