"""PyTorch ``Dataset`` backed by the split CSVs.

A split CSV (one per dataset and split seed, written by :mod:`src.data.splits`)
holds every image with its canonical label and split assignment. Columns:
``image_id``, ``relative_path`` (relative to ``data_root``), ``class_label``,
``class_index``, ``split``.

A single dataset class serves every source (the soybean targets and PlantVillage):
the label space is carried by the CSV's ``class_index`` and by the model head, so
no per-dataset subclass is needed. ``__getitem__`` returns ``(image, label,
image_id)``; the id lets per-image predictions be joined across the within- and
cross-dataset evaluations.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from torchvision import transforms as T


class CsvImageDataset(Dataset):
    """Images listed in a split CSV, filtered to one split.

    Parameters
    ----------
    split_csv:
        Path to the split CSV.
    split:
        One of ``"train"``, ``"val"``, ``"test"``.
    data_root:
        Directory the CSV's ``relative_path`` entries are relative to.
    transform:
        Torchvision transform applied to each image.
    """

    def __init__(
        self,
        split_csv: str | Path,
        split: str,
        data_root: str | Path,
        transform: "T.Compose",
    ) -> None:
        frame = pd.read_csv(split_csv)
        self.df = frame[frame["split"] == split].reset_index(drop=True)
        self.data_root = Path(data_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        image = Image.open(self.data_root / row["relative_path"]).convert("RGB")
        image = self.transform(image)
        return image, int(row["class_index"]), str(row["image_id"])
