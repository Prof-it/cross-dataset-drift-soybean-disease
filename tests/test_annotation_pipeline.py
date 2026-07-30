"""Tests for the annotation batch selection and LabelMe mask conversion.

CPU-only and dataset-free. The concerns worth pinning down are the ones that would
silently corrupt the background intervention: selecting images that are not held
out, re-asking for images that are already annotated, and writing a mask that does
not line up with its source image.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.data.interventions import mask_relative_name

select_batch = pytest.importorskip("scripts.select_annotation_batch")
convert_masks = pytest.importorskip("scripts.convert_masks")

DATASET = "asdid"
FOLDER = "ASDID/frogeye"


def _write_splits(cfg, assignments: dict[int, list[str]], stems: list[str]) -> None:
    """Write one split CSV per seed; ``assignments`` maps seed -> split per stem."""
    cfg.paths.splits_dir.mkdir(parents=True, exist_ok=True)
    for seed, splits in assignments.items():
        rows = [
            {
                "image_id": f"{cfg.data.class_names[i % cfg.data.num_classes]}/{stem}",
                "relative_path": f"{FOLDER}/{stem}.png",
                "class_label": cfg.data.class_names[i % cfg.data.num_classes],
                "class_index": i % cfg.data.num_classes,
                "split": splits[i],
            }
            for i, stem in enumerate(stems)
        ]
        pd.DataFrame(rows).to_csv(
            cfg.paths.splits_dir / f"{DATASET}_splits_seed{seed}.csv", index=False
        )


def _write_images(cfg, stems: list[str], size: tuple[int, int] = (40, 30)) -> None:
    directory = cfg.paths.data_root / FOLDER
    directory.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        Image.fromarray(
            np.full((size[1], size[0], 3), 100, dtype=np.uint8)
        ).save(directory / f"{stem}.png")


@pytest.fixture
def single_dataset_cfg(train_cfg, monkeypatch):
    """A config with one source dataset and three split seeds, over 9 images."""
    from dataclasses import replace

    cfg = replace(train_cfg, source_datasets=(DATASET,), split_seeds=(73, 7, 21))
    # build_splits would rescan data/raw; the CSVs are written directly here.
    monkeypatch.setattr(select_batch, "build_splits", lambda *a, **k: None)
    monkeypatch.setattr(convert_masks, "build_splits", lambda *a, **k: None)
    return cfg


# --------------------------------------------------------------------------- #
# Batch selection                                                             #
# --------------------------------------------------------------------------- #
def test_selection_only_offers_held_out_images(single_dataset_cfg):
    cfg = single_dataset_cfg
    stems = [f"img{i}" for i in range(9)]
    _write_images(cfg, stems)
    # Under seed 73 the first three images train and the rest are held out.
    _write_splits(cfg, {
        73: ["train"] * 3 + ["val", "test"] * 3,
        7: ["test"] * 9,
        21: ["train"] * 9,
    }, stems)

    batch = select_batch.run(cfg, 73, per_class=5, splits=("val", "test"), dry_run=True)
    assert not batch.empty
    assert set(batch["split"]) <= {"val", "test"}
    assert not {"frogeye_leaf_spot/img0", "healthy/img0"} & set(batch["image_id"])


def test_selection_skips_images_that_already_have_a_mask(single_dataset_cfg):
    cfg = single_dataset_cfg
    stems = [f"img{i}" for i in range(9)]
    _write_images(cfg, stems)
    _write_splits(cfg, dict.fromkeys((73, 7, 21), ["test"] * 9), stems)

    frame = pd.read_csv(cfg.paths.splits_dir / f"{DATASET}_splits_seed73.csv")
    done = frame.iloc[0]
    mask = cfg.paths.masks_dir / mask_relative_name(
        DATASET, done["class_label"], done["relative_path"]
    )
    mask.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((30, 40), 255, dtype=np.uint8), mode="L").save(mask)

    batch = select_batch.run(cfg, 73, per_class=3, splits=("val", "test"), dry_run=True)
    assert done["image_id"] not in set(batch["image_id"])
    # The already-annotated image counts toward the target, so its class asks for one fewer.
    per_class = batch["class_label"].value_counts()
    assert per_class[done["class_label"]] == per_class.drop(done["class_label"]).max() - 1


def test_selection_prefers_images_reusable_at_other_seeds(single_dataset_cfg):
    cfg = single_dataset_cfg
    stems = [f"img{i}" for i in range(9)]
    _write_images(cfg, stems)
    # Every image is held out at seed 73; only images 0-2 are held out elsewhere.
    _write_splits(cfg, {
        73: ["test"] * 9,
        7: ["test"] * 3 + ["train"] * 6,
        21: ["test"] * 3 + ["train"] * 6,
    }, stems)

    batch = select_batch.run(cfg, 73, per_class=1, splits=("val", "test"), dry_run=True)
    assert (batch["reusable_seeds"] == 2).all(), "the most reusable candidate must win"


def test_selection_copies_images_and_writes_a_manifest(single_dataset_cfg):
    cfg = single_dataset_cfg
    stems = [f"img{i}" for i in range(9)]
    _write_images(cfg, stems)
    _write_splits(cfg, dict.fromkeys((73, 7, 21), ["test"] * 9), stems)

    batch = select_batch.run(cfg, 73, per_class=2, splits=("val", "test"), dry_run=False)
    manifest = cfg.paths.masks_dir / "annotation_manifest.csv"
    assert manifest.exists()
    assert len(pd.read_csv(manifest)) == len(batch)
    for row in batch.itertuples(index=False):
        assert (cfg.paths.masks_dir / DATASET / f"{row.image_id.split('/')[-1]}.png").exists()


# --------------------------------------------------------------------------- #
# Mask conversion                                                             #
# --------------------------------------------------------------------------- #
def _write_labelme(cfg, stem: str, points, width: int, height: int, label: str = "Leaf") -> None:
    directory = cfg.paths.masks_dir / DATASET
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{stem}.json").write_text(json.dumps({
        "imageWidth": width, "imageHeight": height,
        "shapes": [{"label": label, "shape_type": "polygon", "points": points}],
    }))


def test_conversion_writes_a_binary_mask_matching_the_image(single_dataset_cfg):
    cfg = single_dataset_cfg
    _write_images(cfg, ["img0"], size=(40, 30))
    _write_splits(cfg, {73: ["test"], 7: ["test"], 21: ["test"]}, ["img0"])
    _write_labelme(cfg, "img0", [[0, 0], [20, 0], [20, 30], [0, 30]], 40, 30)

    summary = convert_masks.run(cfg, 73, overwrite=False)
    assert (summary["status"] == "written").all()

    frame = pd.read_csv(cfg.paths.splits_dir / f"{DATASET}_splits_seed73.csv").iloc[0]
    mask = np.asarray(Image.open(
        cfg.paths.masks_dir
        / mask_relative_name(DATASET, frame["class_label"], frame["relative_path"])
    ))
    assert mask.shape == (30, 40)
    assert set(np.unique(mask)) == {0, 255}
    assert mask[:, :10].all() and not mask[:, 30:].any()


def test_conversion_rejects_a_mask_that_does_not_fit_its_image(single_dataset_cfg):
    """A transposed annotation must be caught, not written."""
    cfg = single_dataset_cfg
    _write_images(cfg, ["img0"], size=(40, 30))
    _write_splits(cfg, {73: ["test"], 7: ["test"], 21: ["test"]}, ["img0"])
    _write_labelme(cfg, "img0", [[0, 0], [10, 0], [10, 20], [0, 20]], 30, 40)  # axes swapped

    summary = convert_masks.run(cfg, 73, overwrite=False)
    assert summary.empty, "a misaligned mask must not be written"


@pytest.mark.parametrize(
    ("points", "label"),
    [
        ([[0, 0], [20, 0], [20, 30], [0, 30]], "leaf"),   # wrong label case
        ([[0, 0], [20, 0]], "Leaf"),                      # too few points
    ],
)
def test_conversion_rejects_invalid_annotations(single_dataset_cfg, points, label):
    cfg = single_dataset_cfg
    _write_images(cfg, ["img0"], size=(40, 30))
    _write_splits(cfg, {73: ["test"], 7: ["test"], 21: ["test"]}, ["img0"])
    _write_labelme(cfg, "img0", points, 40, 30, label=label)

    assert convert_masks.run(cfg, 73, overwrite=False).empty


def test_conversion_keeps_existing_masks_unless_overwritten(single_dataset_cfg):
    cfg = single_dataset_cfg
    _write_images(cfg, ["img0"], size=(40, 30))
    _write_splits(cfg, {73: ["test"], 7: ["test"], 21: ["test"]}, ["img0"])
    _write_labelme(cfg, "img0", [[0, 0], [20, 0], [20, 30], [0, 30]], 40, 30)

    convert_masks.run(cfg, 73, overwrite=False)
    again = convert_masks.run(cfg, 73, overwrite=False)
    assert (again["status"] == "kept").all()
    assert (convert_masks.run(cfg, 73, overwrite=True)["status"] == "written").all()
