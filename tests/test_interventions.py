"""Tests for the input-level interventions (background replacement, low-pass).

CPU-only and dataset-free: images and masks are synthesized into ``tmp_path`` and
a stub model stands in for a checkpoint, so these exercise the pixel edits, the
mask lookup, the cache's normalization contract, and the split-aware summary
without touching the real data or the trained models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.data.interventions import (
    ORIGINAL,
    background_edits,
    low_pass,
    mask_relative_name,
    mask_table,
    replace_background,
    sigma_of,
)


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #
@pytest.fixture
def image_and_mask():
    """A 64x64 image whose left half is foreground, with the matching mask."""
    pixels = np.full((64, 64, 3), 200, dtype=np.uint8)
    foreground = np.zeros((64, 64), dtype=np.uint8)
    foreground[:, :32] = 255
    return Image.fromarray(pixels), Image.fromarray(foreground, mode="L")


def _write_dataset(cfg, dataset: str, folder: str, subfolder: str, stems: list[str]) -> None:
    """Write synthetic images plus their masks and a split CSV for ``dataset``."""
    images = cfg.paths.data_root / folder / subfolder
    images.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, stem in enumerate(stems):
        Image.fromarray(np.full((64, 64, 3), 120, dtype=np.uint8)).save(images / f"{stem}.png")
        label = cfg.data.class_names[i % cfg.data.num_classes]
        relative = f"{folder}/{subfolder}/{stem}.png"
        mask = cfg.paths.masks_dir / mask_relative_name(dataset, label, relative)
        mask.parent.mkdir(parents=True, exist_ok=True)
        foreground = np.zeros((64, 64), dtype=np.uint8)
        foreground[:, :32] = 255
        Image.fromarray(foreground, mode="L").save(mask)
        rows.append({
            "image_id": f"{label}/{stem}", "relative_path": relative,
            "class_label": label, "class_index": cfg.data.class_to_idx[label],
            "split": ("train", "val", "test")[i % 3],
        })
    cfg.paths.splits_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        cfg.paths.splits_dir / f"{dataset}_splits_seed{cfg.split_seeds[0]}.csv", index=False
    )


# --------------------------------------------------------------------------- #
# Pixel edits                                                                 #
# --------------------------------------------------------------------------- #
def test_replace_background_only_touches_background(image_and_mask):
    image, mask = image_and_mask
    out = np.array(replace_background(image, mask, (127, 127, 127)))
    assert (out[:, :32] == 200).all(), "foreground must be untouched"
    assert (out[:, 32:] == 127).all(), "background must take the fill value"


def test_replace_background_rejects_size_mismatch(image_and_mask):
    image, _ = image_and_mask
    with pytest.raises(ValueError, match="differ in size"):
        replace_background(image, Image.new("L", (32, 32)), (0, 0, 0))


def test_low_pass_smooths_an_edge_more_at_larger_sigma():
    pixels = np.zeros((64, 64, 3), dtype=np.uint8)
    pixels[:, 32:] = 255
    image = Image.fromarray(pixels)

    def edge_energy(img) -> float:
        arr = np.asarray(img, dtype=np.float32)[:, :, 0]
        return float(np.abs(np.diff(arr, axis=1)).max())

    assert edge_energy(low_pass(image, 8)) < edge_energy(low_pass(image, 2)) < edge_energy(image)


def test_low_pass_is_a_no_op_at_sigma_zero(image_and_mask):
    image, _ = image_and_mask
    assert np.array_equal(np.array(low_pass(image, 0)), np.array(image))


def test_sigma_of_reads_the_condition_name():
    assert sigma_of(ORIGINAL) == 0.0
    assert sigma_of("low_sigma_8") == 8.0


# --------------------------------------------------------------------------- #
# Mask lookup                                                                 #
# --------------------------------------------------------------------------- #
def test_mask_relative_name_matches_the_annotation_layout():
    name = mask_relative_name("asdid", "frogeye_leaf_spot", "ASDID/frogeye/frogeye_1026.jpg")
    assert name == "asdid/frogeye_leaf_spot/asdid_frogeye_1026_mask.png"


def test_mask_table_keeps_only_annotated_rows(train_cfg):
    _write_dataset(train_cfg, "asdid", "ASDID", "frogeye", ["a", "b", "c"])
    # A fourth image with no mask must be dropped rather than silently scored.
    extra = train_cfg.paths.data_root / "ASDID" / "frogeye" / "d.png"
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(extra)
    csv = train_cfg.paths.splits_dir / f"asdid_splits_seed{train_cfg.split_seeds[0]}.csv"
    frame = pd.read_csv(csv)
    frame.loc[len(frame)] = {
        "image_id": "healthy/d", "relative_path": "ASDID/frogeye/d.png",
        "class_label": "healthy", "class_index": 0, "split": "test",
    }
    frame.to_csv(csv, index=False)

    table = mask_table(train_cfg, "asdid", train_cfg.split_seeds[0])
    assert len(table) == 3
    assert "healthy/d" not in set(table["image_id"])
    assert (table["source_dataset"] == "asdid").all()
    assert table["split"].isin({"train", "val", "test"}).all()


def test_mask_table_raises_when_nothing_is_annotated(train_cfg):
    _write_dataset(train_cfg, "asdid", "ASDID", "frogeye", ["a"])
    for mask in train_cfg.paths.masks_dir.rglob("*.png"):
        mask.unlink()
    with pytest.raises(FileNotFoundError, match="no masks found"):
        mask_table(train_cfg, "asdid", train_cfg.split_seeds[0])


# --------------------------------------------------------------------------- #
# Condition cache                                                             #
# --------------------------------------------------------------------------- #
def test_cache_batches_match_the_standard_eval_transform(train_cfg):
    """The cache must reproduce ``build_transforms(train=False)`` exactly.

    The cache stores uint8 crops and normalizes at batch time; that is only a
    memory optimization and must not change a single pixel of what the model sees.
    """
    pytest.importorskip("torchvision")
    from src.data.interventions import ConditionCache
    from src.data.transforms import build_transforms

    _write_dataset(train_cfg, "asdid", "ASDID", "frogeye", ["a", "b"])
    frame = mask_table(train_cfg, "asdid", train_cfg.split_seeds[0])
    cache = ConditionCache(train_cfg, frame, {})

    reference = build_transforms(train_cfg, train=False)
    images, labels, ids = next(cache.batches(ORIGINAL, batch_size=8))
    for i, image_id in enumerate(ids):
        row = frame[frame["image_id"] == image_id].iloc[0]
        expected = reference(
            Image.open(train_cfg.paths.data_root / row["relative_path"]).convert("RGB")
        )
        assert np.allclose(images[i].numpy(), expected.numpy(), atol=1e-6)
    assert list(labels) == list(frame["class_index"])


def test_cache_holds_one_crop_per_image_and_condition(train_cfg):
    pytest.importorskip("torchvision")
    from src.data.interventions import ConditionCache

    _write_dataset(train_cfg, "asdid", "ASDID", "frogeye", ["a", "b", "c"])
    frame = mask_table(train_cfg, "asdid", train_cfg.split_seeds[0])
    cache = ConditionCache(train_cfg, frame, background_edits({"grey": (127, 127, 127)}))

    assert cache.conditions == (ORIGINAL, "grey")
    assert len(cache) == len(frame) * 2
    grey, _, _ = next(cache.batches("grey", batch_size=8))
    original, _, _ = next(cache.batches(ORIGINAL, batch_size=8))
    assert not np.allclose(grey.numpy(), original.numpy()), "the fill must change the crop"


# --------------------------------------------------------------------------- #
# Summary                                                                     #
# --------------------------------------------------------------------------- #
def _prediction_rows(eval_split: str, correct_original: bool, correct_grey: bool) -> list[dict]:
    base = {
        "experiment": "background_intervention", "arch": "resnet50", "path": "direct",
        "train_dataset": "asdid", "eval_dataset": "mh", "direction": "cross",
        "seed": 7, "split_seed": 73, "model_id": "resnet50_direct_asdid",
        "class_label": "healthy", "class_index": 0, "eval_split": eval_split,
    }
    return [
        {**base, "image_id": f"i_{eval_split}_{correct_original}", "condition": ORIGINAL,
         "predicted_index": 0 if correct_original else 1, "confidence": 0.9,
         "correct": correct_original},
        {**base, "image_id": f"i_{eval_split}_{correct_original}", "condition": "grey",
         "predicted_index": 0 if correct_grey else 1, "confidence": 0.9,
         "correct": correct_grey},
    ]


def test_summarize_separates_eligibility_policies():
    from src.experiments.robustness._intervention_driver import summarize

    predictions = pd.DataFrame(
        _prediction_rows("train", False, False)
        + _prediction_rows("val", True, True)
        + _prediction_rows("test", True, True)
    )
    summary = summarize(predictions, ("healthy", "rust", "frogeye_leaf_spot"))
    counts = summary.set_index(["eligibility", "condition"])["n_images"]
    assert counts[("all", ORIGINAL)] == 3
    assert counts[("heldout", ORIGINAL)] == 2, "val + test"
    assert counts[("test", ORIGINAL)] == 1
    # The train-split image is wrong under both conditions, so dropping it must
    # raise accuracy: this is exactly the leakage-vs-held-out distinction.
    accuracy = summary.set_index(["eligibility", "condition"])["accuracy"]
    assert accuracy[("all", ORIGINAL)] < accuracy[("heldout", ORIGINAL)] == 1.0


def test_deltas_are_measured_against_the_original_condition():
    from src.experiments.robustness._intervention_driver import deltas, summarize

    predictions = pd.DataFrame(
        _prediction_rows("test", False, True) + _prediction_rows("val", False, True)
    )
    summary = summarize(predictions, ("healthy", "rust", "frogeye_leaf_spot"))
    changes = deltas(summary)
    assert (changes["condition"] == "grey").all()
    assert (changes["delta_accuracy"] > 0).all(), "grey fixed both images"


# --------------------------------------------------------------------------- #
# Masks annotated at a different resolution than the image                    #
# --------------------------------------------------------------------------- #
def test_is_pure_rescale_absorbs_resize_rounding():
    from src.data.interventions import is_pure_rescale

    # The two real cases: a 512px short side lands on 683x512 and 768x512.
    assert is_pure_rescale((4000, 3000), (683, 512))
    assert is_pure_rescale((5472, 3648), (768, 512))
    assert is_pure_rescale((683, 512), (683, 512))
    # A transpose or a differently cropped image is not a rescale.
    assert not is_pure_rescale((3000, 4000), (683, 512))
    assert not is_pure_rescale((4000, 3000), (512, 512))


def test_conform_rescales_a_full_resolution_mask_and_stays_binary(tmp_path):
    """Thesis masks were drawn on the originals; the pipeline reads 512px copies."""
    from src.data.interventions import conform_mask_to_image

    image_path = tmp_path / "img.png"
    Image.fromarray(np.zeros((512, 683, 3), dtype=np.uint8)).save(image_path)
    image = Image.open(image_path)

    full = np.zeros((3000, 4000), dtype=np.uint8)
    full[:, :2000] = 255                       # left half foreground
    mask, rescaled = conform_mask_to_image(
        image_path, image, Image.fromarray(full, mode="L")
    )

    assert rescaled is True
    assert mask.size == image.size
    out = np.asarray(mask)
    assert set(np.unique(out)) == {0, 255}, "must stay strictly binary"
    assert out[:, :330].all() and not out[:, 350:].any(), "left half must survive"
    assert abs((out == 255).mean() - 0.5) < 0.01, "coverage must be preserved"


def test_conform_is_a_no_op_when_sizes_already_agree(tmp_path):
    from src.data.interventions import conform_mask_to_image

    image_path = tmp_path / "img.png"
    Image.fromarray(np.zeros((30, 40, 3), dtype=np.uint8)).save(image_path)
    image = Image.open(image_path)
    original = np.zeros((30, 40), dtype=np.uint8)
    original[:, :20] = 255

    mask, rescaled = conform_mask_to_image(
        image_path, image, Image.fromarray(original, mode="L")
    )
    assert rescaled is False
    assert np.array_equal(np.asarray(mask), original)


def test_conform_refuses_a_mask_whose_aspect_ratio_differs(tmp_path):
    """Stretching a mask from a differently shaped image would mask the wrong pixels."""
    from src.data.interventions import conform_mask_to_image

    image_path = tmp_path / "img.png"
    Image.fromarray(np.zeros((512, 683, 3), dtype=np.uint8)).save(image_path)
    image = Image.open(image_path)
    transposed = Image.fromarray(np.full((4000, 3000), 255, dtype=np.uint8), mode="L")

    with pytest.raises(ValueError, match="more than a uniform rescale"):
        conform_mask_to_image(image_path, image, transposed)
