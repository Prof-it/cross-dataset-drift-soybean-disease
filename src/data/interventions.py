"""Input-level interventions: background replacement and frequency low-pass.

Both interventions edit the *image*, not the model, and are evaluated by running
already-trained checkpoints over the edited inputs. They answer two different
questions about what the cross-dataset decision rule is keying on:

- **Background replacement** (needs foreground masks): replace every background
  pixel with a uniform fill. If cross-dataset macro F1 *rises* when the background
  is removed, the models were using dataset-specific background context that does
  not transfer.
- **Frequency low-pass** (no masks): Gaussian-blur the image at several sigmas.
  This is the control for the background result: it removes fine texture without
  removing any semantic region, so a null result here says the background effect
  is specific to *what* was removed rather than to *that something* was removed.

Both edits are applied in pixel space at the source resolution, *before* the
deterministic evaluation geometry (resize + center crop), matching the order the
thesis used. The resulting crops are cached as uint8 arrays: normalization happens
at batch time, so one cache is ~6x smaller than the float32 equivalent and is
reused unchanged across every checkpoint and seed.

Public API:
- ``mask_relative_name`` / ``mask_table`` -- locate the foreground masks on disk
  and join them to a split CSV.
- ``replace_background`` / ``low_pass`` -- the two pixel-space edits.
- ``ConditionCache`` -- preprocessed crops for every (image, condition) pair, with
  ``batches`` yielding normalized tensors ready for a forward pass.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from PIL import Image

from src.data.splits import load_split

if TYPE_CHECKING:
    import torch

    from src.config import Config

logger = logging.getLogger(__name__)

ORIGINAL = "original"

# Inverse of the EXIF orientation transform. Masks were annotated on the
# *displayed* (EXIF-applied) image, while ``Image.open`` returns the raw pixel
# layout, so the mask is un-rotated to match before it is applied.
_EXIF_ORIENTATION_TAG = 0x0112
_EXIF_INVERSE: dict[int, int] = {
    2: Image.FLIP_LEFT_RIGHT,
    3: Image.ROTATE_180,
    4: Image.FLIP_TOP_BOTTOM,
    5: Image.TRANSVERSE,
    6: Image.ROTATE_90,
    7: Image.TRANSPOSE,
    8: Image.ROTATE_270,
}


# --------------------------------------------------------------------------- #
# Foreground masks                                                            #
# --------------------------------------------------------------------------- #
def mask_relative_name(dataset: str, class_label: str, relative_path: str) -> str:
    """Location of an image's mask relative to ``paths.masks_dir``.

    Masks are stored as ``<dataset>/<class_label>/<dataset>_<stem>_mask.png``,
    the layout produced by the annotation pipeline (see
    ``data/masks/ANNOTATION_PROTOCOL.md``). ``stem`` is the source image's file
    name without its extension, so the name is derivable from a split CSV row.
    """
    return f"{dataset}/{class_label}/{dataset}_{Path(relative_path).stem}_mask.png"


# EXIF orientations whose transform swaps width and height.
_EXIF_SWAPS_AXES = frozenset({5, 6, 7, 8})


def mask_conforms(image_path: str | Path, mask_path: str | Path) -> "str | None":
    """``None`` if the mask can be used against the image, else why it cannot.

    Reads image headers only (``Image.open`` is lazy), so this is cheap enough to
    screen every annotation before a run. A mask is usable when, after the EXIF
    transpose is undone, it either matches the image exactly or differs from it by
    a uniform rescale.

    The case this exists to catch: masks annotated on the original photographs,
    where the pipeline now reads pre-resized copies. Resizing re-encodes the pixels
    and drops the EXIF orientation tag, so for a photo that was stored rotated
    there is no longer any record of which way to turn the mask back -- two
    different rotations are equally consistent with the dimensions, and picking the
    wrong one masks the wrong half of the leaf while looking perfectly plausible.
    Such masks are reported here so they can be skipped rather than guessed at.
    """
    with Image.open(image_path) as image:
        image_size = image.size
        orientation = int(image.getexif().get(_EXIF_ORIENTATION_TAG, 1))
    with Image.open(mask_path) as mask:
        mask_size = mask.size
    if orientation in _EXIF_SWAPS_AXES:
        mask_size = mask_size[::-1]
    if mask_size == image_size or is_pure_rescale(mask_size, image_size):
        return None
    if is_pure_rescale(mask_size[::-1], image_size):
        return (
            f"mask {mask_size} is a rotated view of the image {image_size} and the "
            f"image carries no EXIF orientation to undo it"
        )
    return f"mask {mask_size} does not match image {image_size}"


def mask_table(cfg: "Config", dataset: str, split_seed: int) -> pd.DataFrame:
    """Split-CSV rows for ``dataset`` that have a foreground mask on disk.

    Returns the split frame filtered to annotated images, with ``mask_path`` and
    ``source_dataset`` added. Only a subset of each dataset is annotated, so this
    is normally far smaller than the full split; the ``split`` column is retained
    because which of those images are held out depends on ``split_seed``.
    """
    frame = load_split(cfg, dataset, split_seed)
    masks_dir = Path(cfg.paths.masks_dir)
    paths = [
        masks_dir / mask_relative_name(dataset, row.class_label, row.relative_path)
        for row in frame.itertuples(index=False)
    ]
    frame = frame.assign(mask_path=paths, source_dataset=dataset)
    annotated = frame[[p.exists() for p in paths]].reset_index(drop=True)
    if annotated.empty:
        raise FileNotFoundError(
            f"no masks found for '{dataset}' under {masks_dir}; expected files like "
            f"{masks_dir / mask_relative_name(dataset, '<class>', '<image>.jpg')}"
        )
    return annotated


def align_mask_to_image(image_path: str | Path, mask: Image.Image) -> Image.Image:
    """Undo the EXIF transpose on ``mask`` so it matches the raw pixel layout."""
    orientation = Image.open(image_path).getexif().get(_EXIF_ORIENTATION_TAG, 1)
    transform = _EXIF_INVERSE.get(int(orientation))
    return mask.transpose(transform) if transform is not None else mask


def is_pure_rescale(
    mask_size: tuple[int, int], image_size: tuple[int, int], tolerance: float = 0.01
) -> bool:
    """Whether two sizes differ only by a uniform scale (same aspect ratio).

    ``tolerance`` is a relative bound on the aspect-ratio difference, loose enough
    to absorb the integer rounding a resize introduces (a 4000x3000 original
    downscaled to a 512px short side lands on 683x512, an aspect ratio 0.05% off).
    """
    mask_ratio = mask_size[0] / mask_size[1]
    image_ratio = image_size[0] / image_size[1]
    return abs(mask_ratio - image_ratio) <= tolerance * image_ratio


def conform_mask_to_image(
    image_path: str | Path, image: Image.Image, mask: Image.Image
) -> tuple[Image.Image, bool]:
    """Put ``mask`` into ``image``'s exact pixel geometry; report whether it was rescaled.

    Two corrections, in order. The EXIF transpose is undone, because LabelMe
    annotates the *displayed* image while ``Image.open`` returns the stored pixel
    layout. Then, if the mask was drawn at a different resolution than the image
    the pipeline reads, it is rescaled to match: masks annotated on the original
    photographs are reused against the pre-resized copies the experiments consume,
    which is a uniform downscale and preserves what the mask means.

    Rescaling uses area-averaging and re-thresholds at the midpoint, so a boundary
    pixel takes the value of whichever class covers most of it -- a majority vote
    rather than the arbitrary single sample nearest-neighbour would pick. The
    result is still strictly binary.

    A size mismatch that is *not* a uniform rescale is not silently repaired: a
    differing aspect ratio means the mask belongs to a differently cropped or
    rotated image, and stretching it would mask the wrong pixels.
    """
    mask = align_mask_to_image(image_path, mask)
    if mask.size == image.size:
        return mask, False
    if not is_pure_rescale(mask.size, image.size):
        hint = (
            "the mask is a rotated view of the image; run "
            "scripts/fix_mask_orientation.py to resolve and correct it"
            if is_pure_rescale(mask.size[::-1], image.size)
            else f"{image_path} may not be the image this mask was annotated on"
        )
        raise ValueError(
            f"mask {mask.size} and image {image.size} differ by more than a uniform "
            f"rescale (aspect ratios {mask.size[0] / mask.size[1]:.4f} vs "
            f"{image.size[0] / image.size[1]:.4f}); {hint}"
        )
    resized = mask.convert("L").resize(image.size, Image.BOX)
    return resized.point(lambda v: 255 if v >= 128 else 0), True


# --------------------------------------------------------------------------- #
# Pixel-space edits                                                           #
# --------------------------------------------------------------------------- #
def replace_background(
    image: Image.Image, mask: Image.Image, fill: Sequence[int]
) -> Image.Image:
    """Replace every background pixel (mask == 0) of ``image`` with ``fill``."""
    pixels = np.array(image.convert("RGB"))
    foreground = np.array(mask.convert("L"))
    if pixels.shape[:2] != foreground.shape:
        raise ValueError(
            f"image {pixels.shape[:2]} and mask {foreground.shape} differ in size"
        )
    out = pixels.copy()
    out[foreground == 0] = np.asarray(fill, dtype=np.uint8)
    return Image.fromarray(out)


def low_pass(image: Image.Image, sigma: float) -> Image.Image:
    """Gaussian low-pass filter applied to the RGB channels of ``image``.

    ``truncate=3`` gives a 6*sigma+1 kernel and ``mode="mirror"`` reflects at the
    border without repeating the edge pixel, so this matches OpenCV's
    ``GaussianBlur(ksize=6*sigma+1, BORDER_REFLECT_101)`` while keeping SciPy (an
    existing dependency) as the only requirement.
    """
    from scipy.ndimage import gaussian_filter

    if sigma <= 0:
        return image
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    blurred = gaussian_filter(pixels, sigma=(sigma, sigma, 0), truncate=3.0, mode="mirror")
    return Image.fromarray(np.clip(blurred, 0, 255).astype(np.uint8))


# --------------------------------------------------------------------------- #
# Condition cache                                                             #
# --------------------------------------------------------------------------- #
# An edit takes the source image (and its mask, if any) and returns the edited
# image. ``None`` is passed for the mask when the frame carries no ``mask_path``.
Edit = Callable[[Image.Image, "Image.Image | None"], Image.Image]


class ConditionCache:
    """Preprocessed uint8 crops for every (image, condition) pair.

    Each source image is opened once; every condition's edit is applied in pixel
    space and then run through the deterministic evaluation geometry (resize +
    center crop). Crops are held as ``uint8`` HWC arrays and normalized only when
    a batch is requested, so the cache survives being reused across all
    checkpoints and seeds at roughly a sixth of the float32 footprint.

    ``frame`` needs ``image_id``, ``relative_path``, ``class_index`` and
    ``source_dataset``; ``mask_path`` is required only if an edit uses the mask.
    """

    def __init__(self, cfg: "Config", frame: pd.DataFrame, edits: Mapping[str, Edit]) -> None:
        self.cfg = cfg
        self.conditions: tuple[str, ...] = (ORIGINAL, *edits)
        self.frame = frame.reset_index(drop=True)
        self._crops: dict[tuple[str, str], np.ndarray] = {}
        self._build(edits)

    def _build(self, edits: Mapping[str, Edit]) -> None:
        from src.data.transforms import eval_geometry  # torchvision: imported lazily

        geometry = eval_geometry(self.cfg)
        data_root = Path(self.cfg.paths.data_root)
        uses_mask = "mask_path" in self.frame.columns
        rescaled = 0
        for count, row in enumerate(self.frame.itertuples(index=False), start=1):
            image_path = data_root / row.relative_path
            image = Image.open(image_path).convert("RGB")
            mask = None
            if uses_mask:
                mask, was_rescaled = conform_mask_to_image(
                    image_path, image, Image.open(row.mask_path)
                )
                rescaled += was_rescaled
            self._crops[(row.image_id, ORIGINAL)] = np.asarray(geometry(image), dtype=np.uint8)
            for name, edit in edits.items():
                edited = geometry(edit(image, mask))
                self._crops[(row.image_id, name)] = np.asarray(edited, dtype=np.uint8)
            if count % 200 == 0:
                logger.info("  cached %d/%d images", count, len(self.frame))
        if rescaled:
            logger.info(
                "  %d/%d masks rescaled to the image resolution (annotated at a "
                "different size)", rescaled, len(self.frame),
            )

    def __len__(self) -> int:
        return len(self._crops)

    def megabytes(self) -> float:
        """Resident size of the cached crops, for the run log."""
        return sum(a.nbytes for a in self._crops.values()) / 1e6

    def batches(
        self, condition: str, batch_size: int
    ) -> "Iterator[tuple[torch.Tensor, np.ndarray, np.ndarray]]":
        """Yield ``(images, labels, image_ids)`` for one condition, in frame order.

        ``images`` is normalized exactly as ``build_transforms(cfg, train=False)``
        would produce: ``ToTensor`` (uint8 -> [0, 1], CHW) then ``Normalize``.
        """
        import torch

        mean = torch.tensor(self.cfg.data.mean).view(3, 1, 1)
        std = torch.tensor(self.cfg.data.std).view(3, 1, 1)
        ids = self.frame["image_id"].to_numpy()
        labels = self.frame["class_index"].to_numpy()
        for start in range(0, len(ids), batch_size):
            chunk = ids[start : start + batch_size]
            stack = np.stack([self._crops[(image_id, condition)] for image_id in chunk])
            images = torch.from_numpy(stack).permute(0, 3, 1, 2).float().div_(255.0)
            yield (images - mean) / std, labels[start : start + batch_size], chunk


# --------------------------------------------------------------------------- #
# Edit factories                                                              #
# --------------------------------------------------------------------------- #
def background_edits(fills: Mapping[str, Sequence[int]]) -> dict[str, Edit]:
    """One edit per named background fill, e.g. ``{"grey": (127, 127, 127)}``."""

    def make(fill: Sequence[int]) -> Edit:
        def edit(image: Image.Image, mask: "Image.Image | None") -> Image.Image:
            if mask is None:
                raise ValueError("background replacement requires a foreground mask")
            return replace_background(image, mask, fill)

        return edit

    return {name: make(fill) for name, fill in fills.items()}


def frequency_edits(sigmas: Sequence[float]) -> dict[str, Edit]:
    """One low-pass edit per sigma, named ``low_sigma_<sigma>``."""

    def make(sigma: float) -> Edit:
        def edit(image: Image.Image, mask: "Image.Image | None") -> Image.Image:  # noqa: ARG001
            return low_pass(image, sigma)

        return edit

    return {f"low_sigma_{_sigma_label(s)}": make(s) for s in sigmas}


def sigma_of(condition: str) -> float:
    """Sigma encoded in a condition name; 0 for ``original``."""
    return 0.0 if condition == ORIGINAL else float(condition.rsplit("_", 1)[1])


def _sigma_label(sigma: float) -> str:
    return str(int(sigma)) if float(sigma).is_integer() else str(sigma)
