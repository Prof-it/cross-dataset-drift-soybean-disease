"""CLI: convert LabelMe polygon annotations into the binary masks the experiment reads.

Turns ``<masks-dir>/<dataset>/<image>.json`` (LabelMe's output, saved beside the
image it annotates) into
``<masks-dir>/<dataset>/<class>/<dataset>_<stem>_mask.png``, the layout
:mod:`src.data.interventions` looks masks up in. Foreground (any ``Leaf``
polygon) is 255, everything else 0; overlapping polygons union, matching the
annotation protocol's handling of overlapping leaves.

Every mask is validated before it counts as converted:

- the JSON's shapes are all ``polygon`` and all labelled ``Leaf``;
- the mask is strictly binary and neither 0% nor 100% foreground;
- **the mask lines up with the image as the experiment will use it** -- that is,
  after the EXIF transpose is undone, its size matches the raw pixel layout that
  ``Image.open`` returns. This is the check that matters: LabelMe records
  dimensions from the *displayed* image, so a photo carrying a 90-degree EXIF
  rotation yields a mask whose axes are swapped relative to the stored pixels,
  and a mask that silently fails to align would replace the wrong region.

Failures are reported and skipped rather than written, so a bad annotation cannot
quietly enter the experiment. Re-running is safe: existing masks are rewritten
only if ``--overwrite`` is passed.

Usage
-----
    python scripts/convert_masks.py
    python scripts/convert_masks.py --overwrite --masks-dir /path/to/masks
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from src.config import Paths, load_config
from src.data.interventions import align_mask_to_image, is_pure_rescale, mask_relative_name
from src.data.splits import build_splits, load_split

logger = logging.getLogger(__name__)

FOREGROUND_LABEL = "Leaf"


def _stem_index(cfg, dataset: str, split_seed: int) -> dict[str, pd.Series]:
    """Map an image file stem to its split-CSV row, for one dataset."""
    frame = load_split(cfg, dataset, split_seed)
    frame = frame.assign(stem=[Path(p).stem for p in frame["relative_path"]])
    return {row.stem: row for row in frame.itertuples(index=False)}


def _polygons(payload: dict) -> tuple[list[np.ndarray], list[str]]:
    """Polygon point arrays from a LabelMe payload, plus any validation problems."""
    problems: list[str] = []
    shapes = payload.get("shapes", [])
    if not shapes:
        problems.append("no shapes")
    points: list[np.ndarray] = []
    for i, shape in enumerate(shapes):
        if shape.get("label") != FOREGROUND_LABEL:
            problems.append(
                f"shape {i} labelled {shape.get('label')!r}, expected {FOREGROUND_LABEL!r}"
            )
        if shape.get("shape_type") != "polygon":
            problems.append(f"shape {i} has type {shape.get('shape_type')!r}, expected 'polygon'")
            continue
        polygon = np.asarray(shape["points"], dtype=float)
        if len(polygon) < 3:
            problems.append(f"shape {i} has {len(polygon)} points, need at least 3")
            continue
        points.append(polygon)
    return points, problems


def _rasterize(points: list[np.ndarray], width: int, height: int) -> Image.Image:
    """Union of the polygons as a binary (0/255) mask image."""
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for polygon in points:
        draw.polygon([(float(x), float(y)) for x, y in polygon], fill=255)
    return mask


def _validate(mask: Image.Image, image_path: Path) -> list[str]:
    """Binary-ness, non-trivial coverage, and usability against the source image."""
    problems: list[str] = []
    array = np.asarray(mask)
    if not set(np.unique(array)).issubset({0, 255}):
        problems.append(f"non-binary values {sorted(set(np.unique(array)))[:5]}")
    coverage = float((array == 255).mean())
    if coverage == 0.0:
        problems.append("0% foreground")
    elif coverage == 1.0:
        problems.append("100% foreground")
    # Accept exactly what the experiment accepts: the mask must line up with the
    # image after the EXIF transpose is undone, or differ from it only by a uniform
    # rescale (masks annotated on the originals, reused against pre-resized copies).
    # A differing aspect ratio means it is not this image's mask.
    raw_size = Image.open(image_path).size
    aligned = align_mask_to_image(image_path, mask)
    if aligned.size != raw_size and not is_pure_rescale(aligned.size, raw_size):
        problems.append(
            f"size {aligned.size} after EXIF alignment is neither equal to the image "
            f"{raw_size} nor a uniform rescale of it"
        )
    return problems


def run(cfg, split_seed: int, overwrite: bool) -> pd.DataFrame:
    masks_dir = Path(cfg.paths.masks_dir)
    data_root = Path(cfg.paths.data_root)
    rows: list[dict] = []
    failures: list[str] = []

    for dataset in cfg.source_datasets:
        build_splits(cfg, dataset, split_seed)
        index = _stem_index(cfg, dataset, split_seed)
        payloads = sorted((masks_dir / dataset).glob("*.json"))
        logger.info("%s: %d LabelMe file(s) under %s", dataset, len(payloads), masks_dir / dataset)

        for path in payloads:
            record = index.get(path.stem)
            if record is None:
                failures.append(f"{dataset}/{path.name}: no split-CSV row with stem {path.stem!r}")
                continue
            out = masks_dir / mask_relative_name(dataset, record.class_label, record.relative_path)
            if out.exists() and not overwrite:
                rows.append({"source_dataset": dataset, "image_id": record.image_id,
                             "class_label": record.class_label, "status": "kept"})
                continue

            payload = json.loads(path.read_text())
            points, problems = _polygons(payload)
            if problems:
                failures += [f"{dataset}/{path.name}: {p}" for p in problems]
                continue
            mask = _rasterize(points, int(payload["imageWidth"]), int(payload["imageHeight"]))
            image_path = data_root / record.relative_path
            problems = _validate(mask, image_path)
            if problems:
                failures += [f"{dataset}/{path.name}: {p}" for p in problems]
                continue

            out.parent.mkdir(parents=True, exist_ok=True)
            mask.save(out)
            rows.append({
                "source_dataset": dataset, "image_id": record.image_id,
                "class_label": record.class_label, "status": "written",
                "foreground_pct": round(float((np.asarray(mask) == 255).mean()) * 100, 2),
            })

    summary = pd.DataFrame(rows)
    if not summary.empty:
        logger.info(
            "\nmasks by dataset, class and status:\n%s",
            summary.pivot_table(index=["source_dataset", "class_label"], columns="status",
                                values="image_id", aggfunc="count", fill_value=0).to_string(),
        )
        if "foreground_pct" in summary:
            written = summary.dropna(subset=["foreground_pct"])
            if not written.empty:
                logger.info(
                    "\nforeground coverage of new masks (protocol expects roughly 20-90%%):\n%s",
                    written.groupby(["source_dataset", "class_label"])["foreground_pct"]
                    .agg(["mean", "min", "max"]).round(1).to_string(),
                )
    if failures:
        logger.error("\n%d annotation(s) rejected and NOT written:", len(failures))
        for failure in failures:
            logger.error("  %s", failure)
        logger.error("Fix these in LabelMe and re-run; the experiment skips images without a mask.")
    else:
        logger.info("\nall annotations converted and validated.")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert LabelMe polygons to binary masks.")
    parser.add_argument("--split-seed", type=int, default=None,
                        help="split CSV used for each image's class (default: first configured)")
    parser.add_argument("--overwrite", action="store_true", help="rewrite masks that already exist")
    parser.add_argument("--data-root", default=None, help="override raw-data location")
    parser.add_argument("--masks-dir", default=None, help="override mask/annotation location")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    paths = Paths.default().with_overrides(data_root=args.data_root, masks_dir=args.masks_dir)
    cfg = load_config(paths=paths)
    run(cfg, args.split_seed if args.split_seed is not None else cfg.split_seeds[0], args.overwrite)


if __name__ == "__main__":
    main()
