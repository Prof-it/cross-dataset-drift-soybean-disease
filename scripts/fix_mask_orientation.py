"""CLI: resolve and correct masks that are rotated relative to their image.

Masks annotated on the original photographs carry the orientation LabelMe
*displayed*, which for a photo stored with an EXIF rotation is not the stored
pixel layout. Re-encoding those photographs into the pre-resized copies the
pipeline reads drops the EXIF tag, so at load time there is no longer any record
of which way to turn the mask back -- and for a portrait mask over a landscape
image, two different rotations fit the dimensions equally well. Picking one
blindly has a 50% chance of masking the wrong half of the leaf while looking
entirely plausible.

This resolves the ambiguity rather than guessing, in two passes.

**Review (default).** For every mask that does not fit its image, each rotation
that *would* fit is scored by how strongly it separates plant from non-plant,
using the excess-green index ``2G - R - B``: a correct leaf mask covers green
tissue and excludes soil, so the right rotation shows a much larger inside-minus-
outside contrast than a wrong one. The best-scoring rotation is written to a
proposals CSV together with every candidate's score and the margin between the
top two, and overlays are rendered so the proposal can be confirmed by eye. A
per-dataset contact sheet covers the confident cases at a glance; anything whose
margin falls below ``--margin`` also gets its own side-by-side sheet of all
candidates, because those are the ones worth looking at individually.

**Apply (``--apply``).** Reads the proposals CSV back -- including any correction
made by hand in the ``chosen`` column -- and rewrites those mask PNGs rotated into
the image's frame. Only the rotation is applied; resolution is left alone, since
the experiment rescales at load time. After this the masks fit their images and
nothing downstream needs to know any of this happened.

Usage
-----
    python scripts/fix_mask_orientation.py                 # review: scores + overlays
    python scripts/fix_mask_orientation.py --apply         # rewrite the masks
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from src.config import Paths, load_config
from src.data.interventions import (
    align_mask_to_image,
    is_pure_rescale,
    mask_conforms,
    mask_table,
)
from src.data.splits import build_splits

logger = logging.getLogger(__name__)

# Plain rotations. ``identity`` is included so an already-fitting mask stays
# representable in the CSV; ``rotate_90`` / ``rotate_270`` are the two that map a
# portrait mask onto a landscape image, and are what an EXIF 6 or 8 photograph
# needs undone.
ROTATIONS: dict[str, "int | None"] = {
    "identity": None,
    "rotate_90": Image.ROTATE_90,
    "rotate_180": Image.ROTATE_180,
    "rotate_270": Image.ROTATE_270,
}
# Mirrored orientations (EXIF 2/4/5/7). A camera writing a flipped image is rare,
# and including these doubles the candidates -- each plain rotation gains a mirror
# twin that scores identically on any symmetric mask, turning clear cases into
# spurious ties. Opt in with --include-mirrored if a source really has them.
MIRRORED: dict[str, int] = {
    "transpose": Image.TRANSPOSE,
    "transverse": Image.TRANSVERSE,
}
CANDIDATES: dict[str, "int | None"] = {**ROTATIONS, **MIRRORED}
REVIEW_DIRNAME = "_orientation_review"
PROPOSALS_NAME = "orientation_proposals.csv"


def _excess_green(image: Image.Image) -> np.ndarray:
    """Excess-green index 2G - R - B: high on foliage, low on soil, litter and sky."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    return 2.0 * rgb[:, :, 1] - rgb[:, :, 0] - rgb[:, :, 2]


def _as_image_frame(mask: Image.Image, transpose, size: tuple[int, int]) -> Image.Image:
    """Apply ``transpose`` and rescale to ``size``, keeping the mask binary."""
    turned = mask if transpose is None else mask.transpose(transpose)
    resized = turned.convert("L").resize(size, Image.BOX)
    return resized.point(lambda v: 255 if v >= 128 else 0)


def _score(image: Image.Image, mask: Image.Image) -> float:
    """Inside-minus-outside excess green. Higher means a better plant/background split."""
    greenness = _excess_green(image)
    foreground = np.asarray(mask) == 255
    if not foreground.any() or foreground.all():
        return float("-inf")
    return float(greenness[foreground].mean() - greenness[~foreground].mean())


def _candidates_for(
    image: Image.Image, mask: Image.Image, allowed: dict[str, "int | None"]
) -> list[tuple[str, float]]:
    """Score every rotation whose result fits the image, best first."""
    scored: list[tuple[str, float]] = []
    for name, transpose in allowed.items():
        turned = mask if transpose is None else mask.transpose(transpose)
        if turned.size != image.size and not is_pure_rescale(turned.size, image.size):
            continue
        scored.append((name, _score(image, _as_image_frame(mask, transpose, image.size))))
    return sorted(scored, key=lambda pair: pair[1], reverse=True)


def _overlay(image: Image.Image, mask: Image.Image, width: int = 300) -> Image.Image:
    """The image with the mask's foreground tinted, downscaled for review."""
    base = image.convert("RGB")
    tint = Image.new("RGB", base.size, (255, 0, 255))
    composed = Image.composite(Image.blend(base, tint, 0.45), base, mask.convert("L"))
    height = round(base.size[1] * width / base.size[0])
    return composed.resize((width, height), Image.BILINEAR)


def _label(canvas: Image.Image, xy: tuple[int, int], text: str) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([xy, (xy[0] + 300, xy[1] + 13)], fill=(0, 0, 0))
    draw.text((xy[0] + 3, xy[1] + 2), text[:56], fill=(255, 255, 255))


def _grid(cells: list[tuple[Image.Image, str]], columns: int) -> Image.Image:
    cell_w = max(c.size[0] for c, _ in cells)
    cell_h = max(c.size[1] for c, _ in cells) + 15
    rows = (len(cells) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * cell_w, rows * cell_h), (30, 30, 30))
    for i, (cell, text) in enumerate(cells):
        x, y = (i % columns) * cell_w, (i // columns) * cell_h
        canvas.paste(cell, (x, y))
        _label(canvas, (x, y + cell.size[1]), text)
    return canvas


def review(cfg, split_seed: int, margin: float, allowed: dict) -> pd.DataFrame:
    """Score every non-fitting mask, render overlays, and write the proposals CSV."""
    masks_dir = Path(cfg.paths.masks_dir)
    data_root = Path(cfg.paths.data_root)
    review_dir = masks_dir / REVIEW_DIRNAME
    rows: list[dict] = []

    for dataset in cfg.source_datasets:
        build_splits(cfg, dataset, split_seed)
        frame = mask_table(cfg, dataset, split_seed)
        bad = [
            row for row in frame.itertuples(index=False)
            if mask_conforms(data_root / row.relative_path, row.mask_path) is not None
        ]
        logger.info("%s: %d of %d masks do not fit their image", dataset, len(bad), len(frame))
        if not bad:
            continue

        confident: list[tuple[Image.Image, str]] = []
        out_dir = review_dir / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        for row in bad:
            image_path = data_root / row.relative_path
            image = Image.open(image_path).convert("RGB")
            mask = align_mask_to_image(image_path, Image.open(row.mask_path))
            scored = _candidates_for(image, mask, allowed)
            if not scored:
                logger.warning("  %s: no rotation fits; leaving alone", row.image_id)
                continue
            best, best_score = scored[0]
            gap = best_score - (scored[1][1] if len(scored) > 1 else 0.0)
            rows.append({
                "source_dataset": dataset, "image_id": row.image_id,
                "class_label": row.class_label, "chosen": best,
                "margin": round(gap, 3),
                **{f"score_{n}": round(s, 3) for n, s in scored},
            })
            stem = row.image_id.replace("/", "__")
            if gap < margin:
                # Ambiguous: show every candidate side by side for a human call.
                cells = [(_overlay(image, _as_image_frame(mask, allowed[n], image.size)),
                          f"{n}  score={s:.1f}") for n, s in scored]
                _grid(cells, len(cells)).save(out_dir / f"AMBIGUOUS_{stem}.png")
            else:
                confident.append((
                    _overlay(image, _as_image_frame(mask, allowed[best], image.size)),
                    f"{stem}  {best}  m={gap:.1f}",
                ))
        if confident:
            _grid(confident, 4).save(review_dir / f"{dataset}_proposed.png")
            logger.info("  contact sheet: %s", review_dir / f"{dataset}_proposed.png")

    proposals = pd.DataFrame(rows)
    if proposals.empty:
        logger.info("nothing to fix: every mask already fits its image")
        return proposals
    path = masks_dir / PROPOSALS_NAME
    proposals.to_csv(path, index=False)
    logger.info(
        "\nwrote %s (%d masks)\n%s", path, len(proposals),
        proposals.groupby(["source_dataset", "chosen"]).size().to_string(),
    )
    ambiguous = int((proposals["margin"] < margin).sum())
    logger.info(
        "\nReview %s, then re-run with --apply.\n"
        "  - %s_proposed.png: the confident proposals, one cell per image. The tinted "
        "region should sit on the leaf.\n"
        "  - AMBIGUOUS_*.png (%d): every candidate side by side; set the 'chosen' "
        "column in the CSV by hand for these.\n"
        "Correct any cell that looks wrong by editing 'chosen' before applying.",
        review_dir, "/".join(cfg.source_datasets), ambiguous,
    )
    return proposals


def apply(cfg, split_seed: int) -> None:
    """Rewrite the masks named in the proposals CSV, rotated into the image's frame."""
    masks_dir = Path(cfg.paths.masks_dir)
    data_root = Path(cfg.paths.data_root)
    path = masks_dir / PROPOSALS_NAME
    if not path.exists():
        raise SystemExit(f"no {path}; run without --apply first, then review it")
    proposals = pd.read_csv(path).set_index(["source_dataset", "image_id"])

    fixed = failed = 0
    for dataset in cfg.source_datasets:
        build_splits(cfg, dataset, split_seed)
        for row in mask_table(cfg, dataset, split_seed).itertuples(index=False):
            key = (dataset, row.image_id)
            if key not in proposals.index:
                continue
            chosen = str(proposals.loc[key, "chosen"])
            if chosen not in CANDIDATES:
                raise SystemExit(
                    f"{key}: 'chosen' is {chosen!r}, expected one of {list(CANDIDATES)}"
                )
            image_path = data_root / row.relative_path
            mask = align_mask_to_image(image_path, Image.open(row.mask_path))
            transpose = CANDIDATES[chosen]
            turned = mask if transpose is None else mask.transpose(transpose)
            # Rotation only: the experiment rescales at load time, so keeping the
            # annotated resolution preserves boundary detail.
            turned.convert("L").point(lambda v: 255 if v >= 128 else 0).save(row.mask_path)
            if mask_conforms(image_path, row.mask_path) is None:
                fixed += 1
            else:
                failed += 1
                logger.error("  %s still does not fit after '%s'", row.image_id, chosen)
    logger.info("rewrote %d mask(s); %d still not fitting", fixed, failed)
    if failed:
        raise SystemExit("some masks still do not fit; check their 'chosen' value")


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve rotated masks against their images.")
    parser.add_argument("--apply", action="store_true",
                        help="rewrite the masks using the reviewed proposals CSV")
    parser.add_argument("--include-mirrored", action="store_true",
                        help="also consider mirrored orientations (EXIF 2/4/5/7); rare")
    parser.add_argument("--margin", type=float, default=5.0,
                        help="excess-green margin below which a case needs manual review")
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--masks-dir", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    paths = Paths.default().with_overrides(data_root=args.data_root, masks_dir=args.masks_dir)
    cfg = load_config(paths=paths)
    seed = args.split_seed if args.split_seed is not None else cfg.split_seeds[0]
    allowed = CANDIDATES if args.include_mirrored else ROTATIONS
    apply(cfg, seed) if args.apply else review(cfg, seed, args.margin, allowed)


if __name__ == "__main__":
    main()
