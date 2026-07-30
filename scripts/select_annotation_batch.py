"""CLI: choose which images to annotate for the background intervention.

The background intervention scores models on hand-annotated images, and an image
is only usable for the *within-dataset* arm if it was held out of that run's
training data. Because the split is seeded, "held out" is a property of a
(image, split seed) pair, not of an image: an image in the test split under one
seed is in the training split under another. Annotating a fixed set that is held
out under every seed is not viable -- the three-way intersection is a handful of
images and leaves some classes almost empty -- so the intervention runs at a
single split seed (see ``configs/experiments/background_intervention.yaml``) and
this script picks the images to annotate for that seed.

What it does:

1. reads the split CSV for ``--split-seed`` and keeps the held-out images;
2. counts how many already have a mask, per dataset and class;
3. samples the shortfall up to ``--per-class``, preferring images that are *also*
   held out under the other split seeds, so the annotation carries over if the
   analysis is ever repeated at another seed;
4. copies the chosen images to ``<masks-dir>/<dataset>/`` -- the same folder the
   existing LabelMe JSONs live in, so annotating in place puts each ``.json``
   beside its image, where ``scripts/convert_masks.py`` expects it;
5. writes ``<masks-dir>/annotation_manifest.csv`` recording the batch.

Nothing is overwritten: images that already have a mask are never re-listed, and
re-running after a partial annotation session tops the batch back up.

Usage
-----
    python scripts/select_annotation_batch.py --per-class 25
    python scripts/select_annotation_batch.py --per-class 15 --split-seed 73 --dry-run
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Paths, load_config
from src.data.interventions import mask_relative_name
from src.data.splits import build_splits, load_split

logger = logging.getLogger(__name__)

HELD_OUT_SPLITS = ("val", "test")


def _eligible(cfg, dataset: str, split_seed: int, splits: tuple[str, ...]) -> pd.DataFrame:
    """Held-out rows for ``dataset``, with mask status and cross-seed reusability."""
    frame = load_split(cfg, dataset, split_seed)
    frame = frame[frame["split"].isin(splits)].copy()

    masks_dir = Path(cfg.paths.masks_dir)
    frame["mask_path"] = [
        masks_dir / mask_relative_name(dataset, row.class_label, row.relative_path)
        for row in frame.itertuples(index=False)
    ]
    frame["annotated"] = [path.exists() for path in frame["mask_path"]]

    # How many *other* split seeds also hold this image out. Ties in the sample are
    # broken toward images that would remain usable at another seed, which costs
    # nothing now and preserves the option later.
    others = [s for s in cfg.split_seeds if s != split_seed]
    reuse = np.zeros(len(frame), dtype=int)
    for other in others:
        membership = load_split(cfg, dataset, other).set_index("image_id")["split"]
        reuse += frame["image_id"].map(membership).isin(splits).to_numpy().astype(int)
    frame["reusable_seeds"] = reuse
    return frame


def _sample(frame: pd.DataFrame, cfg, per_class: int, split_seed: int) -> pd.DataFrame:
    """Sample the shortfall per class, preferring the most reusable candidates."""
    rng = np.random.default_rng(split_seed)
    chosen: list[pd.DataFrame] = []
    for class_label in cfg.data.class_names:
        in_class = frame[frame["class_label"] == class_label]
        have = int(in_class["annotated"].sum())
        deficit = max(0, per_class - have)
        pool = in_class[~in_class["annotated"]]
        if deficit == 0:
            continue
        if len(pool) < deficit:
            logger.warning(
                "  %-18s only %d un-annotated candidates for a deficit of %d",
                class_label, len(pool), deficit,
            )
            deficit = len(pool)
        # Shuffle first so the ordering within a reusability tier is random, then
        # take the most-reusable images.
        pool = pool.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))
        ranked = pool.sort_values("reusable_seeds", ascending=False, kind="stable")
        chosen.append(ranked.head(deficit))
    return pd.concat(chosen) if chosen else frame.iloc[:0]


def _report(dataset: str, frame: pd.DataFrame, batch: pd.DataFrame, per_class: int) -> None:
    rows = []
    for class_label, group in frame.groupby("class_label", sort=True):
        have = int(group["annotated"].sum())
        new = int((batch["class_label"] == class_label).sum())
        rows.append({
            "class": class_label, "eligible": len(group), "annotated": have,
            "to_annotate": new, "after": have + new, "target": per_class,
        })
    logger.info("%s (split-seed held-out pool):\n%s", dataset,
                pd.DataFrame(rows).to_string(index=False))


def run(
    cfg, split_seed: int, per_class: int, splits: tuple[str, ...], dry_run: bool
) -> pd.DataFrame:
    masks_dir = Path(cfg.paths.masks_dir)
    batches: list[pd.DataFrame] = []
    for dataset in cfg.source_datasets:
        build_splits(cfg, dataset, split_seed)
        for other in cfg.split_seeds:
            build_splits(cfg, dataset, other)
        frame = _eligible(cfg, dataset, split_seed, splits)
        batch = _sample(frame, cfg, per_class, split_seed).assign(source_dataset=dataset)
        _report(dataset, frame, batch, per_class)

        if not dry_run and not batch.empty:
            destination = masks_dir / dataset
            destination.mkdir(parents=True, exist_ok=True)
            for row in batch.itertuples(index=False):
                source = Path(cfg.paths.data_root) / row.relative_path
                target = destination / source.name
                if not target.exists():
                    shutil.copy2(source, target)
        batches.append(batch)

    manifest = pd.concat(batches, ignore_index=True) if batches else pd.DataFrame()
    if manifest.empty:
        logger.info("nothing to annotate: every class already meets the target")
        return manifest

    manifest = manifest.assign(
        split_seed=split_seed,
        image_file=[Path(p).name for p in manifest["relative_path"]],
        mask_path=[str(p) for p in manifest["mask_path"]],
    )
    columns = [
        "source_dataset", "image_id", "class_label", "class_index", "split",
        "split_seed", "reusable_seeds", "relative_path", "image_file", "mask_path",
    ]
    if not dry_run:
        path = masks_dir / "annotation_manifest.csv"
        manifest[columns].to_csv(path, index=False)
        logger.info("wrote %s (%d images)", path, len(manifest))
        logger.info(
            "\nNext:\n"
            "  1. Open each dataset folder in LabelMe and annotate the copied images:\n"
            "       labelme %s\n"
            "       labelme %s\n"
            "     One polygon class, label 'Leaf'; rules in %s.\n"
            "     Save each .json beside its image (LabelMe's default).\n"
            "  2. Convert and validate:\n"
            "       python scripts/convert_masks.py\n"
            "  3. Re-run the intervention:\n"
            "       python scripts/run_experiment.py "
            "configs/experiments/background_intervention.yaml",
            masks_dir / cfg.source_datasets[0], masks_dir / cfg.source_datasets[-1],
            masks_dir / "ANNOTATION_PROTOCOL.md",
        )
    else:
        logger.info("dry run: no images copied, no manifest written")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select images to annotate for the background intervention."
    )
    parser.add_argument("--per-class", type=int, default=25,
                        help="target annotated images per dataset and class (default: 25)")
    parser.add_argument("--split-seed", type=int, default=None,
                        help="partition the intervention runs at (default: first configured)")
    parser.add_argument("--include-val", action="store_true", default=True,
                        help="count val as held out alongside test (default: on)")
    parser.add_argument("--test-only", dest="include_val", action="store_false",
                        help="restrict to the test split (stricter, smaller pool)")
    parser.add_argument("--data-root", default=None, help="override raw-data location")
    parser.add_argument("--masks-dir", default=None, help="override mask/annotation location")
    parser.add_argument("--dry-run", action="store_true",
                        help="report the batch without copying anything")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    paths = Paths.default().with_overrides(data_root=args.data_root, masks_dir=args.masks_dir)
    cfg = load_config(paths=paths)
    split_seed = args.split_seed if args.split_seed is not None else cfg.split_seeds[0]
    splits = HELD_OUT_SPLITS if args.include_val else ("test",)
    logger.info("split seed %s | held-out splits %s | target %d per class\n",
                split_seed, splits, args.per_class)
    run(cfg, split_seed, args.per_class, splits, args.dry_run)


if __name__ == "__main__":
    main()
