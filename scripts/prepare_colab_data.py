"""Build slim, pre-resized dataset zips for Colab (run once, locally).

Colab's `/content` is wiped between sessions, so the notebook copies + unzips the
dataset archives and pre-resizes the images every session. ASDID's raw images are
~20 MP, which makes that slow and burns compute units. This script does the
expensive work once, offline, producing archives that contain:

- only the classes the experiments use (ASDID/MH shared classes; all of PlantVillage),
- images already downscaled to ``--short-side`` (so the Colab pre-resize is a no-op),
- a ``.pre_resized`` marker so the pre-resize step skips entirely.

The archive names and internal paths match what ``src.colab.cache_datasets``
expects, so the notebook is unchanged. Upload the results to your Drive at
``<DRIVE_ROOT>/data/raw/``.

Usage
-----
    python scripts/prepare_colab_data.py --out colab_data
    # then upload colab_data/*.zip to Drive at <DRIVE_ROOT>/data/raw/
"""

from __future__ import annotations

import argparse
import io
import logging
import zipfile
from pathlib import Path

from PIL import Image

from src.config import load_config

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def _image_bytes(path: Path, short_side: int) -> bytes:
    """Bytes for ``path``, downscaled to ``short_side`` if larger (else unchanged)."""
    image = Image.open(path)
    fmt = image.format or "JPEG"
    width, height = image.size
    if min(width, height) <= short_side:
        return path.read_bytes()
    scale = short_side / min(width, height)
    image = image.convert("RGB").resize((round(width * scale), round(height * scale)), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **({"quality": 95} if fmt.upper() in ("JPEG", "JPG") else {}))
    return buffer.getvalue()


def _source_dirs(cfg, dataset: str) -> list[Path]:
    registry = cfg.data.datasets[dataset]
    base = cfg.paths.data_root / registry["folder"]
    subfolders = registry["subfolders"]
    return [base] if subfolders is None else [base / sub for sub in subfolders]


def build_archive(cfg, dataset: str, out_dir: Path, short_side: int) -> tuple[Path, int]:
    """Zip ``dataset``'s used images (resized) under their data_root-relative paths."""
    data_root = cfg.paths.data_root
    archive = out_dir / f"{cfg.data.datasets[dataset]['folder'].split('/')[0]}.zip"
    count = 0
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr(".pre_resized", "")  # extraction drops the marker into data/raw/
        for source in _source_dirs(cfg, dataset):
            for path in sorted(source.rglob("*")):
                if path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                zf.writestr(path.relative_to(data_root).as_posix(), _image_bytes(path, short_side))
                count += 1
    return archive, count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build slim pre-resized dataset zips for Colab.")
    parser.add_argument("--out", default="colab_data", help="output directory for the zips")
    parser.add_argument("--short-side", type=int, default=512, help="target short side in pixels")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="datasets to pack (default: source datasets + plantvillage)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = load_config()
    datasets = args.datasets or [*cfg.source_datasets, "plantvillage"]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        archive, count = build_archive(cfg, dataset, out_dir, args.short_side)
        logging.info("%-12s -> %s  (%d images, %.0f MB)",
                     dataset, archive, count, archive.stat().st_size / 1e6)
    logging.info("Upload %s/*.zip to your Drive at <DRIVE_ROOT>/data/raw/", out_dir)


if __name__ == "__main__":
    main()
