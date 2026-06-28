"""Colab setup helpers.

The workarounds that make the training runs reliable on Colab, as reusable
functions so the driver notebook is a few readable calls. Everything is
off-Colab-safe to import; the Colab-only steps (mount, clone) are just not called
when running locally.

Public API:
- ``in_colab()`` / ``load_secrets()`` -- runtime detection and ``.env`` secrets.
- ``mount_drive()`` -- mount Google Drive, return the ``MyDrive`` root.
- ``sparse_clone()`` -- shallow + sparse + blob-filtered clone to local SSD.
- ``install_package()`` -- editable install of the repo and add it to ``sys.path``.
- ``cache_datasets()`` -- copy + unzip dataset archives from Drive to local SSD.
- ``pre_resize_images()`` -- one-time downscale of oversized images (marker-guarded).
- ``resolve_paths()`` -- a ``Paths`` with data on local SSD, artifacts on Drive.
- ``bundle_artifacts()`` -- zip checkpoints / results / logs back to Drive.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config import Paths

logger = logging.getLogger(__name__)

# Minimal tree to check out on Colab (no raw data, no checkpoints).
DEFAULT_SPARSE_PATHS = (
    "src", "configs", "scripts",
    "pyproject.toml", "requirements.txt", "README.md", "CLAUDE.md", ".gitignore",
)
# Dataset archive base names expected under <DRIVE_ROOT>/data/raw/.
DATASET_ARCHIVES = ("ASDID", "MH-SoyaHealthVision", "PlantVillage")
_SECRET_KEYS = ("GITHUB_PAT", "GITHUB_REPO_URL", "GIT_BRANCH", "DRIVE_ROOT")


def in_colab() -> bool:
    """Whether we are running inside a Google Colab runtime."""
    return "COLAB_RELEASE_TAG" in os.environ


def load_secrets(env_path: str | Path = ".env") -> dict[str, str]:
    """Read secrets from a ``.env`` file, overlaid by environment / Colab userdata.

    Recognized keys: GITHUB_PAT, GITHUB_REPO_URL, GIT_BRANCH, DRIVE_ROOT. Values
    in the environment (or Colab ``userdata``) take precedence over the file.
    """
    secrets: dict[str, str] = {}
    path = Path(env_path)
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            secrets[key.strip()] = value.strip()
    try:  # Colab Secrets, if configured
        from google.colab import userdata  # type: ignore[import-not-found]

        for key in _SECRET_KEYS:
            try:
                secrets[key] = userdata.get(key) or secrets.get(key, "")
            except Exception:  # key not set in userdata
                pass
    except ImportError:
        pass
    for key in _SECRET_KEYS:
        if os.environ.get(key):
            secrets[key] = os.environ[key]
    return {k: v for k, v in secrets.items() if v}


def mount_drive(mount_point: str = "/content/drive") -> Path:
    """Mount Google Drive and return the ``MyDrive`` root."""
    from google.colab import drive  # type: ignore[import-not-found]

    drive.mount(mount_point)
    return Path(mount_point) / "MyDrive"


def sparse_clone(
    repo_url: str,
    branch: str,
    dest: str | Path,
    token: str | None = None,
    sparse_paths: tuple[str, ...] = DEFAULT_SPARSE_PATHS,
) -> Path:
    """Shallow + sparse + blob-filtered clone of the code to local SSD.

    Re-cloning each session is fast and avoids carrying git history / data on
    Drive. ``token`` (a read-only PAT) is injected into the HTTPS URL if given.
    """
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    url = repo_url.replace("https://", f"https://{token}@") if token else repo_url
    subprocess.run(
        ["git", "clone", "--depth=1", "--filter=blob:none", "--sparse", "--branch", branch, url, str(dest)],
        check=True,
    )
    subprocess.run(["git", "-C", str(dest), "sparse-checkout", "init", "--no-cone"], check=True)
    subprocess.run(["git", "-C", str(dest), "sparse-checkout", "set", *sparse_paths], check=True)
    logger.info("cloned %s@%s -> %s", repo_url, branch, dest)
    return dest


def install_package(repo: str | Path) -> None:
    """Editable-install the repo (pulls dependencies) and add it to ``sys.path``."""
    repo = Path(repo)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(repo)], check=True)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))


def cache_datasets(
    drive_data_raw: str | Path,
    local_root: str | Path,
    archives: tuple[str, ...] = DATASET_ARCHIVES,
) -> Path:
    """Copy + unzip dataset archives from Drive to local SSD once.

    Expects ``<drive_data_raw>/<name>.zip`` for each name; extracts into
    ``<local_root>/data/raw/``. Returns the local raw-data directory.
    """
    drive_data_raw = Path(drive_data_raw)
    local_raw = Path(local_root) / "data" / "raw"
    # Cache is complete only when every archive's top-level folder is present, so a
    # partial extraction is re-done rather than silently reused.
    if local_raw.exists() and all((local_raw / name).exists() for name in archives):
        logger.info("dataset cache present at %s", local_raw)
        return local_raw
    local_raw.mkdir(parents=True, exist_ok=True)
    for name in archives:
        archive = drive_data_raw / f"{name}.zip"
        if not archive.exists():
            raise FileNotFoundError(f"missing dataset archive: {archive}")
        # Python zipfile extracts straight from Drive and raises BadZipFile on a
        # corrupt/truncated archive (clearer than the unzip binary's exit codes).
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(local_raw)
        logger.info("cached %s", name)
    return local_raw


def pre_resize_images(root: str | Path, short_side: int = 512, marker: str = ".pre_resized") -> None:
    """Downscale oversized images to ``short_side`` once (guarded by a marker file).

    ASDID images are ~20 MP but training uses 224x224, so a one-time resize makes
    every epoch far faster. Idempotent: the marker file short-circuits reruns.
    """
    from PIL import Image

    root = Path(root)
    flag = root / marker
    if flag.exists():
        logger.info("images already pre-resized")
        return
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    resized = 0
    for path in root.rglob("*"):
        if path.suffix.lower() not in exts:
            continue
        image = Image.open(path)
        width, height = image.size
        if min(width, height) <= short_side:
            continue
        scale = short_side / min(width, height)
        image = image.resize((int(width * scale), int(height * scale)), Image.LANCZOS)
        image.save(path, quality=95)
        resized += 1
    flag.touch()
    logger.info("pre-resized %d images", resized)


def resolve_paths(local_root: str | Path, drive_root: str | Path) -> "Paths":
    """Build a ``Paths`` with data on local SSD and artifacts persisted on Drive."""
    from src.config import Paths

    local_root = Path(local_root)
    drive_root = Path(drive_root)
    return Paths(
        data_root=local_root / "data" / "raw",
        splits_dir=local_root / "data" / "splits",
        class_weights_path=local_root / "data" / "class_weights.json",
        checkpoints_dir=drive_root / "checkpoints",
        results_dir=drive_root / "results",
        logs_dir=drive_root / "logs",
    )


def bundle_artifacts(paths: "Paths", drive_dir: str | Path, name: str = "artifacts") -> Path:
    """Zip checkpoints / results / logs to a single archive on Drive."""
    drive_dir = Path(drive_dir)
    staging = drive_dir / name
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for source, label in (
        (paths.checkpoints_dir, "checkpoints"),
        (paths.results_dir, "results"),
        (paths.logs_dir, "logs"),
    ):
        if Path(source).exists():
            shutil.copytree(source, staging / label)
    archive = shutil.make_archive(str(drive_dir / name), "zip", root_dir=str(drive_dir), base_dir=name)
    shutil.rmtree(staging)
    logger.info("bundled artifacts -> %s", archive)
    return Path(archive)
