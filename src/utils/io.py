"""Filesystem I/O helpers for checkpoints, logs, and results.

One place for the small, repeated I/O operations, including the checkpoint-exists
guard that makes training idempotent on Colab.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (and parents) if missing; return it as a ``Path``."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(obj: Any, path: str | Path) -> Path:
    """Write ``obj`` as indented JSON, creating parent directories as needed."""
    out = Path(path)
    ensure_dir(out.parent)
    out.write_text(json.dumps(obj, indent=2))
    return out


def load_json(path: str | Path) -> Any:
    """Read JSON from ``path``."""
    return json.loads(Path(path).read_text())


def checkpoint_exists(save_dir: str | Path) -> bool:
    """Whether a finished training run already lives in ``save_dir``."""
    directory = Path(save_dir)
    return (directory / "best_model.pt").exists() and (directory / "training_log.json").exists()
