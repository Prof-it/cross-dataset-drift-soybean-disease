"""Tests: figure rendering (needs matplotlib)."""

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("matplotlib")

from src.config import load_config
from src.viz import figures
from src.viz.style import set_style


def _finetune_rows():
    rows = []
    for direction, eval_ds, f1 in (("within", "asdid", 0.95), ("cross", "mh", 0.6)):
        for seed in (7, 21):
            rows.append({
                "experiment": "finetune", "arch": "resnet50", "path": "direct",
                "train_dataset": "asdid", "eval_dataset": eval_ds, "direction": direction,
                "seed": seed, "split_seed": 73, "model_id": "resnet50_direct_asdid",
                "accuracy": f1, "macro_f1": f1, "ece": 0.1, "ece_temp": 0.08, "temperature": 1.2,
            })
    return pd.DataFrame(rows)


def test_transfer_gap_renders(tmp_path):
    cfg = load_config()
    set_style(cfg)
    out = figures.transfer_gap(cfg, _finetune_rows(), tmp_path)
    assert out is not None and out.exists() and out.suffix == f".{cfg.figures.format}"


def test_missing_data_returns_none(tmp_path):
    cfg = load_config()
    set_style(cfg)
    empty = pd.DataFrame(columns=["experiment"])
    assert figures.transfer_gap(cfg, empty, tmp_path) is None
