"""Tests: configuration loading and overrides."""

from __future__ import annotations

from src.config import load_config


def test_defaults():
    cfg = load_config()
    assert cfg.seeds == (73, 7, 21)
    assert cfg.split_seeds == (73, 7, 21)
    assert cfg.plantvillage_split_seed == 73
    assert cfg.seed_design == "cross"
    assert cfg.run_name == "finetune"
    assert cfg.data.num_classes == 3
    assert cfg.data.class_to_idx == {"healthy": 0, "rust": 1, "frogeye_leaf_spot": 2}
    assert set(cfg.architectures) == {"densenet201", "resnet50", "vit_small", "vit_base"}
    assert cfg.training.batch_size == 128
    assert cfg.training.class_weighting is True
    total = cfg.data.train_ratio + cfg.data.val_ratio + cfg.data.test_ratio
    assert abs(total - 1.0) < 1e-9


def test_seed_pairs_diagonal_and_cross():
    from src.experiments._common import seed_pairs

    assert len(seed_pairs(load_config())) == 9  # default is cross
    assert seed_pairs(load_config({"seed_design": "diagonal"})) == [(73, 73), (7, 7), (21, 21)]


def test_dict_override_wins():
    cfg = load_config({"training": {"label_smoothing": 0.1, "class_weighting": False}})
    assert cfg.training.label_smoothing == 0.1
    assert cfg.training.class_weighting is False
    # untouched keys keep their defaults
    assert cfg.training.batch_size == 128


def test_finetune_variants_use_distinct_run_names():
    # The unweighted-MH and label-smoothing variants must not collide with the
    # baseline finetune checkpoint namespace, or they get idempotently skipped.
    from pathlib import Path

    configs = Path(__file__).resolve().parents[1] / "configs" / "experiments"
    baseline = load_config(str(configs / "full_finetune.yaml"))
    unweighted = load_config(str(configs / "unweighted_mh.yaml"))
    smoothing = load_config(str(configs / "label_smoothing.yaml"))
    names = {baseline.run_name, unweighted.run_name, smoothing.run_name}
    assert names == {"finetune", "finetune_unweighted", "finetune_label_smoothing"}
    assert unweighted.training.class_weighting is False
    assert smoothing.training.label_smoothing == 0.1


def test_extras_capture_experiment_keys():
    cfg = load_config({"experiment": "linear_probe", "linear_probe": {"probes": ["head_refit"]}})
    assert cfg.extras["experiment"] == "linear_probe"
    assert cfg.extras["linear_probe"]["probes"] == ["head_refit"]
