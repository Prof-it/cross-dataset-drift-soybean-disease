"""Tests: nonparametric significance tests for the comparison claims (Lu feedback)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from src.evaluation.significance import mann_whitney, pairwise_tests, wilcoxon_signed_rank


def test_mann_whitney_matches_scipy_and_effect_bounds():
    rng = np.random.default_rng(0)
    a = rng.normal(0.30, 0.02, 9)
    b = rng.normal(0.06, 0.02, 9)
    res = mann_whitney(a, b)
    assert res["p_value"] == mannwhitneyu(a, b, alternative="two-sided").pvalue
    assert -1.0 <= res["effect_size"] <= 1.0
    assert res["n_a"] == 9 and res["n_b"] == 9
    assert res["median_a"] > res["median_b"]


def test_wilcoxon_identical_is_nonsignificant():
    x = np.array([0.30, 0.31, 0.29, 0.32])
    res = wilcoxon_signed_rank(x, x)
    assert res["p_value"] == 1.0
    assert res["median_diff"] == 0.0
    assert res["n_pairs"] == 4


def test_pairwise_tests_builds_expected_comparisons():
    out = pairwise_tests(_synthetic_eval())
    comps = set(out["comparison"])
    assert any("source" in c for c in comps)
    assert any("architecture" in c for c in comps)
    assert any("path" in c for c in comps)
    assert "scope" in out.columns
    assert {"all", "asdid", "mh"}.issubset(set(out["scope"]))
    src = out[out["comparison"].str.startswith("source")].iloc[0]
    assert src["p_value"] < 0.05           # ASDID vs MH gap is separable here
    assert src["median_a"] > src["median_b"]


def _synthetic_eval() -> pd.DataFrame:
    """Minimal eval_results-shaped frame with a clear source asymmetry."""
    rng = np.random.default_rng(1)
    rows = []
    for arch in ("densenet201", "resnet50", "vit_small", "vit_base"):
        for path in ("direct", "twostage"):
            for ds, gap_mean in (("asdid", 0.32), ("mh", 0.06)):
                within = 0.98 if ds == "asdid" else 0.81
                for split_seed in (1, 2, 3):
                    for seed in (7, 21, 73):
                        gap = gap_mean + rng.normal(0, 0.01)
                        for direction, val in (("within", within), ("cross", within - gap)):
                            rows.append({
                                "experiment": "finetune", "arch": arch, "path": path,
                                "train_dataset": ds, "direction": direction,
                                "split_seed": split_seed, "seed": seed,
                                "macro_f1": val, "ece": 0.10, "ece_temp": 0.10,
                            })
    return pd.DataFrame(rows)
