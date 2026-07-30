"""Tests for the grouped BCa aggregation used by the paper's headline tables.

The contract that matters: an interval is reported only where the bootstrap can
support one, it brackets the mean, and it is reproducible from the same seed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.bootstrap import aggregate_with_ci


def _frame(values_by_group: dict[str, list[float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"group": g, "metric": v} for g, vals in values_by_group.items() for v in vals]
    )


def test_interval_brackets_the_mean_and_carries_the_count():
    rng = np.random.default_rng(0)
    frame = _frame({"a": list(rng.normal(0.35, 0.02, 72)), "b": list(rng.normal(0.07, 0.02, 72))})
    out = aggregate_with_ci(frame, ["group"], ["metric"], n_iterations=2000, seed=73)

    assert list(out["metric_n"]) == [72, 72]
    for row in out.itertuples(index=False):
        assert row.metric_ci_low < row.metric_mean < row.metric_ci_high


def test_same_seed_reproduces_the_interval():
    rng = np.random.default_rng(1)
    frame = _frame({"a": list(rng.normal(0.3, 0.05, 40))})
    kw = dict(n_iterations=2000, seed=73)
    first = aggregate_with_ci(frame, ["group"], ["metric"], **kw)
    second = aggregate_with_ci(frame, ["group"], ["metric"], **kw)
    assert first["metric_ci_low"].iloc[0] == second["metric_ci_low"].iloc[0]
    assert first["metric_ci_high"].iloc[0] == second["metric_ci_high"].iloc[0]


def test_no_interval_is_invented_for_tiny_or_constant_samples():
    """BCa's jackknife terms are meaningless here, so the bounds must be NaN."""
    frame = _frame({"tiny": [0.3, 0.4], "constant": [0.5] * 9})
    out = aggregate_with_ci(frame, ["group"], ["metric"], n_iterations=500, seed=73)
    assert out["metric_ci_low"].isna().all()
    assert out["metric_ci_high"].isna().all()
    # The point estimates are still reported.
    assert not out["metric_mean"].isna().any()


def test_groups_are_summarized_independently():
    frame = _frame({"a": [0.1, 0.2, 0.3, 0.4, 0.5], "b": [0.8, 0.85, 0.9, 0.95, 1.0]})
    out = aggregate_with_ci(
        frame, ["group"], ["metric"], n_iterations=2000, seed=73
    ).set_index("group")
    assert out.loc["a", "metric_ci_high"] < out.loc["b", "metric_ci_low"], "disjoint groups"
