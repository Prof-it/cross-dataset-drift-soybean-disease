"""Bootstrap confidence intervals and seed aggregation.

Quantifies uncertainty on the reported metrics and on paired differences between
conditions, using bias-corrected and accelerated (BCa) intervals with a
percentile fallback for degenerate cases (e.g. zero jackknife variance). Also
aggregates per-seed / per-split metric rows into mean / std / n for the variance
bands (feedback #5/#10).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import bootstrap as _scipy_bootstrap

if TYPE_CHECKING:
    import pandas as pd


def _interval(data: tuple, statistic: Callable, n_iterations: int, ci: float, paired: bool, seed: int):
    """Run scipy bootstrap with BCa, falling back to percentile on failure."""
    kwargs = dict(
        statistic=statistic,
        n_resamples=n_iterations,
        confidence_level=ci,
        paired=paired,
        random_state=np.random.default_rng(seed),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        try:
            result = _scipy_bootstrap(data, method="BCa", **kwargs)
        except (RuntimeWarning, ValueError):
            result = _scipy_bootstrap(data, method="percentile", **kwargs)
    return float(result.confidence_interval.low), float(result.confidence_interval.high)


def bootstrap_ci(
    values: np.ndarray,
    statistic: Callable = np.mean,
    n_iterations: int = 10_000,
    ci: float = 0.95,
    seed: int = 73,
) -> dict:
    """BCa confidence interval for a one-sample statistic (default the mean).

    ``statistic`` must accept an ``axis`` argument (as ``np.mean`` / ``np.median`` do).
    """
    x = np.asarray(values, dtype=float)
    lo, hi = _interval((x,), statistic, n_iterations, ci, paired=False, seed=seed)
    return {"point": float(statistic(x)), "ci_lower": lo, "ci_upper": hi}


def paired_diff_ci(
    correct_a: np.ndarray,
    correct_b: np.ndarray,
    n_iterations: int = 10_000,
    ci: float = 0.95,
    seed: int = 73,
) -> dict:
    """BCa CI on the paired mean difference (A - B), e.g. per-image correctness."""
    a = np.asarray(correct_a, dtype=float)
    b = np.asarray(correct_b, dtype=float)
    if len(a) != len(b):
        raise ValueError(f"Arrays must have equal length, got {len(a)} and {len(b)}.")

    def diff(x: np.ndarray, y: np.ndarray, axis: int) -> np.ndarray:
        return np.mean(x, axis=axis) - np.mean(y, axis=axis)

    lo, hi = _interval((a, b), diff, n_iterations, ci, paired=True, seed=seed)
    return {"diff": float(a.mean() - b.mean()), "ci_lower": lo, "ci_upper": hi}


def aggregate_over_seeds(
    df: "pd.DataFrame",
    group_cols: list[str],
    metric_cols: list[str],
) -> "pd.DataFrame":
    """Aggregate per-seed/per-split rows into ``{metric}_mean/_std/_n`` columns.

    Standard deviation uses ``ddof=1``; groups with a single run report ``std = NaN``.
    """
    missing = (set(group_cols) | set(metric_cols)) - set(df.columns)
    if missing:
        raise KeyError(f"columns missing from dataframe: {sorted(missing)}")
    grouped = df.groupby(group_cols, as_index=False).agg({c: ["mean", "std", "count"] for c in metric_cols})
    grouped.columns = [
        col[0] if col[1] == "" else f"{col[0]}_{col[1].replace('count', 'n')}"
        for col in grouped.columns.to_flat_index()
    ]
    return grouped


def aggregate_with_ci(
    df: "pd.DataFrame",
    group_cols: list[str],
    metric_cols: list[str],
    *,
    n_iterations: int = 10_000,
    ci: float = 0.95,
    seed: int = 73,
    min_runs: int = 3,
) -> "pd.DataFrame":
    """:func:`aggregate_over_seeds` plus a BCa interval per group and metric.

    Adds ``{metric}_ci_low`` / ``{metric}_ci_high`` beside the ``_mean`` / ``_std``
    / ``_n`` columns. Resampling is over the *rows* of each group, i.e. over runs,
    so the interval expresses variation across initialization seeds and data
    partitions rather than sampling error inside one test split.

    Groups with fewer than ``min_runs`` rows, or with no variance at all, get NaN
    bounds instead of a fabricated interval. BCa estimates its bias-correction and
    acceleration by jackknife, and neither is meaningful on a couple of points or
    on a constant sample -- reporting a number there would overstate what the data
    supports. ``seed`` fixes the resampling so the interval is reproducible.
    """
    import pandas as pd

    base = aggregate_over_seeds(df, group_cols, metric_cols)
    rows: list[dict] = []
    for key, group in df.groupby(group_cols, sort=False):
        keys = key if isinstance(key, tuple) else (key,)
        record = dict(zip(group_cols, keys, strict=True))
        for metric in metric_cols:
            values = group[metric].dropna().to_numpy(dtype=float)
            low = high = float("nan")
            if len(values) >= min_runs and np.ptp(values) > 0:
                interval = bootstrap_ci(values, n_iterations=n_iterations, ci=ci, seed=seed)
                low, high = interval["ci_lower"], interval["ci_upper"]
            record[f"{metric}_ci_low"] = low
            record[f"{metric}_ci_high"] = high
        rows.append(record)
    return base.merge(pd.DataFrame(rows), on=group_cols, how="left")
