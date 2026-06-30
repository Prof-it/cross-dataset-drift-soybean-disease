"""Nonparametric significance tests for the paper's comparison claims (Lu feedback).

Uncertainty on the primary metrics is summarised with BCa intervals
(see :mod:`src.evaluation.bootstrap`). This module adds formal tests for the
secondary comparisons the paper makes -- the source-dataset asymmetry,
architecture family (CNN vs ViT), pretraining path (direct vs two-stage), and the
training-time / post-hoc interventions -- so statements of "larger", "smaller" or
"similar" can be backed by a p-value rather than read descriptively.

With only nine runs per configuration we use rank-based tests: Mann-Whitney U for
independent groups and Wilcoxon signed-rank for paired comparisons, each reported
with a rank-based effect size. :func:`pairwise_tests` assembles the standard set of
comparisons and, for the paired ones, reports them both pooled (``scope = all``)
and per transfer direction (``scope = asdid`` / ``mh``), because an intervention can
help one direction and not the other. It degrades gracefully -- any comparison whose
rows are absent is skipped -- mirroring :mod:`src.viz.figures`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

_FT = "finetune"
_GAP_KEYS = ["arch", "path", "train_dataset", "split_seed", "seed"]
_PAIR_KEYS = ["arch", "path", "train_dataset", "split_seed", "seed"]
# (label, train_dataset filter) -- pooled plus one per transfer direction.
_SCOPES = (("all", None), ("asdid", "asdid"), ("mh", "mh"))


# --------------------------------------------------------------------------- #
# Core tests                                                                  #
# --------------------------------------------------------------------------- #
def mann_whitney(a, b, *, alternative: str = "two-sided") -> dict:
    """Mann-Whitney U test for two independent samples, with rank-biserial effect size."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        raise ValueError("both samples must be non-empty")
    res = mannwhitneyu(a, b, alternative=alternative)
    rank_biserial = 1.0 - 2.0 * res.statistic / (len(a) * len(b))
    return {
        "test": "mann_whitney_u",
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
        "effect_size": float(rank_biserial),
        "n_a": len(a),
        "n_b": len(b),
        "median_a": float(np.median(a)),
        "median_b": float(np.median(b)),
    }


def wilcoxon_signed_rank(a, b) -> dict:
    """Wilcoxon signed-rank test on paired samples (A vs B), with rank-biserial effect size.

    Returns a non-significant result for identical samples (all paired
    differences zero), where the test is undefined.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError(f"paired samples must match in length, got {len(a)} and {len(b)}")
    diff = a - b
    n_nonzero = int(np.count_nonzero(diff))
    if n_nonzero == 0:
        return {
            "test": "wilcoxon_signed_rank", "statistic": 0.0, "p_value": 1.0,
            "effect_size": 0.0, "n_pairs": len(a), "median_diff": 0.0,
            "median_a": float(np.median(a)), "median_b": float(np.median(b)),
        }
    res = wilcoxon(a, b)
    total = n_nonzero * (n_nonzero + 1) / 2.0
    rank_biserial = 2.0 * res.statistic / total - 1.0
    return {
        "test": "wilcoxon_signed_rank",
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
        "effect_size": float(rank_biserial),
        "n_pairs": len(a),
        "median_diff": float(np.median(diff)),
        "median_a": float(np.median(a)),
        "median_b": float(np.median(b)),
    }


# --------------------------------------------------------------------------- #
# Driver: standard comparisons from an eval_results frame                     #
# --------------------------------------------------------------------------- #
def _family(arch: str) -> str:
    return "vit" if str(arch).lower().startswith("vit") else "cnn"


def _finetune_gap(eval_results: pd.DataFrame) -> pd.DataFrame | None:
    """Per-run transfer gap (within - cross macro F1) for the baseline finetune models."""
    if not ({"experiment", "direction", "macro_f1", *_GAP_KEYS} <= set(eval_results.columns)):
        return None
    ft = eval_results[eval_results["experiment"] == _FT]
    if ft.empty:
        return None
    piv = ft.pivot_table(index=_GAP_KEYS, columns="direction", values="macro_f1")
    if not {"within", "cross"} <= set(piv.columns):
        return None
    piv = piv.dropna(subset=["within", "cross"]).reset_index()
    piv["gap"] = piv["within"] - piv["cross"]
    return piv


def _paired_metric(df, exp_a, exp_b, metric, direction, train_dataset=None):
    """Matched (a, b) arrays of ``metric`` for two experiments, paired per config-run.

    ``train_dataset`` restricts to one transfer direction (None = both, pooled).
    """
    if not ({"experiment", "direction", metric, *_PAIR_KEYS} <= set(df.columns)):
        return None
    sub = df[df["direction"] == direction]
    if train_dataset is not None:
        sub = sub[sub["train_dataset"] == train_dataset]
    a = sub[sub["experiment"] == exp_a][_PAIR_KEYS + [metric]]
    b = sub[sub["experiment"] == exp_b][_PAIR_KEYS + [metric]]
    if a.empty or b.empty:
        return None
    merged = a.merge(b, on=_PAIR_KEYS, suffixes=("_a", "_b"))
    if merged.empty:
        return None
    return merged[f"{metric}_a"].to_numpy(), merged[f"{metric}_b"].to_numpy()


def _intervention_tests(df: pd.DataFrame, scopes=_SCOPES) -> list[dict]:
    out: list[dict] = []
    specs = [
        ("weighting: weighted vs unweighted (cross macro F1)", _FT, "finetune_unweighted", "macro_f1", "cross"),
        ("label smoothing vs baseline (cross ECE)", "finetune_label_smoothing", _FT, "ece", "cross"),
    ]
    for label, exp_a, exp_b, metric, direction in specs:
        for scope_label, ds in scopes:
            pair = _paired_metric(df, exp_a, exp_b, metric, direction, ds)
            if pair is not None:
                out.append({
                    "comparison": label, "scope": scope_label, "metric": f"{direction} {metric}",
                    "group_a": exp_a, "group_b": exp_b, **wilcoxon_signed_rank(pair[0], pair[1]),
                })
    # Temperature scaling: paired raw vs scaled cross ECE within the baseline.
    if {"experiment", "direction", "ece", "ece_temp"} <= set(df.columns):
        for scope_label, ds in scopes:
            sub = df[(df["experiment"] == _FT) & (df["direction"] == "cross")]
            if ds is not None:
                sub = sub[sub["train_dataset"] == ds]
            sub = sub.dropna(subset=["ece", "ece_temp"])
            if not sub.empty:
                out.append({
                    "comparison": "temperature scaling: ECE vs scaled ECE (cross)", "scope": scope_label,
                    "metric": "cross ECE", "group_a": "ece", "group_b": "ece_temp",
                    **wilcoxon_signed_rank(sub["ece"].to_numpy(), sub["ece_temp"].to_numpy()),
                })
    return out


def pairwise_tests(eval_results: pd.DataFrame, *, interventions: pd.DataFrame | None = None,
                   scopes=_SCOPES) -> pd.DataFrame:
    """Assemble the standard comparison tests; skip any whose data is absent.

    Returns a tidy frame with one row per comparison and ``scope`` (``all`` pooled,
    ``asdid`` / ``mh`` per direction for the paired comparisons). The source-dataset
    asymmetry is inherently across-direction, so it is reported only at ``scope=all``.
    """
    if interventions is None:
        interventions = eval_results
    out: list[dict] = []

    gap = _finetune_gap(eval_results)
    if gap is not None:
        # 1. Source-dataset asymmetry (independent, across direction): ASDID gap vs MH gap.
        a = gap.loc[gap["train_dataset"] == "asdid", "gap"].to_numpy()
        b = gap.loc[gap["train_dataset"] == "mh", "gap"].to_numpy()
        if len(a) and len(b):
            out.append({"comparison": "source: ASDID vs MH", "scope": "all", "metric": "transfer gap",
                        "group_a": "asdid", "group_b": "mh", **mann_whitney(a, b)})

        # 2. Architecture family (independent): CNN gap vs ViT gap, pooled and per direction.
        fam = gap.assign(family=gap["arch"].map(_family))
        for scope_label, ds in scopes:
            sub = fam if ds is None else fam[fam["train_dataset"] == ds]
            a = sub.loc[sub["family"] == "cnn", "gap"].to_numpy()
            b = sub.loc[sub["family"] == "vit", "gap"].to_numpy()
            if len(a) and len(b):
                out.append({"comparison": "architecture: CNN vs ViT", "scope": scope_label, "metric": "transfer gap",
                            "group_a": "cnn", "group_b": "vit", **mann_whitney(a, b)})

        # 3. Pretraining path (paired): direct vs two-stage, pooled and per direction.
        keys = ["arch", "train_dataset", "split_seed", "seed"]
        for scope_label, ds in scopes:
            g = gap if ds is None else gap[gap["train_dataset"] == ds]
            d = g[g["path"] == "direct"][keys + ["gap"]]
            t = g[g["path"] == "twostage"][keys + ["gap"]]
            paired = d.merge(t, on=keys, suffixes=("_direct", "_twostage"))
            if not paired.empty:
                out.append({"comparison": "path: direct vs two-stage", "scope": scope_label, "metric": "transfer gap",
                            "group_a": "direct", "group_b": "twostage",
                            **wilcoxon_signed_rank(paired["gap_direct"], paired["gap_twostage"])})

    # 4. Training-time / post-hoc interventions (paired vs their baseline), pooled and per direction.
    out += _intervention_tests(interventions, scopes)
    return pd.DataFrame(out)
