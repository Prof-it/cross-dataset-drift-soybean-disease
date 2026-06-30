"""Prevalence-vs-conditional decomposition of the cross-dataset error.

Operationalizes the distinction between label (prevalence) shift and conditional
(concept) shift. Given per-class error rates and the source/target class
prevalences, the change in error from within-source to cross-target evaluation
splits into two additive components:

- a **prevalence** term: the error change explained purely by the target's
  different class balance, holding per-class (conditional) errors at source level; and
- a **conditional** term: the error change explained by per-class errors changing
  across datasets, evaluated at target prevalence.

Their sum equals the observed cross-dataset error gap exactly, so the
decomposition is reported alongside the headline numbers.
"""

from __future__ import annotations

import numpy as np


def decompose_error(
    err_source: np.ndarray,
    err_target: np.ndarray,
    prevalence_source: np.ndarray,
    prevalence_target: np.ndarray,
) -> dict:
    """Split the within-source -> cross-target error gap into its two components.

    Parameters
    ----------
    err_source, err_target:
        Per-class error rates (length K) within source and across to target.
    prevalence_source, prevalence_target:
        Per-class prevalences (length K) of the source and target test sets.

    Returns
    -------
    dict with ``prevalence_term``, ``conditional_term``, ``total`` (their sum), and
    ``observed_gap`` (computed directly; equals ``total`` up to floating point).
    """
    e_s = np.asarray(err_source, dtype=float)
    e_t = np.asarray(err_target, dtype=float)
    p_s = np.asarray(prevalence_source, dtype=float)
    p_t = np.asarray(prevalence_target, dtype=float)
    if not (len(e_s) == len(e_t) == len(p_s) == len(p_t)):
        raise ValueError("All inputs must have the same length (number of classes).")

    prevalence_term = float(np.sum((p_t - p_s) * e_s))
    conditional_term = float(np.sum(p_t * (e_t - e_s)))
    observed_gap = float(np.sum(p_t * e_t) - np.sum(p_s * e_s))
    return {
        "prevalence_term": prevalence_term,
        "conditional_term": conditional_term,
        "total": prevalence_term + conditional_term,
        "observed_gap": observed_gap,
    }


def decompose_error_orderings(
    err_source: np.ndarray,
    err_target: np.ndarray,
    prevalence_source: np.ndarray,
    prevalence_target: np.ndarray,
) -> dict:
    """Decompose the gap under all three weightings, to test ordering sensitivity.

    The two-term split is path-dependent. :func:`decompose_error` uses the
    *forward* ordering, which weights the prevalence term by the within-source
    per-class errors. Because those errors are small, that term comes out near
    zero. A reviewer could instead use the *reverse* ordering (prevalence weighted
    by the larger cross-target errors), where the prevalence term need not be
    negligible. We therefore also report the *symmetric* (Shapley) ordering, the
    average of the two. Each prevalence/conditional pair sums to the observed gap.
    """
    e_s = np.asarray(err_source, dtype=float)
    e_t = np.asarray(err_target, dtype=float)
    p_s = np.asarray(prevalence_source, dtype=float)
    p_t = np.asarray(prevalence_target, dtype=float)
    if not (len(e_s) == len(e_t) == len(p_s) == len(p_t)):
        raise ValueError("All inputs must have the same length (number of classes).")

    dp = p_t - p_s
    de = e_t - e_s
    out = {
        "prevalence_forward": float(np.sum(dp * e_s)),
        "conditional_forward": float(np.sum(p_t * de)),
        "prevalence_reverse": float(np.sum(dp * e_t)),
        "conditional_reverse": float(np.sum(p_s * de)),
        "prevalence_symmetric": float(np.sum(dp * (e_s + e_t) / 2.0)),
        "conditional_symmetric": float(np.sum((p_s + p_t) / 2.0 * de)),
        "observed_gap": float(np.sum(p_t * e_t) - np.sum(p_s * e_s)),
    }
    return out
