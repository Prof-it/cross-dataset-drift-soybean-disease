"""Tests: prevalence-vs-conditional decomposition (pure NumPy, feedback #12)."""

from __future__ import annotations

from src.evaluation.decomposition import decompose_error, decompose_error_orderings


def test_components_reconstruct_gap():
    d = decompose_error(
        err_source=[0.02, 0.05, 0.03],
        err_target=[0.10, 0.30, 0.12],
        prevalence_source=[0.34, 0.33, 0.33],
        prevalence_target=[0.15, 0.70, 0.15],
    )
    assert abs(d["total"] - d["observed_gap"]) < 1e-12
    assert d["total"] == d["prevalence_term"] + d["conditional_term"]


def test_no_shift_is_zero_gap():
    err = [0.1, 0.2, 0.3]
    prev = [0.34, 0.33, 0.33]
    d = decompose_error(err, err, prev, prev)
    assert abs(d["observed_gap"]) < 1e-12
    assert abs(d["prevalence_term"]) < 1e-12
    assert abs(d["conditional_term"]) < 1e-12


def test_orderings_each_reconstruct_gap_and_forward_matches():
    args = dict(
        err_source=[0.02, 0.05, 0.03],
        err_target=[0.10, 0.30, 0.12],
        prevalence_source=[0.34, 0.33, 0.33],
        prevalence_target=[0.15, 0.70, 0.15],
    )
    o = decompose_error_orderings(**args)
    for which in ("forward", "reverse", "symmetric"):
        assert abs(o[f"prevalence_{which}"] + o[f"conditional_{which}"] - o["observed_gap"]) < 1e-12
    # symmetric is the average of the two orderings
    assert abs(o["prevalence_symmetric"] - 0.5 * (o["prevalence_forward"] + o["prevalence_reverse"])) < 1e-12
    # forward ordering equals decompose_error
    d = decompose_error(**args)
    assert abs(o["prevalence_forward"] - d["prevalence_term"]) < 1e-12
    assert abs(o["conditional_forward"] - d["conditional_term"]) < 1e-12
