"""CLI: test whether the prevalence/conditional decomposition is robust to ordering.

Reconstructs per-class error rates (1 - recall) and prevalences (support share)
from ``eval_results.csv`` for the baseline finetune within/cross pairs, then
decomposes the gap under the forward, reverse, and symmetric (Shapley) orderings.
Writes ``decomposition_robustness.csv`` and reports, per source block, whether the
conditional term dominates the prevalence term under every ordering. Validates the
reconstruction against the stored forward decomposition in ``decomposition.csv``.

Run locally (``python scripts/compute_decomposition_robustness.py``) or, on Colab,
call ``run(cfg)`` with the resolved config.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from src.config import Paths, load_config
from src.evaluation.decomposition import decompose_error_orderings

CLASSES = ("healthy", "rust", "frogeye_leaf_spot")
KEYS = ["arch", "path", "train_dataset", "seed", "split_seed"]


def _inputs(within: pd.Series, cross: pd.Series):
    e_s = np.array([1.0 - within[f"{c}_recall"] for c in CLASSES], dtype=float)
    e_t = np.array([1.0 - cross[f"{c}_recall"] for c in CLASSES], dtype=float)
    sup_s = np.array([within[f"{c}_support"] for c in CLASSES], dtype=float)
    sup_t = np.array([cross[f"{c}_support"] for c in CLASSES], dtype=float)
    return e_s, e_t, sup_s / sup_s.sum(), sup_t / sup_t.sum()


def run(cfg) -> None:
    results = cfg.paths.results_dir
    eval_csv = results / "eval_results.csv"
    if not eval_csv.exists():
        raise SystemExit(f"missing {eval_csv}; run the 'evaluate' experiment first")

    df = pd.read_csv(eval_csv)
    ft = df[df["experiment"] == "finetune"]
    rows = []
    for key, g in ft.groupby(KEYS):
        w = g[g["direction"] == "within"]
        c = g[g["direction"] == "cross"]
        if len(w) != 1 or len(c) != 1:
            continue
        e_s, e_t, p_s, p_t = _inputs(w.iloc[0], c.iloc[0])
        rows.append({**dict(zip(KEYS, key)), **decompose_error_orderings(e_s, e_t, p_s, p_t)})
    out = pd.DataFrame(rows)
    if out.empty:
        logging.info("no finetune within/cross pairs found in %s", eval_csv)
        return

    out.to_csv(results / "decomposition_robustness.csv", index=False)
    logging.info("wrote %s (%d model-runs)", results / "decomposition_robustness.csv", len(out))

    stored = results / "decomposition.csv"
    if stored.exists():
        s = pd.read_csv(stored)
        s = s[s["experiment"] == "finetune"] if "experiment" in s.columns else s
        m = out.merge(s, on=KEYS, suffixes=("", "_stored"))
        if not m.empty:
            dp = (m["prevalence_forward"] - m["prevalence_term"]).abs().max()
            dc = (m["conditional_forward"] - m["conditional_term"]).abs().max()
            logging.info("reconstruction check vs decomposition.csv: max |Δprev|=%.2e, max |Δcond|=%.2e", dp, dc)

    logging.info("\nblock means (prevalence | conditional) by ordering:")
    for block, sub in list(out.groupby("train_dataset")) + [("ALL", out)]:
        parts = [f"{block:>6}"]
        dominates = True
        for which in ("forward", "reverse", "symmetric"):
            pv = sub[f"prevalence_{which}"].mean()
            cd = sub[f"conditional_{which}"].mean()
            parts.append(f"{which[:3]}: {pv:+.3f} | {cd:+.3f}")
            dominates = dominates and (abs(cd) > abs(pv))
        parts.append("=> conditional dominates" if dominates else "=> NOT dominant in all orderings")
        logging.info("  " + "   ".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser(description="Ordering-robustness of the error decomposition.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    paths = Paths.default().with_overrides(results_dir=args.results_dir)
    cfg = load_config(args.config, paths=paths) if args.config else load_config(paths=paths)
    run(cfg)


if __name__ == "__main__":
    main()
