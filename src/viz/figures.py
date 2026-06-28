"""Paper-standard figures rendered from the evaluation CSVs (thesis style).

Each function reads the tidy tables written by the ``evaluate`` experiment and
saves a styled figure via :mod:`src.viz.style`. Functions return the saved path,
or ``None`` if the required data is absent (so the runner degrades gracefully).
Visual identity matches the thesis: ASDID blue / MH vermillion, Wong colourblind
palette, short DN/RN/ViS/ViB-d/ts labels. Apply ``set_style(cfg)`` once first.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.viz.style import (
    ACCENT,
    ARCH_ORDER,
    ARCH_SHORT,
    CLASS_SHORT,
    DIVERGING_CMAP,
    DS_COLOURS,
    save_figure,
    short_label,
)

if TYPE_CHECKING:
    from src.config import Config

_WITHIN_C = DS_COLOURS["asdid"]
_CROSS_C = DS_COLOURS["mh"]
_RECOVER_C = ACCENT


def _ordered_models(df: pd.DataFrame) -> list[tuple[str, str, str, str]]:
    """(model_id, arch, path, dataset) ordered dataset -> arch -> path, present-only."""
    out = []
    for ds in ("asdid", "mh"):
        for arch in ARCH_ORDER:
            for path in ("direct", "twostage"):
                mid = f"{arch}_{path}_{ds}"
                if (df["model_id"] == mid).any():
                    out.append((mid, arch, path, ds))
    return out


# --------------------------------------------------------------------------- #
# Transfer gap: dumbbell (within vs cross macro F1), ASDID | MH                #
# --------------------------------------------------------------------------- #
def transfer_gap(cfg: "Config", eval_results: pd.DataFrame, out_dir: str | Path) -> Path | None:
    ft = eval_results[eval_results["experiment"] == "finetune"]
    if ft.empty:
        return None
    models = _ordered_models(ft)
    if not models:
        return None

    fig, ax = plt.subplots(figsize=(7.16, 4.2))
    jitter = 0.12
    for i, (mid, _arch, _path, ds) in enumerate(models):
        colour = DS_COLOURS[ds]
        w = ft[(ft["model_id"] == mid) & (ft["direction"] == "within")]["macro_f1"]
        c = ft[(ft["model_id"] == mid) & (ft["direction"] == "cross")]["macro_f1"]
        w_mean, c_mean = w.mean(), c.mean()
        ax.plot([i, i], [w_mean, c_mean], color=colour, lw=1.0, ls="--", alpha=0.55, zorder=1)
        # individual seeds: within slightly left, cross slightly right
        ax.scatter(np.full(len(w), i - jitter), w, s=13, marker="D", color=colour,
                   alpha=0.5, edgecolor="none", zorder=2)
        ax.scatter(np.full(len(c), i + jitter), c, s=13, marker="D", facecolor="white",
                   edgecolor=colour, linewidth=0.6, alpha=0.7, zorder=2)
        # across-seed means
        ax.scatter(i, w_mean, s=48, marker="o", color=colour, edgecolor="black", linewidth=0.5, zorder=4)
        ax.scatter(i, c_mean, s=48, marker="o", facecolor="white", edgecolor=colour, linewidth=1.3, zorder=4)

    n_asdid = sum(1 for _, _, _, ds in models if ds == "asdid")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([short_label(a, p) for _, a, p, _ in models], rotation=45, ha="right")
    ax.set_ylabel("Macro F1")
    ax.set_ylim(-0.02, 1.06)
    if 0 < n_asdid < len(models):
        ax.axvline(n_asdid - 0.5, color="grey", lw=0.5, ls=":", zorder=1)
        ax.text((n_asdid - 1) / 2, 1.04, "ASDID", ha="center", color=DS_COLOURS["asdid"], fontweight="bold")
        ax.text(n_asdid + (len(models) - n_asdid - 1) / 2, 1.04, "MH", ha="center",
                color=DS_COLOURS["mh"], fontweight="bold")
    ax.grid(axis="y", ls=":", alpha=0.4)

    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=DS_COLOURS["asdid"],
               markeredgecolor="black", markeredgewidth=0.5, markersize=8, label="Within (mean)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor="0.3", markeredgewidth=1.3, markersize=8, label="Cross (mean)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="0.4", markersize=5,
               alpha=0.6, label="Individual seed"),
    ]
    ax.legend(handles=legend, loc="lower left", ncol=3, frameon=True)
    fig.tight_layout()
    return save_figure(fig, Path(out_dir) / "transfer_gap", cfg)


# --------------------------------------------------------------------------- #
# Per-class transfer delta: cross - within F1 (diverging), ASDID | MH         #
# --------------------------------------------------------------------------- #
def per_class_f1(cfg: "Config", eval_results: pd.DataFrame, out_dir: str | Path) -> Path | None:
    ft = eval_results[eval_results["experiment"] == "finetune"]
    classes = list(CLASS_SHORT)
    f1_cols = [f"{c}_f1" for c in classes]
    if ft.empty or not all(c in ft.columns for c in f1_cols):
        return None

    # paired per (model, split_seed, seed): delta = cross_f1 - within_f1
    keys = ["model_id", "arch", "path", "train_dataset", "split_seed", "seed"]
    wide = ft.pivot_table(index=keys, columns="direction", values=f1_cols)
    deltas = []
    for col in f1_cols:
        if ("within" in wide[col]) and ("cross" in wide[col]):
            deltas.append((wide[col]["cross"] - wide[col]["within"]).rename(col))
    if not deltas:
        return None
    delta = pd.concat(deltas, axis=1).reset_index()
    agg = delta.groupby("model_id")[f1_cols].agg(["mean", "std"])

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 4.6), sharey=False)
    vmax = float(np.nanmax(np.abs([agg[(c, "mean")].to_numpy() for c in f1_cols]))) or 0.6
    im = None
    for ax, ds, title in zip(axes, ("asdid", "mh"), ("ASDID", "MH")):
        rows = [(short_label(a, p), f"{a}_{p}_{ds}")
                for a in ARCH_ORDER for p in ("direct", "twostage")
                if f"{a}_{p}_{ds}" in agg.index]
        mat = np.array([[agg.loc[mid, (c, "mean")] for c in f1_cols] for _, mid in rows])
        std = np.array([[agg.loc[mid, (c, "std")] for c in f1_cols] for _, mid in rows])
        im = ax.imshow(mat, cmap=DIVERGING_CMAP, vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(f1_cols)))
        ax.set_xticklabels([CLASS_SHORT[c] for c in classes])
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([lbl for lbl, _ in rows])
        ax.set_title(title, color=DS_COLOURS[ds], fontweight="bold")
        for r in range(mat.shape[0]):
            for cc in range(mat.shape[1]):
                ax.text(cc, r, f"{mat[r, cc]:+.2f}\n±{std[r, cc]:.2f}", ha="center", va="center",
                        fontsize=7, color="black")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="Cross − Within F1")
    fig.suptitle("Per-class transfer gap (cross − within macro F1)", y=1.0)
    return save_figure(fig, Path(out_dir) / "per_class_f1", cfg)


# --------------------------------------------------------------------------- #
# Calibration: reliability (within vs cross) + ECE bars                       #
# --------------------------------------------------------------------------- #
def calibration(cfg, eval_results: pd.DataFrame, reliability: pd.DataFrame | None, out_dir) -> Path | None:
    ft = eval_results[eval_results["experiment"] == "finetune"]
    if ft.empty:
        return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.4))

    if reliability is not None and not reliability.empty:
        rel = reliability
        if "experiment" in rel.columns:
            rel = rel[rel["experiment"] == "finetune"]
        for direction, color in (("within", _WITHIN_C), ("cross", _CROSS_C)):
            sub = rel[rel["direction"] == direction]
            confs, accs = [], []
            for _, rows in sub.groupby("bin"):
                w = rows["bin_count"].sum()
                if w == 0:
                    continue
                confs.append(np.average(rows["bin_confidence"], weights=rows["bin_count"]))
                accs.append(np.average(rows["bin_accuracy"], weights=rows["bin_count"]))
            ax1.plot(confs, accs, marker="o", ms=4, label=direction, color=color)
        ax1.plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
        ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
        ax1.set_xlabel("Mean confidence"); ax1.set_ylabel("Accuracy")
        ax1.set_title("Reliability"); ax1.legend()

    ece = ft.groupby("direction")["ece"].mean()
    cross_temp = ft[ft["direction"] == "cross"]["ece_temp"].mean()
    bars = {"within": ece.get("within", np.nan), "cross": ece.get("cross", np.nan), "cross\n+ temp.": cross_temp}
    ax2.bar(list(bars), list(bars.values()), color=[_WITHIN_C, _CROSS_C, _RECOVER_C], edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("ECE"); ax2.set_title("Calibration error")
    fig.tight_layout()
    return save_figure(fig, Path(out_dir) / "calibration", cfg)


# --------------------------------------------------------------------------- #
# Intervention recovery: cross (un-adapted) vs head-refit                      #
# --------------------------------------------------------------------------- #
def intervention_recovery(cfg, eval_results: pd.DataFrame, out_dir) -> Path | None:
    base = eval_results[(eval_results["experiment"] == "finetune")
                        & (eval_results["path"] == "direct")
                        & (eval_results["direction"] == "cross")]
    probe = eval_results[eval_results["experiment"] == "linear_probe"]
    if base.empty or probe.empty:
        return None
    b = base.groupby(["arch", "train_dataset"])["macro_f1"].agg(["mean", "std"])
    r = probe.groupby(["arch", "train_dataset"])["macro_f1"].agg(["mean", "std"])
    keys = [(a, d) for d in ("asdid", "mh") for a in ARCH_ORDER if (a, d) in b.index and (a, d) in r.index]
    if not keys:
        return None
    labels = [f"{ARCH_SHORT.get(a, a)}\n{d}→" for a, d in keys]
    x = np.arange(len(keys)); width = 0.38
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.bar(x - width / 2, [b.loc[k, "mean"] for k in keys], width, yerr=[b.loc[k, "std"] for k in keys],
           capsize=2, label="cross (un-adapted)", color=_CROSS_C, edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, [r.loc[k, "mean"] for k in keys], width, yerr=[r.loc[k, "std"] for k in keys],
           capsize=2, label="cross + head refit", color=_RECOVER_C, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Macro F1"); ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Refitting the head on the target recovers cross-dataset F1")
    ax.grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    return save_figure(fig, Path(out_dir) / "intervention_recovery", cfg)


# --------------------------------------------------------------------------- #
# Decomposition: prevalence vs conditional split of the cross error gap        #
# --------------------------------------------------------------------------- #
def decomposition(cfg, decomposition_df: pd.DataFrame | None, out_dir) -> Path | None:
    if decomposition_df is None or decomposition_df.empty:
        return None
    df = decomposition_df
    if "experiment" in df.columns:
        df = df[df["experiment"] == "finetune"]
    if df.empty:
        return None
    df = df.assign(model_id=df["arch"] + "_" + df["path"] + "_" + df["train_dataset"])
    agg = df.groupby("model_id")[["prevalence_term", "conditional_term", "observed_gap"]].mean()
    present = [(mid, a, p) for mid, a, p, _ds in _ordered_models(df) if mid in agg.index]
    agg = agg.loc[[mid for mid, _, _ in present]]
    labels = [short_label(a, p) for _, a, p in present]
    x = np.arange(len(agg))
    fig, ax = plt.subplots(figsize=(7.16, 3.8))
    ax.bar(x, agg["prevalence_term"], label="prevalence", color=DS_COLOURS["asdid"], edgecolor="black", linewidth=0.5)
    ax.bar(x, agg["conditional_term"], bottom=agg["prevalence_term"], label="conditional",
           color=DS_COLOURS["mh"], edgecolor="black", linewidth=0.5)
    ax.scatter(x, agg["observed_gap"], color="black", marker="_", s=180, label="observed gap", zorder=5)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Cross-dataset error gap"); ax.legend()
    ax.set_title("Prevalence vs conditional decomposition")
    ax.grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    return save_figure(fig, Path(out_dir) / "decomposition", cfg)
