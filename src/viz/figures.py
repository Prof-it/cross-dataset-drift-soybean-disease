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
    GREY,
    save_figure,
    short_label,
)

if TYPE_CHECKING:
    from src.config import Config

_WITHIN_C = DS_COLOURS["asdid"]
_CROSS_C = DS_COLOURS["mh"]
_RECOVER_C = ACCENT

# Bright dataset palette for the two headline paper figures (the approved look,
# matching the thesis transfer dumbbell). Kept local so the muted-editorial
# identity in style.py still governs the exploratory figures below.
ASDID_C = "#1f77b4"
MH_C = "#e8920c"
_REL_WITHIN = "#1f77b4"   # within-dataset reliability curve
_REL_CROSS = "#d62728"    # cross-dataset reliability curve


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
def transfer_dumbbell(cfg: "Config", eval_results: pd.DataFrame, out_dir: str | Path) -> Path | None:
    ft = eval_results[eval_results["experiment"] == "finetune"]
    if ft.empty:
        return None
    order = [(a, p) for a in ARCH_ORDER for p in ("direct", "twostage")]
    labels = [short_label(a, p) for a, p in order]

    fig, ax = plt.subplots(figsize=(11, 4.3))

    def _block(src: str, x0: int, colour: str) -> bool:
        drew = False
        for i, (a, p) in enumerate(order):
            sub = ft[ft["model_id"] == f"{a}_{p}_{src}"]
            win = sub[sub["direction"] == "within"]["macro_f1"].to_numpy()
            cro = sub[sub["direction"] == "cross"]["macro_f1"].to_numpy()
            if len(win) == 0 or len(cro) == 0:
                continue
            drew = True
            x = x0 + i
            ax.plot([x, x], [cro.mean(), win.mean()], ls="--", color=colour, lw=1.1, alpha=0.8, zorder=1)
            ax.scatter(np.full(len(win), x), win, marker="D", s=14, color=colour, alpha=0.30, edgecolors="none", zorder=2)
            ax.scatter(np.full(len(cro), x), cro, marker="D", s=14, color=colour, alpha=0.30, edgecolors="none", zorder=2)
            ax.scatter([x], [win.mean()], marker="o", s=70, color=colour, edgecolors="black", lw=0.8, zorder=3)
            ax.scatter([x], [cro.mean()], marker="o", s=70, facecolors="white", edgecolors=colour, lw=1.8, zorder=3)
        return drew

    drew_a = _block("asdid", 0, ASDID_C)
    drew_m = _block("mh", 9, MH_C)
    if not (drew_a or drew_m):
        plt.close(fig)
        return None
    ax.axvline(8, color=GREY, ls=":", lw=1)
    ax.set_xticks(list(range(8)) + list(range(9, 17)))
    ax.set_xticklabels(labels + labels, rotation=45, ha="right")
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 1.03)
    ax.text(3.5, 1.06, "ASDID", color=ASDID_C, fontsize=14, fontweight="bold", ha="center")
    ax.text(12.5, 1.06, "MH", color=MH_C, fontsize=14, fontweight="bold", ha="center")
    ax.grid(axis="y", ls=":", alpha=0.4)

    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="grey",
               markeredgecolor="black", markersize=9, label="Within (mean)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor="grey", markeredgewidth=1.8, markersize=9, label="Cross (mean)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="grey", alpha=0.4,
               markersize=7, label="Individual seed"),
    ]
    ax.legend(handles=legend, loc="lower center", ncol=3, frameon=True, bbox_to_anchor=(0.5, -0.34))
    fig.tight_layout()
    return save_figure(fig, Path(out_dir) / "transfer_dumbbell", cfg)


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
def calibration_by_dataset(cfg, eval_results: pd.DataFrame, reliability: pd.DataFrame | None, out_dir) -> Path | None:
    ft = eval_results[eval_results["experiment"] == "finetune"]
    if ft.empty or reliability is None or reliability.empty:
        return None
    rel = reliability
    if "experiment" in rel.columns:
        rel = rel[rel["experiment"] == "finetune"]
    rel = rel[rel["bin_count"] > 0]

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5), sharey=True)
    for ax, src, title, tcol in zip(axes, ("asdid", "mh"), ("ASDID", "MH"), (ASDID_C, MH_C)):
        ax.plot([0, 1], [0, 1], ls="--", color=GREY, lw=1)
        for direction, col, ls, face, lab in (
            ("within", _REL_WITHIN, "-", _REL_WITHIN, "within"),
            ("cross", _REL_CROSS, "--", "white", "cross"),
        ):
            sub = rel[(rel["train_dataset"] == src) & (rel["direction"] == direction)]
            confs, accs = [], []
            for _, rows in sub.groupby("bin"):
                if rows["bin_count"].sum() == 0:
                    continue
                confs.append(np.average(rows["bin_confidence"], weights=rows["bin_count"]))
                accs.append(np.average(rows["bin_accuracy"], weights=rows["bin_count"]))
            ax.plot(confs, accs, ls=ls, color=col, lw=1.6, zorder=2)
            ax.scatter(confs, accs, s=34, facecolors=face, edgecolors=col, lw=1.5, zorder=3, label=lab)
        w_ece = ft[(ft["train_dataset"] == src) & (ft["direction"] == "within")]["ece"].mean()
        c_ece = ft[(ft["train_dataset"] == src) & (ft["direction"] == "cross")]["ece"].mean()
        ax.text(0.04, 0.93, f"within ECE {w_ece:.3f}\ncross ECE {c_ece:.3f}", fontsize=9.5, va="top",
                bbox=dict(boxstyle="round", fc="white", ec=GREY, alpha=0.9))
        ax.set_title(title, color=tcol, fontweight="bold")
        ax.set_xlabel("Confidence")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(ls=":", alpha=0.35)
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(loc="lower right", frameon=True)
    fig.tight_layout()
    return save_figure(fig, Path(out_dir) / "calibration_by_dataset", cfg)


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
