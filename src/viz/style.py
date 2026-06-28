"""Figure styling shared by both papers — "muted editorial" identity.

One place that defines how every figure looks. A calm, desaturated palette
(slate blue for ASDID, warm clay for MH, sage as the positive/recovery accent),
colourblind-aware, with light gridlines and thin spines. Driven by the
``figures`` config block for dpi / format / font size.

To retune the identity, edit the colour constants below (or override them live in
results/visualisations.ipynb); ``src.viz.figures`` and ``scripts/make_figures``
both read from here, so changes propagate everywhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

if TYPE_CHECKING:
    from src.config import Config

# --- Muted editorial palette ----------------------------------------------- #
SLATE = "#4C6E91"   # slate blue
CLAY = "#C0694B"    # warm clay / terracotta
SAGE = "#6E9B7C"    # sage green
OCHRE = "#C9A24B"   # muted gold
PLUM = "#8C6F9B"    # dusty plum
GREY = "#8A8F98"    # neutral
INK = "#2B2B2B"     # text
CREAM = "#F2EDE4"   # neutral midpoint for the diverging map

DS_COLOURS: dict[str, str] = {"asdid": SLATE, "mh": CLAY}            # dataset identity
CLASS_COLOURS: dict[str, str] = {                                   # per-class
    "healthy": SAGE, "rust": OCHRE, "frogeye_leaf_spot": PLUM,
}
COND_COLOURS: dict[str, str] = {                                    # calibration interventions
    "baseline": SLATE, "label_smoothing": SAGE,
    "temperature": OCHRE, "ls_temperature": CLAY,
}
ACCENT = SAGE   # positive / "recovery" accent (e.g. head-refit, +temperature)

# Muted diverging map for the per-class delta heatmap: clay (drop) -> cream -> slate (gain).
DIVERGING_CMAP = LinearSegmentedColormap.from_list("editorial_div", [CLAY, CREAM, SLATE])

# --- Short labels (compact axis ticks) ------------------------------------- #
ARCH_SHORT: dict[str, str] = {
    "densenet201": "DN", "resnet50": "RN", "vit_small": "ViS", "vit_base": "ViB",
}
PATH_SHORT: dict[str, str] = {"direct": "d", "twostage": "ts"}
ARCH_ORDER = ("densenet201", "resnet50", "vit_small", "vit_base")
CLASS_SHORT: dict[str, str] = {"healthy": "Healthy", "rust": "Rust", "frogeye_leaf_spot": "Frogeye"}


def short_label(arch: str, path: str) -> str:
    """``densenet201, direct`` -> ``DN-d``."""
    return f"{ARCH_SHORT.get(arch, arch)}-{PATH_SHORT.get(path, path)}"


# IEEE-friendly print sizes (inches): double-column ~7.16in, single ~3.5in.
FIG_FULL = (7.0, 3.6)
FIG_FULL_WIDE = (7.16, 4.2)
FIG_WIDE = (7.16, 3.2)
FIG_SQUARE = (5.5, 5.0)
FIG_HALF = (3.5, 3.0)


def set_style(cfg: "Config") -> None:
    """Apply the shared muted-editorial matplotlib style from ``cfg.figures``."""
    f = cfg.figures
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": f.dpi,
        "savefig.format": f.format,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": f.font_size,
        "axes.titlesize": f.font_size + 1,
        "axes.titleweight": "semibold",
        "axes.labelsize": f.font_size,
        "xtick.labelsize": f.font_size - 1,
        "ytick.labelsize": f.font_size - 1,
        "legend.fontsize": f.font_size - 1,
        "legend.frameon": False,
        "lines.linewidth": 1.4,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#5A5A5A",
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": "#5A5A5A",
        "ytick.color": "#5A5A5A",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "axes.grid": False,
        "grid.color": "#D7D7D7",
        "grid.linestyle": ":",
        "grid.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def class_palette(cfg: "Config") -> dict[str, str]:
    """The fixed per-class colour mapping (config overrides the defaults)."""
    return {**CLASS_COLOURS, **dict(cfg.figures.palette)}


def color_for(cfg: "Config", class_name: str) -> str:
    """Colour for a single class."""
    return class_palette(cfg).get(class_name, GREY)


def save_figure(fig: "plt.Figure", path: str | Path, cfg: "Config") -> Path:
    """Save ``fig`` at the configured format/dpi and close it; return the path."""
    out = Path(path).with_suffix(f".{cfg.figures.format}")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out
