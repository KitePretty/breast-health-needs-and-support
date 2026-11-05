"""Submission-matched colors and contrast-aware labels for all analysis figures."""
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import to_rgb

INK = "#252A30"
BLUE = "#24688F"
GOLD = "#D18C20"
RED = "#B64B45"
GRID = "#E9ECEF"
CONNECTOR = "#6C747D"
MISSING_FILL = "#F1F2F4"
MISSING_INK = "#525C67"


def luminance(color):
    rgb = np.array(to_rgb(color))
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return float(linear @ [0.2126, 0.7152, 0.0722])


def contrast_ratio(first, second):
    a, b = sorted([luminance(first), luminance(second)])
    return (b + 0.05) / (a + 0.05)


def label_color(background):
    """Choose the higher-contrast black/white label (at least 4.5:1)."""
    return max(["#000000", "#FFFFFF"], key=lambda color: contrast_ratio(color, background))


def heatmap():
    cmap = colormaps["cividis"].copy()
    cmap.set_bad(MISSING_FILL)
    return cmap


def configure(plt):
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                         "svg.fonttype": "none", "pdf.fonttype": 42,
                         "hatch.linewidth": 0.8, "savefig.dpi": 300})
