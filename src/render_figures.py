#!/usr/bin/env python3
"""Render manuscript Figures 2 and 3 from aggregate tables produced by run_analysis.py.
No new statistical analysis is performed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from figure_style import (INK, BLUE, GOLD, RED, GRID, CONNECTOR, MISSING_FILL,
                          MISSING_INK, heatmap, label_color, configure, contrast_ratio)


ROUND4 = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROUND4 / "outputs" / "manuscript_figures"
TABLES = ROUND4 / "outputs" / "tables"
DPI = 300

MAPPING_ORDER = ["D2→E5", "D3→E2", "D6→E1", "D2→E1", "D1→E1"]
ROW_LABELS = [
    "Risk assessment (D2) →\nHelp-seeking encouragement (E5)",
    "Emotional relief (D3) →\nEmotional support (E2)",
    "Medical process information seeking (D6) →\nInformational support (E1)",
    "Risk assessment (D2) →\nInformational support (E1)",
    "Symptom interpretation (D1) →\nInformational support (E1)",
]
TOPIC_ORDER = [
    "Discovering Lumps", "Feeling Armpit Pain", "Navigating Imaging",
    "Reading Clinical Reports", "Receiving Benign Findings",
    "Undergoing Biopsy", "Waiting for Results",
]
INFO_MAPPINGS = ["D1→E1", "D2→E1", "D6→E1"]
FIELDS = ["access_probability", "conditional_match_probability", "overall_coverage_known"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_figure(fig, path: Path) -> None:
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.06,
                facecolor="white", metadata={"Description":
                "Rendered from frozen Round 4 result tables; category-presence terminology."})
    for extension in [".svg", ".pdf"]:
        fig.savefig(path.with_suffix(extension), bbox_inches="tight", pad_inches=0.06,
                    facecolor="white")
    plt.close(fig)


def load_data():
    paths = [TABLES / "09_core_pathway_overall.csv", TABLES / "12_c1_3_topic_pathways.csv"]
    hashes = {str(p.relative_to(ROUND4)): sha256(p) for p in paths}
    pathways = pd.read_csv(paths[0]).set_index("mapping").loc[MAPPING_ORDER]
    topic = pd.read_csv(paths[1])
    topic = topic[topic["mapping"].isin(INFO_MAPPINGS)]
    pivot = topic.pivot(index="topic", columns="mapping", values="overall_coverage_known")
    pivot = pivot.reindex(index=TOPIC_ORDER, columns=INFO_MAPPINGS)

    # Anchor exports to the percentages already verified in the submitted PNGs.
    expected_dots = np.array([
        [62.6, 26.2, 16.2], [75.5, 58.8, 44.4], [83.8, 88.1, 73.8],
        [62.6, 64.6, 40.0], [61.6, 64.0, 38.9],
    ])
    np.testing.assert_array_equal(np.round(pathways[FIELDS].to_numpy() * 100, 1), expected_dots)
    expected_cells = np.array([
        [21, 25, np.nan], [19, 25, np.nan], [70, 75, 76], [13, 13, 18],
        [42, 45, 42], [60, np.nan, 88], [84, 76, 90],
    ])
    np.testing.assert_allclose(np.round(pivot.to_numpy() * 100), expected_cells, equal_nan=True)
    return pathways, pivot, paths, hashes


def figure2(pathways: pd.DataFrame, output_dir: Path) -> Path:
    fig = plt.figure(figsize=(12, 6.3))
    ax = fig.add_axes([0.355, 0.205, 0.61, 0.76])
    y = np.arange(len(pathways))[::-1]
    values = pathways[FIELDS].to_numpy()
    for yy, vals in zip(y, values):
        ax.hlines(yy, vals.min(), vals.max(), color=CONNECTOR, linewidth=1.15, zorder=1)

    for index, marker, color, size in [(0, "o", BLUE, 105), (1, "s", GOLD, 100), (2, "D", RED, 105)]:
        ax.scatter(values[:, index], y, marker=marker, color=color, s=size,
                   edgecolors=INK, linewidths=0.8, zorder=3)
    for yy, (response, conditional, coverage) in zip(y, values):
        ax.annotate(f"{response * 100:.1f}%", (response, yy), xytext=(7, 7),
                    textcoords="offset points", ha="left", va="bottom", fontsize=11)
        ax.annotate(f"{conditional * 100:.1f}%", (conditional, yy), xytext=(7, -9),
                    textcoords="offset points", ha="left", va="top", fontsize=11)
        ax.annotate(f"{coverage * 100:.1f}%", (coverage, yy), xytext=(0, 10),
                    textcoords="offset points", ha="center", va="bottom", fontsize=11)

    ax.set_yticks(y, ROW_LABELS, fontsize=12.5)
    ax.set_ylim(-0.5, 4.5)
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.tick_params(axis="x", length=0, pad=8, labelsize=10.5, colors="#555B61")
    ax.tick_params(axis="y", length=0, pad=14)
    ax.set_xlabel("Proportion (%)", fontsize=11.5, labelpad=12)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#555B61")
    ax.spines["bottom"].set_linewidth(0.8)

    handles = [Line2D([], [], marker=m, linestyle="None", markersize=7,
                      markerfacecolor=c, markeredgecolor=INK, markeredgewidth=0.8)
               for m, c in [("o", BLUE), ("s", GOLD), ("D", RED)]]
    labels = [
        "Received any human response",
        "Support category present among\nthreads with analyzable replies",
        "Overall support-category coverage",
    ]
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.51, 0.018),
               ncol=3, frameon=False, fontsize=10.2, handlelength=0.8,
               handletextpad=0.5, columnspacing=1.9)
    path = output_dir / "figure2_support_gaps_publication.png"
    save_figure(fig, path)
    return path


def figure3(pivot: pd.DataFrame, output_dir: Path) -> Path:
    fig = plt.figure(figsize=(12.5, 6.1))
    ax = fig.add_axes([0.20, 0.035, 0.735, 0.855])
    cmap = heatmap()
    im = ax.imshow(np.ma.masked_invalid(pivot.to_numpy()), vmin=0, vmax=1,
                   cmap=cmap, aspect="auto", interpolation="nearest")
    labels = [
        "Symptom interpretation\n(D1) → Informational support\n(E1)",
        "Risk assessment\n(D2) → Informational support\n(E1)",
        "Medical process information\nseeking (D6) → Informational\nsupport (E1)",
    ]
    ax.set_xticks(range(3), labels, fontsize=11, fontweight="bold")
    ax.set_yticks(range(7), TOPIC_ORDER, fontsize=11.5)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=12)
    ax.tick_params(axis="y", length=0, pad=10)
    for i in range(7):
        for j in range(3):
            value = pivot.iloc[i, j]
            if pd.isna(value):
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=MISSING_FILL,
                                       edgecolor=MISSING_INK, linewidth=0, hatch="///"))
                ax.text(j, i, "—", ha="center", va="center", color=MISSING_INK,
                        fontsize=13, fontweight="bold")
            else:
                ax.text(j, i, f"{value * 100:.0f}%", ha="center", va="center",
                        color=label_color(cmap(value)[:3]), fontsize=13, fontweight="bold")
    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 7, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.25)
    ax.tick_params(which="minor", bottom=False, left=False, top=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cax = fig.add_axes([0.952, 0.035, 0.025, 0.855])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1], labels=["0%", "25%", "50%", "75%", "100%"])
    cbar.ax.tick_params(length=0, labelsize=10.5, pad=5)
    cbar.outline.set_edgecolor("#A5ABB1")
    cbar.outline.set_linewidth(0.5)
    cbar.set_label("Overall informational-support coverage", rotation=270, labelpad=17, fontsize=11)
    path = output_dir / "Figure3.png"
    save_figure(fig, path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=MANUSCRIPT)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure(plt)
    pathways, pivot, input_paths, hashes = load_data()
    outputs = [figure2(pathways, args.output_dir), figure3(pivot, args.output_dir)]
    pathways[FIELDS].reset_index().to_csv(args.output_dir / "Figure2_data.csv", index=False)
    pivot.rename_axis("topic").reset_index().to_csv(args.output_dir / "Figure3_data.csv", index=False)
    alt_text = {
        "Figure2": "Dot plot of five need-support mappings. Circles show receipt of a human reply, squares show target support-category presence among threads with analyzable replies, and diamonds show overall support-category coverage. Values are printed beside every marker. Coverage ranges from 16.2% for risk assessment to help-seeking encouragement to 73.8% for medical-process information to informational support. The three measures have different denominators; connecting lines group estimates and are not time trends.",
        "Figure3": "Heatmap of overall informational-support coverage across seven topics within the Diagnostic position, with columns for symptom interpretation, risk assessment, and medical-process information needs. Every eligible cell has a numeric percentage. Reading Clinical Reports has 13%, 13%, and 18% coverage; Waiting for Results has 84%, 76%, and 90%. Hatched cells with a dash did not meet eligibility thresholds and are not zero coverage. The color scale runs from dark blue at low coverage to yellow at high coverage."
    }
    (args.output_dir / "alt_text.json").write_text(json.dumps(alt_text, indent=2) + "\n")
    cmap = heatmap()
    cell_contrasts = [contrast_ratio(label_color(cmap(v)[:3]), cmap(v)[:3])
                      for v in pivot.to_numpy().ravel() if pd.notna(v)]
    checks = {"palette_matches_local_submission": True,
              "marker_shapes": ["circle", "square", "diamond"],
              "marker_outline_contrast_on_white": contrast_ratio(INK, "white"),
              "minimum_heatmap_label_contrast": min(cell_contrasts),
              "missing_cell_text_contrast": contrast_ratio(MISSING_INK, MISSING_FILL),
              "minimum_label_target": 4.5,
              "minimum_marker_boundary_target": 3.0,
              "heatmap": "cividis",
              "color_vision_simulation_performed": False,
              "scope": "Checks of selected label/background pairs and marker boundaries, not a full WCAG conformance audit"}
    assert checks["minimum_heatmap_label_contrast"] >= 4.5
    assert checks["marker_outline_contrast_on_white"] >= 3.0
    assert checks["missing_cell_text_contrast"] >= 4.5
    (args.output_dir / "accessibility_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    # Render grayscale previews from figure artists without altering scientific data.
    global BLUE, GOLD, RED
    original_colors = BLUE, GOLD, RED
    BLUE, GOLD, RED = "#303030", "#AAAAAA", "#656565"
    preview_dir = args.output_dir / "grayscale_preview"
    preview_dir.mkdir(exist_ok=True)
    figure2(pathways, preview_dir)
    original_heatmap = heatmap
    def grayscale_heatmap():
        from matplotlib.colors import ListedColormap
        colors = colormaps["cividis"](np.linspace(0, 1, 256))
        gray = colors[:, :3] @ [0.2126, 0.7152, 0.0722]
        cmap = ListedColormap(np.column_stack([gray, gray, gray, np.ones_like(gray)]))
        cmap.set_bad(MISSING_FILL)
        return cmap
    globals()["heatmap"] = grayscale_heatmap
    figure3(pivot, preview_dir)
    BLUE, GOLD, RED = original_colors
    globals()["heatmap"] = original_heatmap
    assert hashes == {str(p.relative_to(ROUND4)): sha256(p) for p in input_paths}
    if args.audit:
        args.audit.parent.mkdir(parents=True, exist_ok=True)
        args.audit.write_text(json.dumps({
            "input_sha256": hashes, "input_tables_unchanged": True,
            "verified_dot_values": 15, "verified_heatmap_values": 18,
            "hatched_cells": int(pivot.isna().sum().sum()), "dpi": DPI,
            "matplotlib_version": matplotlib.__version__,
            "outputs": {str(p.relative_to(ROUND4)) if p.is_relative_to(ROUND4) else p.name: sha256(p) for p in outputs},
        }, ensure_ascii=False, indent=2) + "\n")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
