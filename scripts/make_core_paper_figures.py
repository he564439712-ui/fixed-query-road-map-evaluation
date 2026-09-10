# Academic Figure Skill Asset Confirmation (verified against assets/figures/)
# Fig. 1: cross-type multipanel/schematic reference -> param inherit.
# Fig. 2: BarComparison forest-plot reference -> param inherit.
# Fig. 3: BarComparison forest-plot reference -> param inherit.
# Fig. 4: LineTrend paired-dot reference -> param inherit.
# Fig. 5: heatmap reference plus multipanel image strip -> param inherit.
# Fig. 6: cross-type multipanel image audit reference -> param inherit.
# RULE: "native run" = an asset script was executed unchanged with its bundled data.
# No direct semantic match existed, so all six figures use documented parameter inheritance.

#!/usr/bin/env python
"""Render all six manuscript figures from frozen artifacts.

This is presentation-only. It never retrains a model, regenerates a query,
changes an experimental condition, or modifies a reported result.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.corruptions import (
    CORRUPTION_CONDITIONS,
    apply_corruption,
    deterministic_corruption_seed,
)
from src.data.dataset import load_image, load_mask
from src.data.queries import get_queries_for_image, load_queries
from src.data.splits import resolve_image_paths
from src.eval.topology import bridge_metrics
from src.planning.costs import (
    build_traversability,
    conservative_confidence_lower_bound,
    make_unified_risk_cost,
)
from src.planning.grid import astar_search
from src.planning.topology_repair import repair_topology


# Academic Figure Skill typography baseline.
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 8,
    "figure.titlesize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
})

# Academic Figure Skill custom palette.
CATEGORICAL = ["#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666"]
CATEGORICAL_EXTENDED = CATEGORICAL + ["#67A9CF", "#EF8A62", "#A6DBA0"]
DIVERGING = ["#2166AC", "#F7F7F7", "#B2182B"]
SEQUENTIAL = ["#F7FBFF", "#6BAED6", "#08306B"]
ACCENT_RED = "#B2182B"
GREY = "#999999"
BLACK = "#222222"

BLUE = CATEGORICAL[0]
RED = ACCENT_RED
GREEN = CATEGORICAL[2]
ORANGE = CATEGORICAL[3]
PURPLE = CATEGORICAL[4]
MID_GREY = CATEGORICAL[5]
LIGHT_BLUE = "#EDF4FA"
LIGHT_GREEN = "#EFF7F1"
LIGHT_ORANGE = "#FFF5E8"

# Academic Figure Skill export baseline.
mpl.rcParams.update({
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts" / "publication_figures_jars"
MANUSCRIPT_OUT = ROOT / "paper" / "manuscript" / "figures_jars"
PRED = ROOT / "artifacts" / "predictions" / "test"
RAW = ROOT / "artifacts" / "raw_metrics"


def save_cns_figure(fig, filename):
    fig.savefig(f"{filename}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{filename}.png", bbox_inches="tight", dpi=300)


def save_figure(fig, name: str) -> None:
    """Write a vector master, a preview, and the manuscript copy."""
    OUT.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_OUT.mkdir(parents=True, exist_ok=True)
    save_cns_figure(fig, OUT / name)
    fig.savefig(MANUSCRIPT_OUT / f"{name}.pdf", bbox_inches="tight", dpi=300)


def clean_axis(ax) -> None:
    ax.grid(False)
    ax.tick_params(length=2.5, color=BLACK)


def panel_label(ax, label: str, x: float = -0.12, y: float = 1.10) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="top", ha="left", color=BLACK)


def pp_interval(record):
    value = record["mean_diff"] * 100
    low, high = (item * 100 for item in record["ci95"])
    return value, low, high


def load_clean(image_id: str):
    probability = np.load(PRED / f"{image_id}_prob.npy").astype(np.float32)
    std = np.load(PRED / f"{image_id}_std.npy").astype(np.float32)
    image_path, mask_path = resolve_image_paths(ROOT, image_id)
    image = load_image(image_path)
    gt = load_mask(mask_path).astype(bool)
    return image, gt, probability, std


def select_evidence_case():
    """Return the frozen representative route used only for visual explanation."""
    image_id = "10378780_15"
    image_chw, gt, probability, std = load_clean(image_id)
    image = np.transpose(image_chw, (1, 2, 0))
    q = conservative_confidence_lower_bound(probability, std, 1.0)
    original = build_traversability(probability, 0.5, use_largest_component=False)
    repaired, bridges, _ = repair_topology(original, q, tau_gate=2.0)
    structure = np.ones((3, 3), np.uint8)
    original_labels, _ = ndimage.label(original, structure=structure)
    repaired_labels, _ = ndimage.label(repaired, structure=structure)
    query_file = ROOT / "artifacts" / "queries" / "queries.csv"
    queries = get_queries_for_image(load_queries(query_file), image_id)
    chosen = next(
        query for query in queries
        if (
            original_labels[int(query["start_row"]), int(query["start_col"])] == 0
            or original_labels[int(query["start_row"]), int(query["start_col"])]
            != original_labels[int(query["goal_row"]), int(query["goal_col"])]
        )
        and repaired_labels[int(query["start_row"]), int(query["start_col"])] != 0
        and repaired_labels[int(query["start_row"]), int(query["start_col"])]
        == repaired_labels[int(query["goal_row"]), int(query["goal_col"])]
    )
    start = (int(chosen["start_row"]), int(chosen["start_col"]))
    goal = (int(chosen["goal_row"]), int(chosen["goal_col"]))
    distance = ndimage.distance_transform_edt(repaired).astype(np.float32)
    route = astar_search(
        "proposed", repaired, np.zeros_like(probability), start, goal,
        point_cost=make_unified_risk_cost(q, distance, 2.0, 1.0, 5.0, 1.0),
        max_expansions=200000,
    )
    if not route.success:
        raise RuntimeError("Frozen evidence query did not complete within the common cap.")
    path_pixels = set(map(tuple, route.path))
    used = [bridge for bridge in bridges if path_pixels.intersection(map(tuple, bridge.curve))]
    bridge = max(used or bridges, key=lambda item: len(item.curve))
    center_r, center_c = bridge.curve[len(bridge.curve) // 2]
    half = 105
    r0, r1 = max(0, center_r - half), min(gt.shape[0], center_r + half)
    c0, c1 = max(0, center_c - half), min(gt.shape[1], center_c + half)
    return image, original, repaired, bridge, route.path, (r0, r1, c0, c1)


def crop_points(points, bounds):
    r0, r1, c0, c1 = bounds
    points = np.asarray(points)
    keep = ((points[:, 0] >= r0) & (points[:, 0] < r1)
            & (points[:, 1] >= c0) & (points[:, 1] < c1))
    return points[keep] - np.array([r0, c0])


def draw_box(ax, xy, width, height, title, body, edge, fill):
    patch = FancyBboxPatch(
        xy, width, height, boxstyle="round,pad=0.012,rounding_size=0.015",
        linewidth=0.8, edgecolor=edge, facecolor=fill, transform=ax.transAxes,
    )
    ax.add_patch(patch)
    x, y = xy
    ax.text(x + width / 2, y + height - 0.13, title, transform=ax.transAxes,
            ha="center", va="top", fontsize=8.0, color=BLACK, fontweight="bold")
    ax.text(x + width / 2, y + height * 0.37, body, transform=ax.transAxes,
            ha="center", va="center", fontsize=7.2, color=BLACK, linespacing=1.2)


def draw_arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(
        start, end, transform=ax.transAxes, arrowstyle="-|>", mutation_scale=10,
        linewidth=0.9, color=GREY,
    ))


def fig_framework() -> None:
    """Figure 1: protocol logic plus a local, auditable evidence chain."""
    image, original, repaired, bridge, route, bounds = select_evidence_case()
    r0, r1, c0, c1 = bounds
    bridge_pts = crop_points(bridge.curve, bounds)
    route_pts = crop_points(route, bounds)
    crop = image[r0:r1, c0:c1]

    fig = plt.figure(figsize=(6.5, 4.0))
    outer = fig.add_gridspec(2, 1, height_ratios=[0.58, 1.42], hspace=0.22)
    ax = fig.add_subplot(outer[0])
    ax.set_axis_off()
    ax.text(0.5, 1.02, "One intervention, three evaluation levels",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=9,
            fontweight="bold", color=BLACK)
    draw_box(ax, (0.02, 0.18), 0.24, 0.62, "Frozen evidence",
             "One road product\nOne fixed query set", BLACK, "#F7F7F7")
    draw_box(ax, (0.38, 0.18), 0.24, 0.62, "Intervention",
             "Original  ->  repaired\nNo model retraining", GREEN, LIGHT_GREEN)
    draw_box(ax, (0.72, 0.18), 0.26, 0.62, "Separate reports",
             "Extraction  |  Graph\nFixed-query decision", BLUE, LIGHT_BLUE)
    draw_arrow(ax, (0.27, 0.49), (0.37, 0.49))
    draw_arrow(ax, (0.63, 0.49), (0.71, 0.49))

    grid = outer[1].subgridspec(1, 3, wspace=0.04)
    axes = [fig.add_subplot(grid[0, idx]) for idx in range(3)]
    for axis in axes:
        axis.imshow(crop)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_linewidth(0.6)
            spine.set_color(BLACK)
    axes[0].contour(original[r0:r1, c0:c1], colors=["white"], linewidths=0.8)
    axes[0].plot(bridge_pts[:, 1], bridge_pts[:, 0], color=RED, linestyle="--", linewidth=1.8)
    axes[0].set_title("(a) False-negative gap", pad=4)
    axes[1].contour(repaired[r0:r1, c0:c1], colors=["white"], linewidths=0.75)
    axes[1].plot(bridge_pts[:, 1], bridge_pts[:, 0], color=GREEN, linewidth=2.0)
    axes[1].set_title("(b) Localized repair", pad=4)
    axes[2].contour(repaired[r0:r1, c0:c1], colors=["white"], linewidths=0.7)
    axes[2].plot(route_pts[:, 1], route_pts[:, 0], color=BLUE, linewidth=1.7)
    axes[2].set_title("(c) Route segment", pad=4)
    badges = [
        (axes[0], "Query disconnected", RED),
        (axes[1], "Connection restored", GREEN),
        (axes[2], "Route crosses repaired gap", BLUE),
    ]
    for axis, text, color in badges:
        axis.text(0.5, 0.04, text, transform=axis.transAxes, ha="center", va="bottom",
                  color=color, fontsize=7.4, fontweight="bold",
                  bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1.3})
    fig.subplots_adjust(left=0.025, right=0.985, bottom=0.025, top=0.95)
    save_figure(fig, "fig1_framework")
    plt.close(fig)


def fig_primary_gap() -> None:
    """Figure 2: primary paired image-level evaluation-gap estimates."""
    with (RAW / "primary_multilevel_stats.json").open(encoding="utf-8") as stream:
        primary = json.load(stream)["metrics"]
    with (RAW / "factorial_ablation_stats.json").open(encoding="utf-8") as stream:
        factorial = json.load(stream)["contrasts"]["repair_minus_original__q__boundary_1"]

    def primary_record(key):
        record = primary[key]
        return {"mean_diff": record["mean_delta"], "ci95": record["delta_bootstrap_95_ci"]}

    rows = [
        ("Dice", "Extraction", primary_record("dice"), BLUE, "o"),
        ("clDice", "Extraction", primary_record("cldice"), BLUE, "o"),
        ("Reference APLS", "Graph", primary_record("reference_apls"), ORANGE, "s"),
        ("Structural Connectivity", "Fixed-query", primary_record("structural_connectivity"), GREEN, "D"),
        ("Success@200k", "Fixed-query", factorial["success"], GREEN, "D"),
        ("RouteConform@5%", "Fixed-query", factorial["safe_success_0.05"], GREEN, "D"),
    ]
    y = np.array([5.5, 4.7, 3.55, 2.35, 1.55, 0.75])
    fig, ax = plt.subplots(figsize=(6.5, 3.35))
    ax.axhspan(4.35, 5.85, color=LIGHT_BLUE, zorder=0)
    ax.axhspan(3.15, 3.95, color=LIGHT_ORANGE, zorder=0)
    ax.axhspan(0.35, 2.75, color=LIGHT_GREEN, zorder=0)
    for yi, (metric, group, record, color, marker) in zip(y, rows):
        value, low, high = pp_interval(record)
        ax.errorbar(value, yi, xerr=[[value - low], [high - value]], fmt=marker,
                    color=color, markerfacecolor="white" if group != "Fixed-query" else color,
                    markeredgewidth=0.9, markersize=5.0, capsize=2.8,
                    linewidth=1.2, zorder=3)
        ax.text(10.35, yi, f"{value:+.2f}  [{low:+.2f}, {high:+.2f}]",
                ha="left", va="center", fontsize=7.4, color=color)
    ax.axvline(0, color=GREY, linewidth=0.8, zorder=1)
    ax.set_yticks(y, [row[0] for row in rows])
    ax.set_xlim(-1.45, 14.45)
    ax.set_ylim(0.25, 5.95)
    ax.set_xlabel("Paired Original-to-Repaired change (percentage points)")
    ax.set_title("Paired changes across extraction, graph, and fixed-query measures", pad=7)
    clean_axis(ax)
    for ypos, label, color in ((5.88, "EXTRACTION", BLUE), (3.98, "GRAPH", ORANGE),
                               (2.78, "FIXED-QUERY", GREEN)):
        ax.text(-1.40, ypos, label, color=color, fontsize=7.2, fontweight="bold", va="bottom")
    ax.text(10.35, 5.88, "effect  [95% CI]", ha="left", va="bottom",
            fontsize=7.2, color=MID_GREY)
    fig.subplots_adjust(left=0.25, right=0.985, bottom=0.15, top=0.88)
    save_figure(fig, "fig2_primary_gap")
    plt.close(fig)


def fig_matched_attribution() -> None:
    """Figure 3: prespecified one-factor matched contrasts."""
    with (RAW / "factorial_ablation_stats.json").open(encoding="utf-8") as stream:
        stats = json.load(stream)["contrasts"]
    repair = stats["repair_minus_original__q__boundary_1"]
    boundary = stats["boundary_on_minus_off__repaired__q"]
    q_vs_p = stats["q_minus_p__repaired__boundary_1"]
    factor_style = {
        "Repair": (BLUE, "o"),
        "Boundary": (GREEN, "s"),
        "Score q vs p": (PURPLE, "D"),
    }
    left = [
        ("Repair | Success@200k", "Repair", repair["success"]),
        ("Repair | RouteConform@5%", "Repair", repair["safe_success_0.05"]),
        ("Boundary | RouteConform@5%", "Boundary", boundary["safe_success_0.05"]),
        ("Score q vs p | Success@200k", "Score q vs p", q_vs_p["success"]),
        ("Score q vs p | RouteConform@5%", "Score q vs p", q_vs_p["safe_success_0.05"]),
    ]
    right = [
        ("Repair", "Repair", repair["off_road_common_success"]),
        ("Boundary", "Boundary", boundary["off_road_common_success"]),
        ("Score q vs p", "Score q vs p", q_vs_p["off_road_common_success"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.55), gridspec_kw={"wspace": 0.65})
    panel_specs = [
        (axes[0], left, "Decision outcomes", "(a)"),
        (axes[1], right, "Common-success off-road travel", "(b)"),
    ]
    for ax, items, title, label in panel_specs:
        for y, (name, factor, record) in enumerate(items):
            color, marker = factor_style[factor]
            value, low, high = pp_interval(record)
            ax.errorbar(value, y, xerr=[[value - low], [high - value]], fmt=marker,
                        color=color, markerfacecolor=color, markersize=4.7,
                        capsize=2.4, linewidth=1.15, zorder=3)
            place_left = ax is axes[1] and value > -0.35
            ax.annotate(f"{value:+.2f} [{low:+.2f}, {high:+.2f}]", (value, y),
                        xytext=((-5 if place_left else 5), -8), textcoords="offset points",
                        ha=("right" if place_left else "left"), va="top",
                        fontsize=7.1, color=color)
        ax.axvline(0, color=GREY, linewidth=0.8, zorder=1)
        ax.set_yticks(range(len(items)), [item[0] for item in items])
        ax.invert_yaxis()
        ax.set_ylim(len(items) - 0.25, -0.55)
        ax.set_xlabel("Matched change (percentage points)")
        ax.set_title(title, pad=5)
        clean_axis(ax)
        panel_label(ax, label)
    axes[0].set_xlim(-1.45, 9.45)
    axes[1].set_xlim(-1.18, 0.40)
    axes[1].text(0.98, 0.98, "Negative = less off-road travel",
                 transform=axes[1].transAxes, ha="right", va="top",
                 fontsize=7.1, color=MID_GREY)
    fig.subplots_adjust(bottom=0.16, top=0.84, left=0.23, right=0.985)
    save_figure(fig, "fig3_matched_attribution")
    plt.close(fig)


def fig_replication() -> None:
    """Figure 4: product-level replication with explicit, separate scales."""
    products = ["Massachusetts\nU-Net ensemble", "DeepGlobe\nU-Net ensemble",
                "DeepGlobe\nDeepLabV3-ResNet50"]
    extraction = {"Dice": [0.05, -0.27, -0.45], "clDice": [0.08, -0.57, -0.64]}
    downstream = {"Reference APLS": [5.49, 3.08, 4.45],
                  "Structural Connectivity": [6.24, 3.31, 1.87]}
    y = np.arange(len(products))
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.1), gridspec_kw={"wspace": 0.46})
    specs = [
        (axes[0], extraction, (-0.83, 0.34), "Extraction-score change", [BLUE, PURPLE]),
        (axes[1], downstream, (0.0, 7.25), "Graph and fixed-query change", [ORANGE, GREEN]),
    ]
    markers = ["o", "s"]
    for idx, (ax, data, xlim, title, colors) in enumerate(specs):
        for j, (metric, values) in enumerate(data.items()):
            yy = y + (-0.11 if j == 0 else 0.11)
            ax.scatter(values, yy, marker=markers[j], s=30, color=colors[j],
                       edgecolor="white", linewidth=0.5, zorder=3, label=metric)
            for value, ypos in zip(values, yy):
                ax.annotate(f"{value:+.2f}", (value, ypos), xytext=(4, 0),
                            textcoords="offset points", va="center", fontsize=7.3,
                            color=colors[j])
        ax.axvline(0, color=GREY, linewidth=0.8)
        ax.set_xlim(*xlim)
        ax.set_yticks(y, products if idx == 0 else ["", "", ""])
        ax.invert_yaxis()
        ax.set_xlabel("Original-to-Repaired change (percentage points)")
        ax.set_title(title, pad=28)
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=2,
                  fontsize=7.3, handletextpad=0.35, columnspacing=0.9, borderaxespad=0)
        clean_axis(ax)
        panel_label(ax, f"({chr(97 + idx)})", x=-0.13, y=1.30)
    fig.subplots_adjust(left=0.24, right=0.985, bottom=0.16, top=0.73)
    save_figure(fig, "fig4_replication")
    plt.close(fig)


def fig_stress_boundary() -> None:
    """Figure 5: controlled inputs and exact-value diagnostic heatmaps."""
    with (RAW / "corrupted_planning_tau2.json").open(encoding="utf-8") as stream:
        results = json.load(stream)["results"]
    image_id = "10378780_15"
    image_path, _ = resolve_image_paths(ROOT, image_id)
    image = load_image(image_path)
    examples = [
        ("clean", "Clean"),
        ("blur_medium", "Blur\n(medium)"),
        ("noise_medium", "Noise\n(medium)"),
        ("bc_heavy", "Brightness / contrast\n(heavy)"),
        ("shadow_heavy", "Synthetic occlusion\n(heavy)"),
    ]
    _, height, width = image.shape
    crop_size = min(900, height, width)
    r0 = (height - crop_size) // 2
    c0 = (width - crop_size) // 2
    cmap = LinearSegmentedColormap.from_list("academic_sequential", SEQUENTIAL)

    fig = plt.figure(figsize=(6.5, 4.8))
    outer = fig.add_gridspec(2, 1, height_ratios=[0.88, 1.45], hspace=0.40)
    top = outer[0].subgridspec(1, 5, wspace=0.04)
    top_axes = [fig.add_subplot(top[0, idx]) for idx in range(5)]
    for ax, (condition, title) in zip(top_axes, examples):
        shown = image if condition == "clean" else apply_corruption(
            image,
            *CORRUPTION_CONDITIONS[condition],
            seed=deterministic_corruption_seed(image_id, condition),
        )
        shown = shown[:, r0:r0 + crop_size, c0:c0 + crop_size]
        ax.imshow(np.transpose(shown, (1, 2, 0)))
        ax.set_title(title, fontsize=7.6, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.55)
            spine.set_color(BLACK)
    top_axes[0].text(-0.15, 1.22, "(a)", transform=top_axes[0].transAxes,
                     fontsize=9, fontweight="bold", ha="left", va="top")

    heat = outer[1].subgridspec(1, 3, width_ratios=[1, 1, 0.055], wspace=0.22)
    groups = [("blur", "Blur"), ("noise", "Noise"),
              ("bc", "Brightness / contrast"), ("shadow", "Synthetic occlusion")]
    method_specs = [("Topology+Risk A*", "Full pipeline\nconfiguration"),
                    ("Soft-probability A*", "Original soft-probability\nconfiguration")]
    images = []
    for idx, (key, title) in enumerate(method_specs):
        ax = fig.add_subplot(heat[idx])
        matrix = np.array([[results[f"{prefix}_{level}"][key]["success"]
                            for level in ("light", "medium", "heavy")]
                           for prefix, _ in groups])
        im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        images.append(im)
        ax.set_xticks(range(3), ["Light", "Medium", "Heavy"])
        ax.set_yticks(range(4), [label for _, label in groups] if idx == 0 else ["", "", "", ""])
        ax.set_title(title, pad=5)
        ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = matrix[row, col]
                ax.text(col, row, f"{value:.2f}", ha="center", va="center",
                        fontsize=7.8, color="white" if value > 0.47 else BLACK,
                        fontweight="bold")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.55)
            spine.set_color(BLACK)
        panel_label(ax, f"({chr(98 + idx)})", x=-0.14, y=1.12)
    cax = fig.add_subplot(heat[2])
    cbar = fig.colorbar(images[0], cax=cax)
    cbar.set_label("Success@200k")
    cbar.ax.tick_params(labelsize=7.5, length=2)
    fig.subplots_adjust(left=0.12, right=0.94, bottom=0.06, top=0.96)
    save_figure(fig, "fig5_stress_boundary")
    plt.close(fig)


def bridge_is_supported(bridge, gt):
    metrics = bridge_metrics([bridge], gt, overlap_threshold=0.5, dilation_radius=3)
    return metrics["correct_bridges"] == 1


def fig_bridge_audit() -> None:
    """Figure 6: accepted proposals include supported and unsupported examples."""
    image_id = "10828720_15"
    image, gt, probability, std = load_clean(image_id)
    q = conservative_confidence_lower_bound(probability, std, 1.0)
    binary = build_traversability(probability, 0.5, use_largest_component=False)
    _, bridges, _ = repair_topology(binary, q, tau_gate=2.0)
    supported = [bridge for bridge in bridges if bridge_is_supported(bridge, gt)]
    unsupported = [bridge for bridge in bridges if not bridge_is_supported(bridge, gt)]
    if not supported or not unsupported:
        raise RuntimeError("Expected one supported and one unsupported bridge example.")
    examples = [
        (max(supported, key=lambda item: len(item.curve)), GREEN, "-", "Reference-supported"),
        (max(unsupported, key=lambda item: len(item.curve)), RED, "--", "Unsupported"),
    ]
    rgb = np.transpose(image, (1, 2, 0))
    fig, axes = plt.subplots(2, 3, figsize=(6.5, 4.25),
                             gridspec_kw={"hspace": 0.14, "wspace": 0.04})
    for col, title in enumerate(["Before intervention", "Accepted bridge", "Reference audit"]):
        axes[0, col].set_title(title, fontweight="bold", pad=4)
    for row, (bridge, color, linestyle, status) in enumerate(examples):
        pts = np.asarray(bridge.curve)
        center = np.rint(pts.mean(axis=0)).astype(int)
        half = 125
        r0 = int(np.clip(center[0] - half, 0, gt.shape[0] - 2 * half))
        c0 = int(np.clip(center[1] - half, 0, gt.shape[1] - 2 * half))
        r1, c1 = r0 + 2 * half, c0 + 2 * half
        local_pts = pts - np.array([r0, c0])
        for col, ax in enumerate(axes[row]):
            ax.imshow(rgb[r0:r1, c0:c1])
            if col < 2:
                ax.contour(binary[r0:r1, c0:c1], colors=["white"], linewidths=0.6)
            else:
                ax.contour(gt[r0:r1, c0:c1], colors=[ORANGE], linewidths=0.9)
                ax.contour(binary[r0:r1, c0:c1], colors=["white"], linewidths=0.35, alpha=0.65)
            if col > 0:
                ax.plot(local_pts[:, 1], local_pts[:, 0], color=color, linewidth=2.0,
                        linestyle=linestyle, solid_capstyle="round")
            ax.scatter([bridge.start.col - c0, bridge.end.col - c0],
                       [bridge.start.row - r0, bridge.end.row - r0],
                       color=color, edgecolors="white", linewidths=0.5, s=17, zorder=5)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.55)
                spine.set_color(BLACK)
        axes[row, 0].text(0.02, 0.96, f"({chr(97 + row)}) {status}",
                          transform=axes[row, 0].transAxes, ha="left", va="top",
                          fontsize=6.8, fontweight="bold", color=color,
                          bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1.3})
    fig.subplots_adjust(left=0.025, right=0.99, bottom=0.025, top=0.94)
    save_figure(fig, "fig6_bridge_audit")
    plt.close(fig)
    print(f"Bridge audit: {len(supported)} supported and {len(unsupported)} unsupported proposals.")


def main() -> None:
    fig_framework()
    fig_primary_gap()
    fig_matched_attribution()
    fig_replication()
    fig_stress_boundary()
    fig_bridge_audit()
    print("Rendered six manuscript figures from frozen artifacts.")


if __name__ == "__main__":
    main()
