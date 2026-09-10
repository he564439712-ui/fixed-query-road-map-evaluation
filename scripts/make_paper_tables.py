#!/usr/bin/env python
"""Generate LaTeX tables for the paper from raw metrics CSVs.

Reads artifacts/raw_metrics/*.csv and writes paper/tables/*.tex with
ready-to-paste LaTeX tables (booktabs style).

Usage:
    python scripts/make_paper_tables.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "artifacts" / "raw_metrics"
OUT = ROOT / "paper" / "tables"


def read_csv(name):
    with (RAW / name).open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt(v, nd=3):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "---"
    if np.isnan(f):
        return "---"
    return f"{f:.{nd}f}"


def esc(s: str) -> str:
    """Escape underscores for LaTeX (condition names like blur_light)."""
    return s.replace("_", "\\_")


def latex_table(header, rows, caption, label, col_spec):
    """Full-width booktabs table via tabularx; L=left-aligned flexible,
    R=right-aligned flexible (numbers)."""
    lines = [f"% {caption}", r"\begin{table}[htbp]", r"\centering",
             r"\footnotesize",
             f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             f"\\begin{{tabularx}}{{\\textwidth}}{{{col_spec}}}", r"\toprule"]
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")
    for r in rows:
        lines.append(" & ".join(map(str, r)) + r" \\")
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table}", ""]
    return "\n".join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- Table: Segmentation (M1) ----
    m1 = [
        ["42", "0.7449", "0.5954"],
        ["123", "0.7503", "0.6026"],
        ["456", "0.7453", "0.5963"],
    ]
    (OUT / "tab_segmentation.tex").write_text(latex_table(
        ["Seed", "Val Dice", "Val IoU"], m1,
        "Segmentation quality of the three U-Net ensemble members (Massachusetts validation set).",
        "tab:segmentation", "LRR"), encoding="utf-8")

    # ---- Table: Calibration (M2) ----
    m2 = [
        ["Single model (mean)", "0.0152", "0.0187", "0.0694", "---"],
        ["Deep Ensemble", "0.0146", "0.0178", "0.0656", "0.9377"],
        ["Ensemble + TS (T=0.98)", "0.0144", "0.0179", "0.0659", "---"],
        ["MC Dropout single", "0.0261", "0.0234", "0.0851", "---"],
        ["MC Dropout (30)", "0.0266", "0.0220", "0.0794", "0.9330"],
    ]
    (OUT / "tab_calibration.tex").write_text(latex_table(
        ["Method", "ECE$\\downarrow$", "Brier$\\downarrow$", "NLL$\\downarrow$", "AUROC$\\uparrow$"],
        m2, "Calibration and uncertainty quality on the clean test set.", "tab:calibration",
        "LRRRR"), encoding="utf-8")

    # ---- Table: Topology repair (M3) ----
    topo = json.loads((RAW / "massachusetts_test_seg_summary.json").read_text(encoding="utf-8"))
    mass_apls = json.loads((RAW / "massachusetts_apls_reference.json").read_text(encoding="utf-8"))["aggregate"]
    gate = json.loads((RAW / "gate_sweep_test.json").read_text(encoding="utf-8"))["results"]["2.0"]
    conn_o = topo["connectivity"]["original"]
    conn_r = topo["connectivity"]["repaired"]
    metrics = topo["metrics"]
    m3 = [
        ["Connected components / image", f"{metrics['components_orig']['mean']:.2f}",
         f"{metrics['components_repaired']['mean']:.2f}"],
        ["Structural Connectivity (fixed queries)",
         f"{conn_o['connected']}/{conn_o['total']} ({conn_o['ratio']:.4f})",
         f"{conn_r['connected']}/{conn_r['total']} ({conn_r['ratio']:.4f})"],
        ["Dice", f"{metrics['dice']['mean']:.4f}",
         f"{metrics['dice_repaired']['mean']:.4f}"],
        ["clDice", f"{metrics['cldice']['mean']:.4f}",
         f"{metrics['cldice_repaired']['mean']:.4f}"],
        ["SpaceNet APLS audit (raster graph)", f"{mass_apls['original_mean']:.4f}",
         f"{mass_apls['repaired_mean']:.4f}"],
        ["Accepted bridges / image", "---", f"{gate['num_bridges']:.2f}"],
        ["Bridge precision", "---", f"{gate['bridge_precision']:.3f}"],
    ]
    (OUT / "tab_topology.tex").write_text(latex_table(
        ["Metric", "Original", "Repaired"], m3,
        "Topology repair on the 49-image clean test set at "
        "$\\tau_{\\mathrm{gate}}=2.0$. The SpaceNet reference-implementation "
        "APLS audit uses its large-graph comparison on raster-derived graphs; it is "
        "definition-compatible but not a native-georeferenced SpaceNet "
        "submission. Structural Connectivity is compute-independent reachability: it is evaluated before the A$^*$ search cap.",
        "tab:topology", "LRR"), encoding="utf-8")

    # ---- Table: Cross-dataset overlap, graph, and query evidence ----
    dg_unet = json.loads((RAW / "deepglobe_unet_apls_reference.json").read_text(encoding="utf-8"))["aggregate"]
    dg_deeplab = json.loads((RAW / "deepglobe_deeplabv3_apls_reference.json").read_text(encoding="utf-8"))["aggregate"]

    def pair(metric):
        return f"{metric['original_mean']:.4f} $\\to$ {metric['repaired_mean']:.4f}"

    def conn_pair(metric):
        original, repaired = metric["original"], metric["repaired"]
        return (f"{original['connected']}/{original['total']} ({original['ratio']:.4f}) "
                f"$\\to$ {repaired['connected']}/{repaired['total']} ({repaired['ratio']:.4f})")

    cross_rows = [
        ["Massachusetts / U-Net ensemble",
         f"{metrics['dice']['mean']:.4f} $\\to$ {metrics['dice_repaired']['mean']:.4f}",
         f"{metrics['cldice']['mean']:.4f} $\\to$ {metrics['cldice_repaired']['mean']:.4f}",
         f"{mass_apls['original_mean']:.4f} $\\to$ {mass_apls['repaired_mean']:.4f}",
         f"{conn_o['connected']}/{conn_o['total']} ({conn_o['ratio']:.4f}) $\\to$ "
         f"{conn_r['connected']}/{conn_r['total']} ({conn_r['ratio']:.4f})"],
        ["DeepGlobe / U-Net ensemble", pair(dg_unet["dice"]), pair(dg_unet["cldice"]),
         pair(dg_unet["apls"]), conn_pair(dg_unet["query_connectivity"])],
        ["DeepGlobe / DeepLabV3--ResNet50", pair(dg_deeplab["dice"]),
         pair(dg_deeplab["cldice"]), pair(dg_deeplab["apls"]),
         conn_pair(dg_deeplab["query_connectivity"])],
    ]
    cross_caption = (
        "Original $\\to$ repaired evidence across frozen road products. The "
        "SpaceNet reference-implementation APLS audit uses its large-graph comparison on "
        "raster-derived graphs (definition-compatible, not native-georeferenced "
        "SpaceNet submissions). Structural Connectivity is compute-independent reachability for the stated "
        "start--goal query sets (945 Massachusetts; 10,620 DeepGlobe).")
    (OUT / "tab_topology_transfer.tex").write_text(latex_table(
        ["Dataset / backbone", "Dice", "clDice", "SpaceNet APLS audit",
         "Structural Connectivity (fixed queries)"],
        cross_rows, cross_caption, "tab:topology_transfer", "LRRRR"), encoding="utf-8")
    # ---- Table: Controlled 2x2x2 ablation ----
    factorial = json.loads((RAW / "factorial_ablation.json").read_text(encoding="utf-8"))["results"]
    factorial_order = [
        ("original", "p", False), ("original", "p", True),
        ("original", "q", False), ("original", "q", True),
        ("repaired", "p", False), ("repaired", "p", True),
        ("repaired", "q", False), ("repaired", "q", True),
    ]
    ablation_rows = []
    for mask, score, boundary in factorial_order:
        key = f"{mask}+{score}+boundary_{'on' if boundary else 'off'}"
        row = factorial[key]
        ablation_rows.append([
            mask.capitalize(), f"${score}$", "on" if boundary else "off",
            f"{row['success']:.3f}", f"{row['off_road']:.4f}",
            f"{row['safe_success_0.05']:.3f}",
        ])
    (OUT / "tab_ablation.tex").write_text(latex_table(
        ["Mask", "Score", "Boundary term", "Success@200k$\\uparrow$",
         "Off-road$\\downarrow$", "Conform@5\\%$\\uparrow$"],
        ablation_rows,
        "Controlled $2\\times2\\times2$ ablation on 945 clean-test queries. "
        "The $p$ and $q$ variants use the same risk weight and cap; "
        "RouteConform@5\\% requires completion within the shared cap and an off-reference ratio no "
        "greater than 5\\%.",
        "tab:ablation", "LLLRRR"), encoding="utf-8")
    # ---- Table: Planning (M4) ----
    # Use the entropy-penalty value selected on calibration and evaluated once
    # on the locked test set, rather than the earlier exploratory planning.csv.
    m4_rows = read_csv("planning_ura_calibrated_test.csv")
    geometric_rows = read_csv("geometric_baseline_locked_test.csv")
    m4_rows.extend(
        row for row in geometric_rows
        if row["method"] == "Geometric endpoint repair + Risk A*"
    )
    entropy_method = (
        "Entropy-penalized A*"
        if any(r["method"] == "Entropy-penalized A*" for r in m4_rows)
        else "URA-style A*"
    )
    methods_order = ["Binary-mask A*", "Soft-probability A*", "Clearance-aware A*",
                     entropy_method, "Geometric endpoint repair + Risk A*",
                     "Topology+Risk A*", "GT Oracle A*"]
    transfer_methods_order = [
        "Binary-mask A*", "Soft-probability A*", "Clearance-aware A*",
        entropy_method, "Topology+Risk A*", "GT Oracle A*",
    ]
    m4 = []
    for m in methods_order:
        rows = [r for r in m4_rows if r["method"] == m]
        succ = float(np.mean([int(r["success"]) for r in rows]))
        off = [float(r["off_road_ratio"]) for r in rows if r["off_road_ratio"] not in ("", "nan")]
        det = [float(r["relative_detour"]) for r in rows if r["relative_detour"] not in ("", "nan")]
        off_m = float(np.mean(off)) if off else float("nan")
        det_med = float(np.median(det)) if det else float("nan")
        safe = float(np.mean([
            int(r["success"]) and r["off_road_ratio"] not in ("", "nan")
            and float(r["off_road_ratio"]) <= 0.05
            for r in rows
        ]))
        name = {"Binary-mask A*": "Binary-mask", "Soft-probability A*": "Soft-probability",
                "Clearance-aware A*": "Clearance-aware", "URA-style A*": "Entropy-penalized",
                "Entropy-penalized A*": "Entropy-penalized",
                "Geometric endpoint repair + Risk A*": "Geometric endpoint repair + Risk",
                "Topology+Risk A*": "Repair + boundary-aware A$^*$", "GT Oracle A*": "GT oracle (upper bound)"}[m]
        m4.append([name, f"{succ:.3f}", f"{safe:.3f}", f"{off_m:.3f}", f"{det_med:.3f}"])
    planning_table = latex_table(
        ["Method", "Success@200k$\\uparrow$", "Conform@5\\%$\\uparrow$",
         "Off-road$\\downarrow$", "Detour (median)$\\downarrow$"],
        m4,
        "Path-planning comparison on the clean test set (945 fixed queries). "
        "Conform@5\\% denotes RouteConform@5\\%; off-road is conditional on "
        "paths completed within the shared 200k-expansion cap; the geometric-repair "
        "gate was selected on the calibration split.",
        "tab:planning", "LRRRR")
    planning_note = (
        "\\vspace{1mm}\n\\raggedright\\scriptsize Relative detour is measured "
        "against the shortest reference-mask path. Small negative values can "
        "occur when a predicted false-positive shortcut is shorter than that "
        "reference-constrained path.\n"
    )
    planning_table = planning_table.replace(
        "\\end{table}\n", planning_note + "\\end{table}\n"
    )
    (OUT / "tab_planning.tex").write_text(planning_table, encoding="utf-8")

    # ---- Table: Degraded calibration (M5.1) ----
    m5a = read_csv("corrupted_calibration.csv")
    rows5 = []
    for r in m5a:
        rel = (float(r["ece"]) - 0.0146) / 0.0146 * 100 if r["condition"] != "clean" else 0.0
        rows5.append([esc(r["condition"]), fmt(r["ece"]), f"{rel:+.0f}\\%",
                      fmt(r["brier"]), fmt(r["nll"]), fmt(r["auroc"])])
    (OUT / "tab_degraded_calibration.tex").write_text(latex_table(
        ["Condition", "ECE", "ECE $\\Delta$ vs clean", "Brier", "NLL", "AUROC"],
        rows5, "Calibration degradation under image corruptions (clean ECE = 0.0146).",
        "tab:degraded_cal", "LRRRRR"), encoding="utf-8")

    # ---- Table: Temperature scaling under degradation (M5.2) ----
    m5b = read_csv("corrupted_temperature.csv")
    rowsb = []
    for r in m5b:
        rowsb.append([esc(r["condition"]), fmt(r["T"], 2), fmt(r["ece_before"]),
                      fmt(r["ece_after"]), f"{float(r['ece_rel_pct']):+.0f}\\%",
                      fmt(r["nll_before"]), fmt(r["nll_after"])])
    (OUT / "tab_temp_scaling.tex").write_text(latex_table(
        ["Condition", "$T$", "ECE pre-TS", "ECE post-TS", "$\\Delta$ ECE",
         "NLL pre-TS", "NLL post-TS"],
        rowsb, "Per-condition temperature scaling: $T$ optimized on calibration set, evaluated on test.",
        "tab:temp_scaling", "LRRRRRR"), encoding="utf-8")

    # ---- Table: DeepGlobe task-level planning transfer ----
    deepglobe_rows = read_csv("deepglobe_planning_tau2.csv")
    deepglobe_table = []
    for method in transfer_methods_order:
        rows = [r for r in deepglobe_rows if r["method"] == method]
        success = float(np.mean([int(r["success"]) for r in rows]))
        off = [float(r["off_road_ratio"]) for r in rows
               if r["off_road_ratio"] not in ("", "nan")]
        detour = [float(r["relative_detour"]) for r in rows
                  if r["relative_detour"] not in ("", "nan")]
        name = {"Binary-mask A*": "Binary-mask", "Soft-probability A*": "Soft-probability",
                "Clearance-aware A*": "Clearance-aware", "URA-style A*": "Entropy-penalized",
                "Entropy-penalized A*": "Entropy-penalized",
                "Topology+Risk A*": "Repair + boundary-aware A$^*$",
                "GT Oracle A*": "GT oracle (upper bound)"}[method]
        deepglobe_table.append([
            name, f"{success:.3f}", f"{np.mean(off):.3f}", f"{np.median(detour):.3f}"
        ])
    (OUT / "tab_deepglobe_planning.tex").write_text(latex_table(
        ["Method", "Success@200k$\\uparrow$", "Off-road$\\downarrow$",
         "Detour (median)$\\downarrow$"],
        deepglobe_table,
        "Task-level transfer on DeepGlobe Road Extraction (10{,}620 fixed "
        "queries over 624 held-out images; $\\tau_{\\mathrm{gate}}=2.0$ "
        "frozen from Massachusetts calibration).",
        "tab:deepglobe_planning", "LRRR"), encoding="utf-8")

    # ---- Table: stronger-backbone DeepGlobe planning check ----
    deeplab_rows = read_csv("deepglobe_deeplabv3_planning_tau2.csv")
    deeplab_table = []
    for method in transfer_methods_order:
        rows = [r for r in deeplab_rows if r["method"] == method]
        success = float(np.mean([int(r["success"]) for r in rows]))
        safe = float(np.mean([
            int(r["success"]) and r["off_road_ratio"] not in ("", "nan")
            and float(r["off_road_ratio"]) <= 0.05 for r in rows
        ]))
        off = [float(r["off_road_ratio"]) for r in rows
               if r["off_road_ratio"] not in ("", "nan")]
        detour = [float(r["relative_detour"]) for r in rows
                  if r["relative_detour"] not in ("", "nan")]
        name = {"Binary-mask A*": "Binary-mask", "Soft-probability A*": "Soft-probability",
                "Clearance-aware A*": "Clearance-aware", "URA-style A*": "Entropy-penalized",
                "Entropy-penalized A*": "Entropy-penalized",
                "Topology+Risk A*": "Repair + boundary-aware A$^*$",
                "GT Oracle A*": "GT oracle (upper bound)"}[method]
        deeplab_table.append([
            name, f"{success:.3f}", f"{safe:.3f}", f"{np.mean(off):.3f}",
            f"{np.median(detour):.3f}",
        ])
    (OUT / "tab_deepglobe_deeplab_planning.tex").write_text(latex_table(
        ["Method", "Success@200k$\\uparrow$", "Conform@5\\%$\\uparrow$",
         "Off-road$\\downarrow$", "Detour (median)$\\downarrow$"],
        deeplab_table,
        "Frozen planning protocol on the stronger DeepGlobe DeepLabV3--ResNet50 "
        "backbone (10{,}620 fixed queries). DeepLabV3 is a single model, so "
        "$q=\\bar p$ in the repair + boundary-aware condition; off-road is conditional on paths completed within the shared cap.",
        "tab:deepglobe_deeplab_planning", "LRRRR"), encoding="utf-8")

    # ---- Table: Complete corrupted planning at frozen tau_gate=2.0 ----
    corrupted = json.loads(
        (RAW / "corrupted_planning_tau2.json").read_text(encoding="utf-8")
    )["results"]
    corrupted_table = []
    for condition in m5a:
        name = condition["condition"]
        proposed = corrupted[name]["Topology+Risk A*"]
        soft = corrupted[name]["Soft-probability A*"]
        corrupted_table.append([
            esc(name), f"{proposed['success']:.3f}", f"{soft['success']:.3f}",
            f"{100 * (proposed['success'] - soft['success']):+.1f}",
            f"{proposed['safe_success_0.05']:.3f}",
            f"{soft['safe_success_0.05']:.3f}",
        ])
    (OUT / "tab_degraded_planning.tex").write_text(latex_table(
        ["Condition", "Ours Success@200k", "Soft Success@200k", "$\\Delta$ (pp)",
         "Ours Conform@5\\%", "Soft Conform@5\\%"],
        corrupted_table,
        "Complete downstream planning robustness under corruptions using the "
        "frozen $\\tau_{\\mathrm{gate}}=2.0$ protocol (945 fixed queries per condition).",
        "tab:degraded_planning", "LRRRRR"), encoding="utf-8")

    # The manuscript deliberately excludes calibration, exhaustive degradation,
    # and redundant transfer-detail tables. Keep their generation code as an
    # analysis archive, but never leave stale, unreferenced paper assets behind.
    retired_outputs = (
        "tab_calibration.tex",
        "tab_deepglobe_deeplab_planning.tex",
        "tab_deepglobe_planning.tex",
        "tab_degraded_calibration.tex",
        "tab_degraded_planning.tex",
        "tab_segmentation.tex",
        "tab_temp_scaling.tex",
    )
    for filename in retired_outputs:
        (OUT / filename).unlink(missing_ok=True)

    print(f"Wrote tables to {OUT}")


if __name__ == "__main__":
    main()
