#!/usr/bin/env python
"""Bridge diagnostics for topology repair and the case-study path lengths.

M1 — false-bridge impact (clean test, q-gated repair, q-cost planning):
  For each successful proposed path, does it traverse a bridge pixel, and is
  that bridge correct or wrong (dilated-GT overlap >= 0.5)? Reports:
    (a) fraction of successful paths that use >=1 bridge;
    (b) among those, correct vs wrong bridge usage and their off-road ratio;
    (c) planning success when ALL wrong bridges are removed (correct-only mask);
    (d) a per-bridge CSV stratified by reference support and actual path use;
    (e) a query-level incremental-success audit.  The latter contrasts the
        original and repaired masks under the same confidence and boundary-aware
        cost, then re-plans after removing every unsupported bridge.  It answers
        whether a newly completed query genuinely requires unsupported repair,
        rather than merely whether its full-condition path happens to touch one.

M9 — case-study consistency: print the actual path lengths for image
  10378780_15 query index 2 (oracle / soft-prob / proposed).

Usage:
    conda run -n thermal python scripts/review_bridges.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import load_mask
from src.data.queries import get_queries_for_image, load_queries
from src.data.splits import build_splits, resolve_image_paths
from src.eval.planning import planning_metrics
from src.eval.topology import bridge_metrics
from src.planning.costs import (
    build_traversability,
    conservative_confidence_lower_bound,
    make_unified_risk_cost,
)
from src.planning.grid import SearchResult, astar_search, largest_connected_road, path_length
from src.planning.topology_repair import repair_topology, apply_bridges_to_mask

MAX_EXP = 200000
LABEL_STRUCT = np.ones((3, 3), np.uint8)


def reachable(labels, start, goal):
    cs, cg = labels[start], labels[goal]
    return cs != 0 and cs == cg


def bridge_is_correct(bridge, gt, dilation_radius=3, overlap_threshold=0.5):
    metrics = bridge_metrics(
        [bridge],
        gt,
        overlap_threshold=overlap_threshold,
        dilation_radius=dilation_radius,
    )
    return metrics["correct_bridges"] == 1


def plan(repaired, q, gt, start, goal, oracle_len, labels, distance,
         lr=2.0, lb=1.0, td=5.0, cap=1.0):
    """Plan once with precomputed image-level topology and distance fields."""
    if not reachable(labels, start, goal):
        r = SearchResult("v", [], 0.0, 0, False)
    else:
        r = astar_search("v", repaired, np.zeros_like(q), start, goal,
                         point_cost=make_unified_risk_cost(q, distance, lr, lb, td, cap),
                         max_expansions=MAX_EXP)
    return planning_metrics(r, gt, oracle_length=oracle_len), r


def main():
    parser = argparse.ArgumentParser(description="False-bridge impact + case-study check.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--prediction-dir", type=Path, default=Path("artifacts/predictions/test"))
    parser.add_argument("--queries-csv", type=Path, default=Path("artifacts/queries/queries.csv"))
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("artifacts/raw_metrics/bridge_diagnostics.csv"),
        help="per-bridge diagnostics; relative paths are resolved against --root",
    )
    parser.add_argument(
        "--query-output-csv",
        type=Path,
        default=Path("artifacts/raw_metrics/bridge_incremental_success_audit.csv"),
        help=("per-query original/full/supported-only audit; relative paths are "
              "resolved against --root"),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--counterfactual-wrong-removal",
        action="store_true",
        help=("also replan after removing every non-reference-supported bridge; "
              "slower and not needed for the primary bridge-use diagnostic"),
    )
    parser.add_argument(
        "--incremental-only",
        action="store_true",
        help=("audit only archived original+q+boundary_on fail -> "
              "repaired+q+boundary_on success queries; recomputed full-condition "
              "success must reproduce the frozen factorial CSV"),
    )
    args = parser.parse_args()

    root = args.root
    splits = build_splits(root)
    image_ids = splits["test_ids"]
    if args.limit:
        image_ids = image_ids[: args.limit]
    queries = load_queries(root / args.queries_csv)
    pred_dir = root / args.prediction_dir

    # The original versus repaired completion labels were already produced by
    # the frozen 2x2x2 factorial ablation.  Reusing those labels restricts the expensive
    # counterfactual re-planning to the exact incremental-success cohort,
    # without re-selecting any query after looking at bridge support.
    archived_incremental: set[int] = set()
    if args.incremental_only:
        ablation_path = root / "artifacts" / "raw_metrics" / "factorial_ablation.csv"
        with ablation_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        success_by_variant = {
            (row["variant"], int(row["query_index"])): int(row["success"])
            for row in rows
            if row["variant"] in {
                "original+q+boundary_on", "repaired+q+boundary_on"
            }
        }
        query_ids = {int(query["query_index"]) for query in queries}
        missing = [
            query_id for query_id in query_ids
            if ("original+q+boundary_on", query_id) not in success_by_variant
            or ("repaired+q+boundary_on", query_id) not in success_by_variant
        ]
        if missing:
            raise RuntimeError(f"Frozen ablation labels missing for query IDs: {missing[:5]}")
        archived_incremental = {
            query_id for query_id in query_ids
            if success_by_variant[("original+q+boundary_on", query_id)] == 0
            and success_by_variant[("repaired+q+boundary_on", query_id)] == 1
        }
        print(
            f"Incremental-only cohort: {len(archived_incremental)} frozen clean-test "
            "original+q+boundary_on fail -> repaired+q+boundary_on success queries",
            flush=True,
        )

    # accumulators for M1
    n_success = 0
    n_use_bridge = 0
    n_use_correct = 0
    n_use_wrong = 0
    off_correct = []
    off_wrong = []
    succ_full = []
    succ_correct_only = []
    bridge_rows: list[dict] = []
    query_rows: list[dict] = []

    for image_id in image_ids:
        pf = pred_dir / f"{image_id}_prob.npy"
        sf = pred_dir / f"{image_id}_std.npy"
        if not pf.exists():
            continue
        prob = np.load(pf).astype(np.float32)
        std = np.load(sf).astype(np.float32) if sf.exists() else np.zeros_like(prob)
        _, mp = resolve_image_paths(root, image_id)
        gt = load_mask(mp).astype(bool)

        q = conservative_confidence_lower_bound(prob, std, 1.0)
        traversable = build_traversability(prob, 0.5, use_largest_component=False)
        repaired, bridges, _ = repair_topology(traversable, q)

        correct = [b for b in bridges if bridge_is_correct(b, gt)]
        wrong = [b for b in bridges if not bridge_is_correct(b, gt)]
        local_bridge_rows = []
        for bridge_index, bridge in enumerate(bridges):
            curve_q = np.asarray([q[r, c] for r, c in bridge.curve], dtype=float)
            components = bridge.cost_components
            bridge_footprint = apply_bridges_to_mask(
                np.zeros_like(traversable, dtype=bool), [bridge]
            )
            record = {
                "image_id": image_id,
                "bridge_index": bridge_index,
                "reference_supported": int(bridge_is_correct(bridge, gt)),
                "curve_nodes": len(bridge.curve),
                "euclidean_gap": float(np.hypot(
                    bridge.start.row - bridge.end.row,
                    bridge.start.col - bridge.end.col,
                )),
                "mean_q": float(np.mean(curve_q)),
                "min_q": float(np.min(curve_q)),
                "total_cost": float(bridge.cost),
                "confidence_cost": float(components.get("confidence", float("nan"))),
                "length_cost": float(components.get("length", float("nan"))),
                "direction_cost": float(components.get("direction", float("nan"))),
                "width_cost": float(components.get("width", float("nan"))),
                "connectivity_gain": float(components.get("connectivity_gain", float("nan"))),
                "used_by_successful_paths": 0,
                "used_by_any_path": 0,
                "_bridge_pixels": set(
                    (int(r), int(c)) for r, c in np.argwhere(bridge_footprint)
                ),
            }
            local_bridge_rows.append(record)
            bridge_rows.append(record)
        correct_px = set()
        for b in correct:
            footprint = apply_bridges_to_mask(np.zeros_like(traversable, dtype=bool), [b])
            for r, c in np.argwhere(footprint):
                correct_px.add((r, c))
        wrong_px = set()
        for b in wrong:
            footprint = apply_bridges_to_mask(np.zeros_like(traversable, dtype=bool), [b])
            for r, c in np.argwhere(footprint):
                wrong_px.add((r, c))
        all_bridge_px = correct_px | wrong_px

        full_labels, _ = ndimage.label(repaired, structure=LABEL_STRUCT)
        full_distance = ndimage.distance_transform_edt(repaired).astype(np.float32)
        original_labels, _ = ndimage.label(traversable, structure=LABEL_STRUCT)
        original_distance = ndimage.distance_transform_edt(traversable).astype(np.float32)
        # This audit always computes the supported-only counterfactual.  It is
        # a frozen, post-hoc diagnostic, not a selection criterion or a new
        # planner: every original setting, bridge, query, and cost parameter is
        # held fixed while unsupported bridges are removed.
        correct_only_mask = apply_bridges_to_mask(traversable, correct) if correct else traversable.copy()
        correct_labels, _ = ndimage.label(correct_only_mask, structure=LABEL_STRUCT)
        correct_distance = ndimage.distance_transform_edt(correct_only_mask).astype(np.float32)

        gt_comp = largest_connected_road(gt)
        gt_labels, _ = ndimage.label(gt_comp, structure=LABEL_STRUCT)

        for query in get_queries_for_image(queries, image_id):
            query_index = int(query["query_index"])
            if args.incremental_only and query_index not in archived_incremental:
                continue
            start = (int(query["start_row"]), int(query["start_col"]))
            goal = (int(query["goal_row"]), int(query["goal_col"]))
            if reachable(gt_labels, start, goal):
                oracle = astar_search("o", gt_comp, np.zeros_like(prob), start, goal, max_expansions=MAX_EXP)
                olen = path_length(oracle.path) if oracle.success else None
            else:
                olen = None

            if args.incremental_only:
                original_success = 0
            else:
                pm_original, _ = plan(
                    traversable, q, gt, start, goal, olen, original_labels, original_distance
                )
                original_success = int(pm_original["success"])
            pm_full, full_result = plan(
                repaired, q, gt, start, goal, olen, full_labels, full_distance
            )
            if args.incremental_only and pm_full["success"] != 1:
                raise RuntimeError(
                    f"Frozen full-condition success did not reproduce for query {query_index}"
                )
            succ_full.append(int(pm_full["success"]))
            pm_corr, _ = plan(
                correct_only_mask, q, gt, start, goal, olen,
                correct_labels, correct_distance,
            )
            succ_correct_only.append(int(pm_corr["success"]))

            path_px = set((int(p[0]), int(p[1])) for p in full_result.path)
            uses_supported = int(bool(path_px & correct_px)) if pm_full["success"] else 0
            uses_unsupported = int(bool(path_px & wrong_px)) if pm_full["success"] else 0
            incremental = int(original_success == 0 and pm_full["success"] == 1)
            query_rows.append({
                "image_id": image_id,
                "query_index": query_index,
                "original_success": original_success,
                "full_success": int(pm_full["success"]),
                "supported_only_success": int(pm_corr["success"]),
                "incremental_success": incremental,
                "full_path_uses_supported_bridge": uses_supported,
                "full_path_uses_unsupported_bridge": uses_unsupported,
                "incremental_success_requires_unsupported_bridge": int(
                    incremental and pm_corr["success"] == 0
                ),
            })

            if pm_full["success"] == 1:
                n_success += 1
                use_bridge = path_px & all_bridge_px
                for record in local_bridge_rows:
                    if path_px & record["_bridge_pixels"]:
                        record["used_by_successful_paths"] += 1
                        record["used_by_any_path"] = 1
                if use_bridge:
                    n_use_bridge += 1
                    use_correct = path_px & correct_px
                    use_wrong = path_px & wrong_px
                    if use_correct:
                        n_use_correct += 1
                        off_correct.append(pm_full["off_road_ratio"])
                    if use_wrong:
                        n_use_wrong += 1
                        off_wrong.append(pm_full["off_road_ratio"])

    # ---- M1 report ----
    print("=== M1: false-bridge impact on planning (clean test) ===")
    print(f"successful proposed paths: {n_success}")
    print(f"  use >=1 bridge:       {n_use_bridge}  ({100*n_use_bridge/max(n_success,1):.1f}%)")
    print(f"  use a CORRECT bridge: {n_use_correct} ({100*n_use_correct/max(n_success,1):.1f}%)  "
          f"off-road mean={np.mean(off_correct) if off_correct else float('nan'):.4f}")
    print(f"  use a WRONG bridge:   {n_use_wrong} ({100*n_use_wrong/max(n_success,1):.1f}%)  "
          f"off-road mean={np.mean(off_wrong) if off_wrong else float('nan'):.4f}")
    if not args.incremental_only:
        print(f"success (full repair, all bridges):   {np.mean(succ_full):.4f}  n={len(succ_full)}")
        print(f"success (supported-only bridges):     {np.mean(succ_correct_only):.4f}  n={len(succ_correct_only)}")
    incremental_rows = [row for row in query_rows if row["incremental_success"]]
    incremental_requires_unsupported = sum(
        row["incremental_success_requires_unsupported_bridge"] for row in incremental_rows
    )
    incremental_uses_unsupported = sum(
        row["full_path_uses_unsupported_bridge"] for row in incremental_rows
    )
    incremental_uses_supported = sum(
        row["full_path_uses_supported_bridge"] for row in incremental_rows
    )
    print("\n=== Incremental-success bridge attribution ===")
    print(f"original fail -> full success:         {len(incremental_rows)}")
    print(f"  full path uses supported bridge:     {incremental_uses_supported}/{len(incremental_rows)}")
    print(f"  full path uses unsupported bridge:   {incremental_uses_unsupported}/{len(incremental_rows)}")
    print(f"  requires unsupported bridge:          {incremental_requires_unsupported}/{len(incremental_rows)}")
    print("\n=== Bridge-level diagnostics ===")
    for name, group in [
        ("reference-supported", [r for r in bridge_rows if r["reference_supported"] == 1]),
        ("not reference-supported", [r for r in bridge_rows if r["reference_supported"] == 0]),
    ]:
        if group:
            print(
                f"  {name}: n={len(group)} mean_gap={np.mean([r['euclidean_gap'] for r in group]):.2f} "
                f"mean_q={np.mean([r['mean_q'] for r in group]):.4f} "
                f"path-used={sum(r['used_by_any_path'] for r in group)}/{len(group)}"
            )

    output_csv = args.output_csv if args.output_csv.is_absolute() else root / args.output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [key for key in bridge_rows[0] if not key.startswith("_")] if bridge_rows else []
    if fieldnames:
        with output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in bridge_rows:
                writer.writerow({key: row[key] for key in fieldnames})
        print(f"Saved {len(bridge_rows)} bridge rows to {output_csv}")

    query_output_csv = (
        args.query_output_csv if args.query_output_csv.is_absolute()
        else root / args.query_output_csv
    )
    query_output_csv.parent.mkdir(parents=True, exist_ok=True)
    if query_rows:
        with query_output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(query_rows[0]))
            writer.writeheader()
            writer.writerows(query_rows)
        print(f"Saved {len(query_rows)} query rows to {query_output_csv}")

    # ---- M9 report ----
    print("\n=== M9: case-study path lengths (10378780_15, query index 2) ===")
    for image_id in image_ids:
        if image_id != "10378780_15":
            continue
        pf = pred_dir / f"{image_id}_prob.npy"
        sf = pred_dir / f"{image_id}_std.npy"
        prob = np.load(pf).astype(np.float32)
        std = np.load(sf).astype(np.float32)
        _, mp = resolve_image_paths(root, image_id)
        gt = load_mask(mp).astype(bool)
        q = conservative_confidence_lower_bound(prob, std, 1.0)
        traversable = build_traversability(prob, 0.5, use_largest_component=False)
        repaired, bridges, _ = repair_topology(traversable, q)
        drep = ndimage.distance_transform_edt(repaired).astype(np.float32)
        gt_comp = largest_connected_road(gt)
        for query in get_queries_for_image(queries, image_id):
            if int(query["query_index"]) != 2:
                continue
            start = (int(query["start_row"]), int(query["start_col"]))
            goal = (int(query["goal_row"]), int(query["goal_col"]))
            o = astar_search("oracle", gt_comp, np.zeros_like(prob), start, goal, max_expansions=MAX_EXP)
            from src.planning.costs import make_probability_cost
            s = astar_search("soft", traversable, np.zeros_like(prob), start, goal,
                             point_cost=make_probability_cost(prob), max_expansions=MAX_EXP)
            p = astar_search("prop", repaired, np.zeros_like(prob), start, goal,
                             point_cost=make_unified_risk_cost(q, drep, 2.0, 1.0, 5.0, 1.0), max_expansions=MAX_EXP)
            print(f"  query {query['query_index']} start={start} goal={goal}")
            print(f"    oracle  path_len={path_length(o.path):.1f}px  nodes={len(o.path)}  success={o.success}")
            print(f"    soft    path_len={path_length(s.path):.1f}px  nodes={len(s.path)}  success={s.success}")
            print(f"    prop    path_len={path_length(p.path):.1f}px  nodes={len(p.path)}  success={p.success}")


if __name__ == "__main__":
    main()
