#!/usr/bin/env python
"""M4 path planning evaluation: all 6 methods on fixed queries, TEST split.

Optimized: per-image precomputation (traversable, repaired mask, distance,
connected-component labels) + reachability pre-check so unreachable queries
are recorded as failures WITHOUT an expensive full-grid A* search.

Methods:
  1. GT Oracle A* (upper bound)
  2. Binary-mask A*
  3. Soft-probability A*
  4. Clearance-aware A*
  5. Entropy-penalized A* (internal control; not a URA* reproduction)
  6. Topology+Risk A* (proposed: repaired map + uncertainty-discounted q)

Metrics per query: success, off-road ratio, relative detour (vs GT oracle).

Usage:
    python scripts/evaluate_planning.py --split test --kappa 1.0
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
from src.planning.costs import (
    build_traversability,
    uncertainty_discounted_score,
    make_clearance_cost,
    make_probability_cost,
    make_uncertainty_cost,
    make_unified_risk_cost,
    normalized_entropy,
)
from src.planning.grid import astar_search, largest_connected_road, path_length
from src.planning.topology_repair import repair_topology, repair_topology_geometric

MAX_EXPANSIONS = 200000  # safety valve against pathological full-grid searches
LABEL_STRUCT = np.ones((3, 3), dtype=np.uint8)


def _cached_costs(prob, distance, q, uncertainty, lambda_r, lambda_b, tau_d,
                  risk_cap, lambda_u=2.0):
    """Build point-cost arrays once per image instead of once per query."""
    eps = 1e-9
    p = np.clip(prob.astype(np.float32), eps, 1.0 - eps)
    qv = np.clip(q.astype(np.float32), eps, 1.0 - eps)
    dist = distance.astype(np.float32)
    return {
        "prob": -np.log(p),
        "clearance": -np.log(p) + lambda_b * np.exp(-dist / tau_d),
        "ura": -np.log(p) + lambda_u * uncertainty.astype(np.float32),
        "proposed": lambda_r * np.minimum(-np.log(qv), risk_cap) + lambda_b * np.exp(-dist / tau_d),
    }


def reachable(labels: np.ndarray, start, goal) -> bool:
    """True if start and goal lie in the same connected component."""
    c_s, c_g = labels[start], labels[goal]
    return c_s != 0 and c_s == c_g


def run_methods(traversable, labels, prob, distance, q, uncertainty, gt, start, goal,
                oracle_len, methods: list[tuple[str, str]],
                lambda_r=2.0, lambda_b=1.0, tau_d=5.0, risk_cap=1.0,
                cost_maps=None):
    """Run the A* methods that share a mask; unreachable ones fail fast.

    lambda_r / lambda_b / tau_d are threaded through so that command-line
    overrides actually reach the cost functions (they were dead code before).
    """
    rows = []
    for name, cost_kind in methods:
        if not reachable(labels, start, goal):
            from src.planning.grid import SearchResult
            r = SearchResult(name=name, path=[], planning_time_ms=0.0,
                             expanded_nodes=0, success=False)
        else:
            if cost_kind == "binary":
                r = astar_search(name, traversable, np.zeros_like(prob), start, goal,
                                 max_expansions=MAX_EXPANSIONS)
            elif cost_kind == "prob":
                r = astar_search(name, traversable, np.zeros_like(prob), start, goal,
                                 point_cost=(lambda p: float(cost_maps["prob"][p])) if cost_maps is not None else make_probability_cost(prob),
                                 max_expansions=MAX_EXPANSIONS)
            elif cost_kind == "clearance":
                r = astar_search(name, traversable, np.zeros_like(prob), start, goal,
                                 point_cost=(lambda p: float(cost_maps["clearance"][p])) if cost_maps is not None else make_clearance_cost(prob, distance, lambda_b, tau_d),
                                 max_expansions=MAX_EXPANSIONS)
            elif cost_kind == "ura":
                r = astar_search(name, traversable, np.zeros_like(prob), start, goal,
                                 point_cost=(lambda p: float(cost_maps["ura"][p])) if cost_maps is not None else make_uncertainty_cost(prob, uncertainty),
                                 max_expansions=MAX_EXPANSIONS)
            elif cost_kind == "proposed":
                r = astar_search(name, traversable, np.zeros_like(prob), start, goal,
                                 point_cost=(lambda p: float(cost_maps["proposed"][p])) if cost_maps is not None else make_unified_risk_cost(q, distance, lambda_r, lambda_b, tau_d, risk_cap),
                                 max_expansions=MAX_EXPANSIONS)
            else:
                raise ValueError(cost_kind)
        pm = planning_metrics(r, gt, distance=None, risk=None, oracle_length=oracle_len)
        rows.append(pm)
    return rows


def main():
    parser = argparse.ArgumentParser(description="M4 planning evaluation.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--prediction-dir", type=Path, default=Path("artifacts/predictions/test"))
    parser.add_argument("--queries-csv", type=Path, default=Path("artifacts/queries/queries.csv"))
    parser.add_argument("--dataset", choices=["massachusetts", "deepglobe"], default="massachusetts")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--split", default="test")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--tau-gate", type=float, default=2.0)
    parser.add_argument(
        "--include-geometric-repair",
        action="store_true",
        help=("evaluate the confidence-free geometric endpoint-reconnection "
              "baseline; tune --geometric-tau-gate on calibration before use on test"),
    )
    parser.add_argument(
        "--geometric-tau-gate",
        type=float,
        default=1.0,
        help="exploratory geometric-repair gate; not a test-selected operating point",
    )
    parser.add_argument("--lambda-r", type=float, default=2.0)
    parser.add_argument("--lambda-b", type=float, default=1.0)
    parser.add_argument(
        "--lambda-u", type=float, default=2.0,
        help="entropy-penalty weight for the internal entropy control",
    )
    parser.add_argument("--tau-d", type=float, default=5.0)
    parser.add_argument("--risk-cap", type=float, default=1.0,
                        help="cap on the -log(q) uncertainty penalty per pixel")
    parser.add_argument("--limit", type=int, default=None, help="only first N images (debug)")
    parser.add_argument("--skip", type=int, default=0, help="skip first N images (sharding)")
    parser.add_argument("--output-csv", type=Path, default=Path("artifacts/raw_metrics/planning.csv"))
    args = parser.parse_args()

    if args.dataset == "deepglobe":
        from src.data.deepglobe import build_deepglobe_splits, resolve_deepglobe_paths
        splits = build_deepglobe_splits(args.root, seed=args.split_seed)
        image_ids = splits["test_ids"]
        if args.skip:
            image_ids = image_ids[args.skip:]
        if args.limit:
            image_ids = image_ids[: args.limit]
        pred_dir = args.root / args.prediction_dir
        queries = load_queries(args.root / args.queries_csv)
        resolve = resolve_deepglobe_paths
        print(f"Split 'test' (DeepGlobe, seed={args.split_seed}): "
              f"{len(image_ids)} images (skip={args.skip}), {len(queries)} queries")
    else:
        splits = build_splits(args.root)
        image_ids = splits[f"{args.split}_ids"]
        if args.skip:
            image_ids = image_ids[args.skip:]
        if args.limit:
            image_ids = image_ids[: args.limit]
        pred_dir = args.root / args.prediction_dir
        queries = load_queries(args.root / args.queries_csv)
        resolve = resolve_image_paths
        print(f"Split '{args.split}': {len(image_ids)} images (skip={args.skip}), "
              f"{len(queries)} queries")

    all_plans: list[dict] = []
    skipped_imgs = 0

    for img_idx, image_id in enumerate(image_ids):
        prob_file = pred_dir / f"{image_id}_prob.npy"
        std_file = pred_dir / f"{image_id}_std.npy"
        if not prob_file.exists():
            skipped_imgs += 1
            continue
        prob = np.load(prob_file).astype(np.float32)
        std = np.load(std_file).astype(np.float32) if std_file.exists() else np.zeros_like(prob)
        _, mask_path = resolve(args.root, image_id)
        gt = load_mask(mask_path).astype(bool)

        q = uncertainty_discounted_score(prob, std, args.kappa)
        traversable = build_traversability(prob, args.threshold, use_largest_component=False)
        distance = ndimage.distance_transform_edt(traversable).astype(np.float32)
        repaired, _bridges, _skel = repair_topology(traversable, q, tau_gate=args.tau_gate)
        geometric_repaired = None
        geometric_labels = None
        geometric_distance = None
        geometric_cost_maps = None
        if args.include_geometric_repair:
            geometric_repaired, _geometric_bridges, _ = repair_topology_geometric(
                traversable, tau_gate=args.geometric_tau_gate
            )
            geometric_labels, _ = ndimage.label(
                geometric_repaired, structure=LABEL_STRUCT
            )
            geometric_distance = ndimage.distance_transform_edt(geometric_repaired).astype(np.float32)
        # Distance for the proposed method must be recomputed on the REPAIRED
        # mask: bridge pixels are non-traversable in the original EDT, giving
        # distance=0 and a maximum boundary penalty that punishes the method
        # for its own repair work.
        repaired_distance = ndimage.distance_transform_edt(repaired).astype(np.float32)
        # The entropy control uses normalized predictive entropy (in [0,1]),
        # not raw epistemic std (which is bounded by ~0.5 and gets dominated by
        # -log(p), reducing the baseline to a Soft-probability duplicate).
        entropy = normalized_entropy(prob).astype(np.float32)
        labels_orig, _ = ndimage.label(traversable, structure=LABEL_STRUCT)
        labels_rep, _ = ndimage.label(repaired, structure=LABEL_STRUCT)

        cost_maps_orig = _cached_costs(
            prob, distance, q, entropy, args.lambda_r, args.lambda_b, args.tau_d,
            args.risk_cap, args.lambda_u
        )
        cost_maps_rep = _cached_costs(
            prob, repaired_distance, q, entropy,
            args.lambda_r, args.lambda_b, args.tau_d, args.risk_cap, args.lambda_u,
        )
        if geometric_repaired is not None:
            geometric_cost_maps = _cached_costs(
                prob, geometric_distance, q, entropy,
                args.lambda_r, args.lambda_b, args.tau_d, args.risk_cap, args.lambda_u,
            )

        gt_comp = largest_connected_road(gt)
        gt_labels, _ = ndimage.label(gt_comp, structure=LABEL_STRUCT)

        base_methods = [
            ("Binary-mask A*", "binary"),
            ("Soft-probability A*", "prob"),
            ("Clearance-aware A*", "clearance"),
            ("Entropy-penalized A*", "ura"),
        ]

        for query in get_queries_for_image(queries, image_id):
            start = (int(query["start_row"]), int(query["start_col"]))
            goal = (int(query["goal_row"]), int(query["goal_col"]))

            # GT Oracle (upper bound), computed once when reachable
            from src.planning.grid import SearchResult
            if reachable(gt_labels, start, goal):
                oracle = astar_search("GT Oracle A*", gt_comp, np.zeros_like(prob),
                                      start, goal, max_expansions=MAX_EXPANSIONS)
                oracle_len = path_length(oracle.path) if oracle.success else None
            else:
                oracle = SearchResult("GT Oracle A*", [], 0.0, 0, False)
                oracle_len = None

            rows = run_methods(traversable, labels_orig, prob, distance, q, entropy, gt,
                               start, goal, oracle_len, base_methods,
                               args.lambda_r, args.lambda_b, args.tau_d, args.risk_cap,
                               cost_maps_orig)
            pm = planning_metrics(oracle, gt, oracle_length=oracle_len)
            rows.append(pm)
            # Proposed method (repaired mask + unified risk): search on the
            # repaired mask and use the repaired-mask distance transform.
            rows.extend(run_methods(repaired, labels_rep, prob, repaired_distance, q, entropy, gt,
                                    start, goal, oracle_len,
                                    [("Topology+Risk A*", "proposed")],
                                    args.lambda_r, args.lambda_b, args.tau_d, args.risk_cap,
                                    cost_maps_rep))
            if geometric_repaired is not None:
                rows.extend(run_methods(
                    geometric_repaired, geometric_labels, prob, geometric_distance,
                    q, entropy, gt, start, goal, oracle_len,
                    [("Geometric endpoint repair + Risk A*", "proposed")],
                    args.lambda_r, args.lambda_b, args.tau_d, args.risk_cap,
                    geometric_cost_maps,
                ))

            for r in rows:
                r["query_index"] = query["query_index"]
                r["image_id"] = image_id
                all_plans.append(r)

        if (img_idx + 1) % 5 == 0:
            print(f"  [{img_idx+1}/{len(image_ids)}] {image_id} ({len(all_plans)} rows)", flush=True)

    if not all_plans:
        print("No planning results.")
        return

    methods = sorted({p["method"] for p in all_plans})
    print(f"\n=== Planning on '{args.split}' (n={len(all_plans)} rows, skipped_imgs={skipped_imgs}) ===")
    print(f"{'method':<26} {'success':>7} {'off_road':>9} {'detour_mean':>11} {'detour_med':>10} {'detour_p90':>10} {'len':>7}")
    summary = {}
    detour_vals: dict[str, list[float]] = {}
    for method in methods:
        rows = [p for p in all_plans if p["method"] == method]
        success = float(np.mean([r["success"] for r in rows]))
        off = [r["off_road_ratio"] for r in rows if not np.isnan(r["off_road_ratio"])]
        det = [r["relative_detour"] for r in rows if not np.isnan(r["relative_detour"])]
        ln = [r["path_length"] for r in rows if not np.isnan(r["path_length"])]
        off_road = float(np.mean(off)) if off else float("nan")
        detour = float(np.mean(det)) if det else float("nan")
        det_med = float(np.median(det)) if det else float("nan")
        det_p90 = float(np.percentile(det, 90)) if det else float("nan")
        length = float(np.mean(ln)) if ln else float("nan")
        print(f"{method:<26} {success:7.3f} {off_road:9.4f} {detour:11.4f} {det_med:10.4f} {det_p90:10.3f} {length:7.1f}")
        summary[method] = {"success": success, "off_road": off_road, "detour": detour}
        detour_vals[method] = det

    proposed = summary.get("Topology+Risk A*")
    if proposed:
        for b in ["Binary-mask A*", "Soft-probability A*", "Entropy-penalized A*"]:
            if b in summary:
                s_rel = (proposed["success"] - summary[b]["success"]) / max(summary[b]["success"], 1e-9)
                o_rel = (summary[b]["off_road"] - proposed["off_road"]) / max(summary[b]["off_road"], 1e-9)
                print(f"\nProposed vs {b}: success {s_rel*100:+.1f}%  off_road {o_rel*100:+.1f}%")

        # Common-success detour: detour is only meaningful when BOTH methods
        # succeed AND the oracle succeeds. The proposed method uniquely
        # succeeds on many queries (topology repair extends reachability where
        # every baseline fails); on those queries its path is much longer than
        # the GT oracle's, which inflates the mean detour. Comparing detour on
        # the common-success set isolates routing quality from reachability.
        print("\n--- Detour on common-success queries (both methods + oracle succeed) ---")
        proposed_rows = {int(r["query_index"]): r for r in all_plans if r["method"] == "Topology+Risk A*"}
        for b in ["Binary-mask A*", "Soft-probability A*", "Entropy-penalized A*", "Clearance-aware A*"]:
            b_rows = {int(r["query_index"]): r for r in all_plans if r["method"] == b}
            common = [
                (proposed_rows[k]["relative_detour"], b_rows[k]["relative_detour"])
                for k in proposed_rows.keys() & b_rows.keys()
                if proposed_rows[k]["success"] == 1 and b_rows[k]["success"] == 1
                and not np.isnan(proposed_rows[k]["relative_detour"])
                and not np.isnan(b_rows[k]["relative_detour"])
            ]
            if common:
                p_det = np.mean([c[0] for c in common])
                b_det = np.mean([c[1] for c in common])
                n = len(common)
                print(f"  {b:<26} n={n:4d} proposed_detour={p_det:.4f} vs {b}_detour={b_det:.4f}  (Δ {p_det-b_det:+.4f})")

    out_csv = args.output_csv if args.output_csv.is_absolute() else args.root / args.output_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_plans[0].keys()))
        writer.writeheader()
        writer.writerows(all_plans)
    print(f"\nSaved {len(all_plans)} rows to {out_csv}")


if __name__ == "__main__":
    main()
