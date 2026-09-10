#!/usr/bin/env python
"""M5: planning evaluation under degradations (13 conditions).

For each corruption condition, runs the same 6-method planning comparison as
M4 on the corrupted-input ensemble predictions, reporting per-condition
success / off-road / detour.

The M5 hypothesis: on clean data the proposed method's off-road advantage
over URA/Clearance baselines is small (they all avoid false positives). Under
degradation the model becomes overconfident, the conservative lower bound
q = p - kappa*std should keep paths on road while baselines using raw p (or
entropy) degrade — so the proposed advantage widens.

Reuses the M4 evaluation core (evaluate_planning.run_methods) but with a
prediction dir per condition. Pure CPU (planning), fast.

Usage:
    python scripts/evaluate_corrupted_planning.py --conditions noise_heavy
    python scripts/evaluate_corrupted_planning.py   # all 13
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
    conservative_confidence_lower_bound,
    make_clearance_cost,
    make_probability_cost,
    make_uncertainty_cost,
    make_unified_risk_cost,
    normalized_entropy,
)
from src.planning.grid import SearchResult, astar_search, largest_connected_road, path_length
from src.planning.topology_repair import repair_topology
from src.data.corruptions import CORRUPTION_CONDITIONS

MAX_EXPANSIONS = 200000
LABEL_STRUCT = np.ones((3, 3), dtype=np.uint8)


def reachable(labels, start, goal) -> bool:
    c_s, c_g = labels[start], labels[goal]
    return c_s != 0 and c_s == c_g


def cached_costs(prob, distance, q, uncertainty, lambda_r, lambda_b, tau_d, risk_cap):
    """Build point-cost arrays once per image, not once per query."""
    eps = 1e-9
    p = np.clip(prob.astype(np.float32), eps, 1.0 - eps)
    qv = np.clip(q.astype(np.float32), eps, 1.0 - eps)
    dist = distance.astype(np.float32)
    return {
        "prob": -np.log(p),
        "clearance": -np.log(p) + lambda_b * np.exp(-dist / tau_d),
        "ura": -np.log(p) + 2.0 * uncertainty.astype(np.float32),
        "proposed": lambda_r * np.minimum(-np.log(qv), risk_cap)
                    + lambda_b * np.exp(-dist / tau_d),
    }


def run_methods(traversable, labels, prob, distance, q, uncertainty, gt, start, goal,
                oracle_len, methods, lambda_r=2.0, lambda_b=1.0, tau_d=5.0, risk_cap=1.0,
                cost_maps=None):
    """Run the A* methods that share a mask; unreachable ones fail fast."""
    rows = []
    for name, cost_kind in methods:
        if not reachable(labels, start, goal):
            r = SearchResult(name=name, path=[], planning_time_ms=0.0,
                             expanded_nodes=0, success=False)
        else:
            if cost_kind == "prob":
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
    parser = argparse.ArgumentParser(description="M5 corrupted planning.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--prediction-dir", type=Path,
                        default=Path("artifacts/predictions/corruptions/test"))
    parser.add_argument("--queries-csv", type=Path, default=Path("artifacts/queries/queries.csv"))
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--tau-gate", type=float, default=2.0)
    parser.add_argument("--lambda-r", type=float, default=2.0)
    parser.add_argument("--lambda-b", type=float, default=1.0)
    parser.add_argument("--tau-d", type=float, default=5.0)
    parser.add_argument("--risk-cap", type=float, default=1.0)
    parser.add_argument("--queries-per-image", type=int, default=None,
                        help="cap queries per image (speeds up degraded planning)")
    parser.add_argument("--max-expansions", type=int, default=200000,
                        help="A* expansion cap (reduce for degraded planning)")
    args = parser.parse_args()

    # Allow overriding the module-level MAX_EXPANSIONS used by run_methods
    global MAX_EXPANSIONS
    MAX_EXPANSIONS = args.max_expansions

    root = args.root
    splits = build_splits(root)
    image_ids = splits["test_ids"]
    queries = load_queries(root / args.queries_csv)
    conditions = args.conditions or list(CORRUPTION_CONDITIONS.keys())

    print(f"{'condition':<16} {'proposed_succ':>13} {'softprob_succ':>13} "
          f"{'proposed_off':>12} {'softprob_off':>11} {'proposed_det':>12} "
          f"{'softprob_det':>11} {'n':>4}")
    print("-" * 100)

    summary_rows = []
    for cond in conditions:
        pred_dir = root / args.prediction_dir / cond
        if not pred_dir.exists():
            print(f"  {cond}: MISSING, skip")
            continue

        agg = {"succ_p": [], "succ_s": [], "off_p": [], "off_s": [], "det_p": [], "det_s": [], "n": 0}
        for img_idx, image_id in enumerate(image_ids):
            prob_file = pred_dir / f"{image_id}_prob.npy"
            if not prob_file.exists():
                continue
            prob = np.load(prob_file).astype(np.float32)
            std_file = pred_dir / f"{image_id}_std.npy"
            std = np.load(std_file).astype(np.float32) if std_file.exists() else np.zeros_like(prob)
            _, mask_path = resolve_image_paths(root, image_id)
            gt = load_mask(mask_path).astype(bool)

            q = conservative_confidence_lower_bound(prob, std, args.kappa)
            traversable = build_traversability(prob, 0.5, use_largest_component=False)
            distance = ndimage.distance_transform_edt(traversable).astype(np.float32)
            repaired, _b, _s = repair_topology(traversable, q, tau_gate=args.tau_gate)
            repaired_distance = ndimage.distance_transform_edt(repaired).astype(np.float32)
            entropy = normalized_entropy(prob).astype(np.float32)
            cost_maps_orig = cached_costs(
                prob, distance, q, entropy,
                args.lambda_r, args.lambda_b, args.tau_d, args.risk_cap,
            )
            cost_maps_rep = cached_costs(
                prob, repaired_distance, q, entropy,
                args.lambda_r, args.lambda_b, args.tau_d, args.risk_cap,
            )
            labels_orig, _ = ndimage.label(traversable, structure=np.ones((3, 3), np.uint8))
            labels_rep, _ = ndimage.label(repaired, structure=np.ones((3, 3), np.uint8))

            gt_comp = largest_connected_road(gt)
            gt_labels, _ = ndimage.label(gt_comp, structure=np.ones((3, 3), np.uint8))

            base = [
                ("Soft-probability A*", "prob"),
            ]
            img_queries = get_queries_for_image(queries, image_id)
            if args.queries_per_image:
                img_queries = img_queries[: args.queries_per_image]
            for query in img_queries:
                start = (int(query["start_row"]), int(query["start_col"]))
                goal = (int(query["goal_row"]), int(query["goal_col"]))
                agg["n"] += 1

                if reachable(gt_labels, start, goal):
                    oracle = astar_search("oracle", gt_comp, np.zeros_like(prob), start, goal,
                                          max_expansions=MAX_EXPANSIONS)
                    oracle_len = path_length(oracle.path) if oracle.success else None
                else:
                    oracle = SearchResult("oracle", [], 0.0, 0, False)
                    oracle_len = None

                rows = run_methods(traversable, labels_orig, prob, distance, q, entropy, gt,
                                   start, goal, oracle_len, base,
                                   args.lambda_r, args.lambda_b, args.tau_d, args.risk_cap,
                                   cost_maps_orig)
                rows.extend(run_methods(repaired, labels_rep, prob, repaired_distance, q, entropy, gt,
                                        start, goal, oracle_len,
                                        [("Topology+Risk A*", "proposed")],
                                        args.lambda_r, args.lambda_b, args.tau_d, args.risk_cap,
                                        cost_maps_rep))
                for r in rows:
                    if r["method"] == "Topology+Risk A*":
                        agg["succ_p"].append(r["success"])
                        if not np.isnan(r["off_road_ratio"]):
                            agg["off_p"].append(r["off_road_ratio"])
                        if not np.isnan(r["relative_detour"]):
                            agg["det_p"].append(r["relative_detour"])
                    elif r["method"] == "Soft-probability A*":
                        agg["succ_s"].append(r["success"])
                        if not np.isnan(r["off_road_ratio"]):
                            agg["off_s"].append(r["off_road_ratio"])
                        if not np.isnan(r["relative_detour"]):
                            agg["det_s"].append(r["relative_detour"])

        if agg["n"] == 0:
            print(f"  {cond}: no predictions")
            continue
        row = {
            "condition": cond, "n": agg["n"],
            "proposed_succ": float(np.mean(agg["succ_p"])),
            "softprob_succ": float(np.mean(agg["succ_s"])),
            "proposed_off": float(np.mean(agg["off_p"])) if agg["off_p"] else float("nan"),
            "softprob_off": float(np.mean(agg["off_s"])) if agg["off_s"] else float("nan"),
            "proposed_det": float(np.mean(agg["det_p"])) if agg["det_p"] else float("nan"),
            "softprob_det": float(np.mean(agg["det_s"])) if agg["det_s"] else float("nan"),
        }
        summary_rows.append(row)
        print(f"{cond:<16} {row['proposed_succ']:13.3f} {row['softprob_succ']:13.3f} "
              f"{row['proposed_off']:12.4f} {row['softprob_off']:11.4f} "
              f"{row['proposed_det']:12.4f} {row['softprob_det']:11.4f} {agg['n']:4d}",
              flush=True)

        # Write incrementally so a long run never loses prior conditions
        _partial = list(summary_rows)
        with (root / "artifacts/raw_metrics/corrupted_planning.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(_partial[0].keys()))
            w.writeheader(); w.writerows(_partial)

    out = root / "artifacts" / "raw_metrics"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "corrupted_planning.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)
    print(f"\nSaved to {out / 'corrupted_planning.csv'}")


if __name__ == "__main__":
    main()
