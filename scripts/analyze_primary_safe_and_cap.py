"""Construct-validity diagnostics for the frozen primary planning experiment.

This script does not alter or rerun the primary 200k-budget experiment. It:
1) computes image-level paired inference for RouteConform@5%; and
2) evaluates planning-success sensitivity to 100k/200k/500k/unlimited A* caps.

For caps above 200k, only primary rows that stopped at the 200k cap are rerun.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage, stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import load_mask
from src.data.queries import load_queries
from src.data.splits import resolve_image_paths
from src.planning.costs import (
    build_traversability,
    conservative_confidence_lower_bound,
)
from src.planning.grid import astar_search
from src.planning.grid import largest_connected_road
from src.planning.topology_repair import repair_topology


SOFT = "Soft-probability A*"
PROPOSED = "Topology+Risk A*"
GT = "GT Oracle A*"
METHODS = (SOFT, PROPOSED, GT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        type=Path,
        default=ROOT / "artifacts/raw_metrics/planning_tau2.csv",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=ROOT / "artifacts/queries/queries.csv",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=ROOT / "artifacts/predictions/test",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "artifacts/raw_metrics",
    )
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=2025)
    return parser.parse_args()


def read_primary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row["method"] in METHODS]


def safe_indicator(row: dict[str, str]) -> float:
    if int(row["success"]) != 1:
        return 0.0
    try:
        off_road = float(row["off_road_ratio"])
    except (TypeError, ValueError):
        return 0.0
    return float(np.isfinite(off_road) and off_road <= 0.05)


def paired_safe_report(
    rows: list[dict[str, str]], bootstrap: int, seed: int
) -> dict[str, object]:
    by_method_image: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        if row["method"] in (SOFT, PROPOSED):
            by_method_image[row["method"]][row["image_id"]].append(
                safe_indicator(row)
            )

    image_ids = sorted(
        set(by_method_image[SOFT]).intersection(by_method_image[PROPOSED])
    )
    soft = np.asarray(
        [np.mean(by_method_image[SOFT][image_id]) for image_id in image_ids]
    )
    proposed = np.asarray(
        [np.mean(by_method_image[PROPOSED][image_id]) for image_id in image_ids]
    )
    diff = proposed - soft

    rng = np.random.default_rng(seed)
    samples = rng.choice(diff, size=(bootstrap, len(diff)), replace=True).mean(axis=1)
    ci_low, ci_high = np.quantile(samples, [0.025, 0.975])
    t_result = stats.ttest_rel(proposed, soft)
    try:
        w_result = stats.wilcoxon(proposed, soft, zero_method="wilcox")
        wilcoxon = {
            "statistic": float(w_result.statistic),
            "p_value": float(w_result.pvalue),
        }
    except ValueError:
        wilcoxon = {"statistic": 0.0, "p_value": 1.0}
    sd = float(np.std(diff, ddof=1))

    return {
        "unit": "image",
        "n_images": len(image_ids),
        "definition": "RouteConform@5%: success and reference-road off-road ratio <= 0.05",
        "soft_mean": float(np.mean(soft)),
        "proposed_mean": float(np.mean(proposed)),
        "paired_difference": float(np.mean(diff)),
        "bootstrap_seed": seed,
        "bootstrap_replicates": bootstrap,
        "bootstrap_95_ci": [float(ci_low), float(ci_high)],
        "paired_t": {
            "statistic": float(t_result.statistic),
            "p_value": float(t_result.pvalue),
        },
        "wilcoxon": wilcoxon,
        "cohen_dz": float(np.mean(diff) / sd) if sd > 0 else 0.0,
    }


def load_prediction(predictions: Path, image_id: str) -> tuple[np.ndarray, np.ndarray]:
    prob = np.load(predictions / f"{image_id}_prob.npy").astype(np.float32)
    std_path = predictions / f"{image_id}_std.npy"
    std = (
        np.load(std_path).astype(np.float32)
        if std_path.exists()
        else np.zeros_like(prob)
    )
    return prob, std


def build_method_inputs(
    predictions: Path, image_id: str
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    prob, std = load_prediction(predictions, image_id)
    original = build_traversability(
        prob, threshold=0.5, use_largest_component=False
    )
    q_map = conservative_confidence_lower_bound(
        prob, std, kappa=1.0, epsilon=1e-6
    )
    repaired, _bridges, _skeleton = repair_topology(
        original,
        q_map,
        min_radius=2,
        max_radius=50,
        tau_gate=2.0,
        w_p=1.0,
        w_l=0.5,
        w_theta=1.0,
        w_w=0.5,
        w_g=1.0,
        bridge_width=2,
    )
    distance = ndimage.distance_transform_edt(repaired)
    risk = np.minimum(-np.log(np.clip(q_map, 1e-9, 1.0)), 1.0)
    boundary = np.exp(-distance / 5.0)
    proposed_cost = 2.0 * risk + boundary
    soft_cost = -np.log(np.clip(prob, 1e-9, 1.0))
    _, gt_path = resolve_image_paths(ROOT, image_id)
    gt = largest_connected_road(load_mask(gt_path).astype(bool))
    return {
        SOFT: (original, soft_cost),
        PROPOSED: (repaired, proposed_cost),
        GT: (gt, np.zeros_like(prob, dtype=np.float32)),
    }


def cap_sensitivity(
    rows: list[dict[str, str]], predictions: Path, query_path: Path
) -> tuple[list[dict[str, object]], dict[str, object]]:
    query_list = load_queries(query_path)
    query_map = {
        (query["image_id"], str(query["query_index"])): query
        for query in query_list
    }
    primary: dict[tuple[str, str, str], dict[str, str]] = {}
    capped_by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (row["method"], row["image_id"], row["query_index"])
        primary[key] = row
        if int(row["success"]) == 0 and int(float(row["expanded_nodes"])) >= 200_000:
            capped_by_image[row["image_id"]].append(row)

    rerun: dict[tuple[str, str, str, str], tuple[bool, int]] = {}
    for image_id in sorted(capped_by_image):
        method_inputs = build_method_inputs(predictions, image_id)
        for row in capped_by_image[image_id]:
            method = row["method"]
            query = query_map[(image_id, row["query_index"])]
            start = (int(query["start_row"]), int(query["start_col"]))
            goal = (int(query["goal_row"]), int(query["goal_col"]))
            traversable, cost_map = method_inputs[method]
            for label, cap in (("500k", 500_000), ("unlimited", None)):
                result = astar_search(
                    method,
                    traversable,
                    np.zeros_like(cost_map),
                    start,
                    goal,
                    point_cost=lambda point, cm=cost_map: float(cm[point]),
                    max_expansions=cap,
                )
                rerun[(label, method, image_id, row["query_index"])] = (
                    result.success,
                    result.expanded_nodes,
                )

    output_rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "primary_cap": 200_000,
        "interpretation": (
            "Diagnostic only; the frozen primary results retain the 200k cap. "
            "Failure under a finite cap is planner non-completion, not proof of "
            "structural disconnection."
        ),
        "caps": {},
        "primary_cap_limited_failures": {
            method: sum(
                int(row["success"]) == 0
                and int(float(row["expanded_nodes"])) >= 200_000
                for row in rows
                if row["method"] == method
            )
            for method in METHODS
        },
    }

    for label, threshold in (
        ("100k", 100_000),
        ("200k", 200_000),
        ("500k", None),
        ("unlimited", None),
    ):
        cap_stats: dict[str, object] = {}
        for method in METHODS:
            method_rows = [row for row in rows if row["method"] == method]
            outcomes: list[bool] = []
            rerun_expanded: list[int] = []
            for row in method_rows:
                primary_success = int(row["success"]) == 1
                primary_expanded = int(float(row["expanded_nodes"]))
                if label == "100k":
                    success = primary_success and primary_expanded <= threshold
                elif label == "200k":
                    success = primary_success
                elif primary_success:
                    success = True
                elif primary_expanded >= 200_000:
                    success, expanded = rerun[
                        (label, method, row["image_id"], row["query_index"])
                    ]
                    rerun_expanded.append(expanded)
                else:
                    success = False
                outcomes.append(bool(success))
            success_count = int(np.sum(outcomes))
            rate = success_count / len(outcomes)
            cap_stats[method] = {
                "n": len(outcomes),
                "success_count": success_count,
                "success_rate": rate,
                "rerun_max_expanded": max(rerun_expanded) if rerun_expanded else None,
            }
            output_rows.append(
                {
                    "cap": label,
                    "method": method,
                    "n": len(outcomes),
                    "success_count": success_count,
                    "success_rate": rate,
                }
            )
        cap_stats["proposed_minus_soft"] = (
            cap_stats[PROPOSED]["success_rate"] - cap_stats[SOFT]["success_rate"]
        )
        summary["caps"][label] = cap_stats

    return output_rows, summary


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_primary(args.metrics)

    safe_report = paired_safe_report(rows, args.bootstrap, args.seed)
    safe_path = args.out_dir / "primary_safe_stats.json"
    safe_path.write_text(json.dumps(safe_report, indent=2), encoding="utf-8")
    print(json.dumps({"safe_stats": safe_report}, indent=2))

    cap_rows, cap_report = cap_sensitivity(rows, args.predictions, args.queries)
    cap_csv = args.out_dir / "search_cap_sensitivity.csv"
    with cap_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=cap_rows[0].keys())
        writer.writeheader()
        writer.writerows(cap_rows)
    cap_json = args.out_dir / "search_cap_sensitivity.json"
    cap_json.write_text(json.dumps(cap_report, indent=2), encoding="utf-8")
    print(json.dumps({"cap_sensitivity": cap_report}, indent=2))


if __name__ == "__main__":
    main()
