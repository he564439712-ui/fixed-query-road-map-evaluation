#!/usr/bin/env python
"""MRA-GATE-016: audit metric-to-route ranking alignment on calibration data.

The Gate evaluates one frozen road extractor under five fixed image conditions.
It asks whether image-level rankings by Dice, clDice, or raster APLS agree with
rankings by safe downstream route availability.  It is calibration-only and
uses the unmodified thresholded road map with a fixed probability-cost A*.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage, stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import load_mask
from src.data.queries import generate_queries_for_image
from src.data.splits import build_splits, resolve_image_paths
from src.eval.segmentation import segmentation_metrics
from src.planning.costs import build_traversability
from src.planning.grid import astar_search


LABEL_STRUCTURE = np.ones((3, 3), dtype=np.uint8)
FROZEN_CONDITIONS = ("clean", "blur_medium", "shadow_medium", "noise_medium", "bc_medium")
FROZEN_METRICS = ("dice", "cldice", "apls")


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def _connected(labels: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> bool:
    label = int(labels[start])
    return label != 0 and label == int(labels[goal])


def _route_utility(probability: np.ndarray, ground_truth: np.ndarray, queries: list[dict]) -> dict[str, float]:
    """Fixed non-repair planner and its safe-route utility for one image."""
    traversable = build_traversability(probability, threshold=0.5, use_largest_component=False)
    labels, _ = ndimage.label(traversable, structure=LABEL_STRUCTURE)
    point_cost = -np.log(np.clip(probability, 1e-6, 1.0)).astype(np.float32)
    successes: list[int] = []
    safe: list[int] = []
    off_road: list[float] = []
    for query in queries:
        start = (int(query["start_row"]), int(query["start_col"]))
        goal = (int(query["goal_row"]), int(query["goal_col"]))
        if not _connected(labels, start, goal):
            successes.append(0)
            safe.append(0)
            continue
        result = astar_search(
            "fixed-probability-astar", traversable, np.zeros_like(probability), start, goal,
            point_cost=lambda point, cost=point_cost: float(cost[point]), max_expansions=200_000,
        )
        if not result.success:
            successes.append(0)
            safe.append(0)
            continue
        path_rows = np.asarray([point[0] for point in result.path], dtype=np.int32)
        path_cols = np.asarray([point[1] for point in result.path], dtype=np.int32)
        off = float(1.0 - np.mean(ground_truth[path_rows, path_cols]))
        successes.append(1)
        safe.append(int(off <= 0.05))
        off_road.append(off)
    return {
        "route_success": float(np.mean(successes)) if successes else float("nan"),
        "safe_route_success": float(np.mean(safe)) if safe else float("nan"),
        "mean_off_road_successes": float(np.mean(off_road)) if off_road else float("nan"),
        "queries": int(len(queries)),
    }


def _bootstrap_median(values: list[float], repeats: int, seed: int) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    data = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.asarray([np.median(data[rng.integers(0, len(data), len(data))]) for _ in range(repeats)])
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _alignment(rows: list[dict[str, object]], conditions: tuple[str, ...], repeats: int) -> dict[str, object]:
    by_image: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        by_image[str(row["image_id"])][str(row["condition"])] = row
    expected_pairs = len(conditions) * (len(conditions) - 1) // 2
    metric_report: dict[str, object] = {}
    for metric_index, metric in enumerate(FROZEN_METRICS):
        tau_values: list[float] = []
        reversals_by_image: dict[str, int] = {}
        informative_images: list[str] = []
        for image_id in sorted(by_image):
            image_rows = by_image[image_id]
            if set(image_rows) != set(conditions):
                raise RuntimeError(f"Missing condition for {image_id}")
            utility = np.asarray([float(image_rows[condition]["safe_route_success"]) for condition in conditions])
            scores = np.asarray([float(image_rows[condition][metric]) for condition in conditions])
            n_queries = int(image_rows[conditions[0]]["queries"])
            if len(np.unique(utility)) > 1:
                informative_images.append(image_id)
                tau = stats.kendalltau(scores, utility, variant="b").statistic
                if np.isfinite(tau):
                    tau_values.append(float(tau))
            reversals = 0
            for left in range(len(conditions)):
                for right in range(left + 1, len(conditions)):
                    score_delta = scores[left] - scores[right]
                    utility_delta = utility[left] - utility[right]
                    # A reversal requires the preferred product to lose at least
                    # two of that image's fixed route queries.
                    if score_delta * utility_delta < 0 and abs(utility_delta) >= 2.0 / n_queries:
                        reversals += 1
            reversals_by_image[image_id] = reversals
        ci_low, ci_high = _bootstrap_median(tau_values, repeats, 2025 + metric_index)
        metric_report[metric] = {
            "informative_images": len(informative_images),
            "tau_images_with_finite_metric_ranking": len(tau_values),
            "median_tau_b": float(np.median(tau_values)) if tau_values else float("nan"),
            "image_bootstrap_95_ci": [ci_low, ci_high],
            "images_with_any_rank_reversal": int(sum(value > 0 for value in reversals_by_image.values())),
            "reversal_image_fraction": float(sum(value > 0 for value in reversals_by_image.values()) / max(len(by_image), 1)),
            "pairwise_reversals": int(sum(reversals_by_image.values())),
            "pairwise_comparisons": int(len(by_image) * expected_pairs),
            "per_image_reversals": reversals_by_image,
        }
    route_informative = sum(
        len({float(row["safe_route_success"]) for row in image_rows.values()}) > 1
        for image_rows in by_image.values()
    )
    return {
        "images": len(by_image),
        "route_informative_images": int(route_informative),
        "conditions": list(conditions),
        "metrics": metric_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--split", choices=["calibration"], default="calibration")
    parser.add_argument("--prediction-dir", type=Path, default=Path("artifacts/predictions/mra_gate_016/calibration"))
    parser.add_argument("--conditions", nargs="+", default=list(FROZEN_CONDITIONS))
    parser.add_argument("--max-images", type=int, default=20)
    parser.add_argument("--queries-per-image", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--bootstrap-repeats", type=int, default=10_000)
    parser.add_argument("--output", type=Path, default=Path("artifacts/raw_metrics/metric_route_alignment_gate.json"))
    parser.add_argument("--per-image-output", type=Path, default=Path("artifacts/raw_metrics/metric_route_alignment_gate_per_image.csv"))
    args = parser.parse_args()
    conditions = tuple(args.conditions)
    if conditions != FROZEN_CONDITIONS:
        raise ValueError(f"MRA-GATE-016 freezes conditions at {FROZEN_CONDITIONS}")
    if args.max_images not in (3, 20):
        raise ValueError("MRA-GATE-016 permits only max-images=3 smoke or 20 full Gate")
    if args.queries_per_image != 12:
        raise ValueError("MRA-GATE-016 freezes 12 requested queries/image")
    root = args.root.resolve()
    prediction_dir = args.prediction_dir if args.prediction_dir.is_absolute() else root / args.prediction_dir
    output = args.output if args.output.is_absolute() else root / args.output
    per_image_output = args.per_image_output if args.per_image_output.is_absolute() else root / args.per_image_output
    output.parent.mkdir(parents=True, exist_ok=True)
    per_image_output.parent.mkdir(parents=True, exist_ok=True)
    image_ids = sorted(build_splits(root)["calibration_ids"])[:args.max_images]
    if len(image_ids) != args.max_images:
        raise RuntimeError("Insufficient calibration images")

    started = time.perf_counter()
    rows: list[dict[str, object]] = []
    for image_index, image_id in enumerate(image_ids):
        _, label_path = resolve_image_paths(root, image_id)
        ground_truth = load_mask(label_path).astype(bool)
        queries = generate_queries_for_image(
            ground_truth, num_queries=args.queries_per_image, seed=args.seed + image_index * 1000
        )
        if len(queries) < 2:
            raise RuntimeError(f"Too few deterministic queries for {image_id}: {len(queries)}")
        for condition in conditions:
            probability_path = prediction_dir / condition / f"{image_id}_prob.npy"
            if not probability_path.exists():
                raise FileNotFoundError(f"Missing prediction: {probability_path}")
            probability = np.load(probability_path).astype(np.float32)
            metrics = segmentation_metrics(probability, ground_truth, threshold=0.5)
            route = _route_utility(probability, ground_truth, queries)
            rows.append({"image_id": image_id, "condition": condition, **metrics, **route})
        print(f"  evaluated {image_index + 1}/{len(image_ids)}: {image_id}", flush=True)

    report = _alignment(rows, conditions, args.bootstrap_repeats)
    payload: dict[str, object] = {
        "experiment_id": "MRA-GATE-016",
        "split": "calibration",
        "locked_test_split_used": False,
        "protocol": {
            "image_ids": image_ids,
            "conditions": list(conditions),
            "requested_queries_per_image": args.queries_per_image,
            "query_seed": args.seed,
            "threshold": 0.5,
            "planner": "fixed probability-cost A* on original thresholded map; no repair or learned selector",
            "safe_route_success": "A* success with exact-GT centerline off-road ratio <=0.05",
            "rank_reversal": "metric-preferred condition has safe-route success lower by at least two image queries",
        },
        "wall_seconds": time.perf_counter() - started,
        "per_image_alignment": report,
    }
    if args.max_images == 3:
        payload["status"] = "smoke-complete-no-decision"
    else:
        dice = report["metrics"]["dice"]
        cldice = report["metrics"]["cldice"]
        decision = "GO" if (
            report["route_informative_images"] >= 15
            and dice["median_tau_b"] < 0.70 and dice["image_bootstrap_95_ci"][1] < 0.80
            and cldice["median_tau_b"] < 0.70 and cldice["image_bootstrap_95_ci"][1] < 0.80
            and cldice["reversal_image_fraction"] >= 0.25
        ) else "NO-GO"
        payload.update({
            "status": decision.lower(),
            "decision": decision,
            "go_criteria": {
                "route_informative_images_at_least": 15,
                "dice_and_cldice_median_tau_b_below": 0.70,
                "dice_and_cldice_bootstrap_upper_below": 0.80,
                "cldice_reversal_image_fraction_at_least": 0.25,
            },
        })
    fieldnames = sorted({key for row in rows for key in row})
    with per_image_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default))
    print(f"Saved per-image rows to {per_image_output}")


if __name__ == "__main__":
    main()
