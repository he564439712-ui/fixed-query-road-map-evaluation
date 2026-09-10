#!/usr/bin/env python
"""Calibration-only OAT sensitivity and URA baseline calibration.

The test split is never opened.  The proposed pipeline is evaluated one factor
at a time around the frozen nominal configuration, while the URA-style entropy
weight is selected on the same independent calibration images.  The output is
intended to document robustness and a fair baseline-tuning protocol, not to
replace the primary frozen test evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import load_mask
from src.data.queries import generate_queries_for_image
from src.data.splits import build_splits, resolve_image_paths
from src.planning.costs import (
    build_traversability,
    conservative_confidence_lower_bound,
    normalized_entropy,
)
from src.planning.grid import astar_search
from src.planning.topology_repair import repair_topology


MAX_EXPANSIONS = 200_000
LABEL_STRUCT = np.ones((3, 3), dtype=np.uint8)
NOMINAL = {
    "kappa": 1.0,
    "lambda_r": 2.0,
    "lambda_b": 1.0,
    "tau_d": 5.0,
    "risk_cap": 1.0,
    "tau_gate": 2.0,
}
SPECS = {
    "kappa": [0.0, 0.5, 1.0, 1.5, 2.0],
    "lambda_r": [0.5, 1.0, 2.0, 3.0, 4.0],
    "lambda_b": [0.25, 0.5, 1.0, 1.5, 2.0],
    "tau_d": [2.0, 3.5, 5.0, 7.5, 10.0],
    "risk_cap": [0.5, 0.75, 1.0, 1.25, 1.5],
}
URA_LAMBDAS = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0]


def reachable(labels: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> bool:
    return labels[start] != 0 and labels[start] == labels[goal]


def load_cache(root: Path, prediction_dir: Path, image_ids: list[str],
               queries_per_image: int, seed: int) -> list[dict]:
    cache: list[dict] = []
    for index, image_id in enumerate(image_ids):
        prob = np.load(prediction_dir / f"{image_id}_prob.npy").astype(np.float32)
        std_path = prediction_dir / f"{image_id}_std.npy"
        std = np.load(std_path).astype(np.float32) if std_path.exists() else np.zeros_like(prob)
        _, mask_path = resolve_image_paths(root, image_id)
        gt = load_mask(mask_path).astype(bool)
        traversable = build_traversability(prob, 0.5, use_largest_component=False)
        labels, _ = ndimage.label(traversable, structure=LABEL_STRUCT)
        queries = generate_queries_for_image(
            gt, num_queries=queries_per_image, seed=seed + index * 1000
        )
        cache.append({
            "id": image_id,
            "prob": prob,
            "std": std,
            "gt": gt,
            "traversable": traversable,
            "labels": labels,
            "entropy": normalized_entropy(prob).astype(np.float32),
            "queries": queries,
            "repair": {},
        })
    return cache


def repaired_for(item: dict, config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    key = (float(config["kappa"]), float(config["tau_gate"]))
    if key not in item["repair"]:
        q = conservative_confidence_lower_bound(item["prob"], item["std"], key[0])
        repaired, _, _ = repair_topology(item["traversable"], q, tau_gate=key[1])
        labels, _ = ndimage.label(repaired, structure=LABEL_STRUCT)
        distance = ndimage.distance_transform_edt(repaired).astype(np.float32)
        item["repair"][key] = (q, repaired, labels, distance)
    return item["repair"][key]


def summarize_image(successes: list[int], off_roads: list[float]) -> dict[str, float]:
    return {
        "success": float(np.mean(successes)) if successes else float("nan"),
        "safe_at_5": float(sum(value <= 0.05 for value in off_roads) / max(len(successes), 1)),
        "off_road": float(np.mean(off_roads)) if off_roads else float("nan"),
        "n_queries": len(successes),
    }


def aggregate(per_image: list[dict[str, float]]) -> dict[str, float]:
    def defined(key: str) -> float:
        values = [row[key] for row in per_image if not np.isnan(row[key])]
        return float(np.mean(values)) if values else float("nan")

    return {
        "image_mean_success": defined("success"),
        "image_mean_safe_at_5": defined("safe_at_5"),
        "image_mean_off_road": defined("off_road"),
        "n_images": len(per_image),
        "n_queries": int(sum(row["n_queries"] for row in per_image)),
    }


def evaluate_proposed(cache: list[dict], config: dict) -> dict[str, float]:
    per_image = []
    for item in cache:
        q, repaired, labels, distance = repaired_for(item, config)
        p_cost = config["lambda_r"] * np.minimum(-np.log(np.clip(q, 1e-9, 1.0)), config["risk_cap"])
        p_cost = p_cost + config["lambda_b"] * np.exp(-distance / config["tau_d"])
        successes: list[int] = []
        off_roads: list[float] = []
        for query in item["queries"]:
            start = (int(query["start_row"]), int(query["start_col"]))
            goal = (int(query["goal_row"]), int(query["goal_col"]))
            if not reachable(labels, start, goal):
                successes.append(0)
                continue
            result = astar_search(
                "proposed", repaired, np.zeros_like(q), start, goal,
                point_cost=lambda point: float(p_cost[point]), max_expansions=MAX_EXPANSIONS,
            )
            successes.append(int(result.success))
            if result.success:
                off_roads.append(1.0 - sum(
                    1 for row, col in result.path if item["gt"][row, col]
                ) / len(result.path))
        per_image.append(summarize_image(successes, off_roads))
    return aggregate(per_image)


def evaluate_ura(cache: list[dict], lambda_u: float) -> dict[str, float]:
    per_image = []
    for item in cache:
        point_cost = -np.log(np.clip(item["prob"], 1e-9, 1.0)) + lambda_u * item["entropy"]
        successes: list[int] = []
        off_roads: list[float] = []
        for query in item["queries"]:
            start = (int(query["start_row"]), int(query["start_col"]))
            goal = (int(query["goal_row"]), int(query["goal_col"]))
            if not reachable(item["labels"], start, goal):
                successes.append(0)
                continue
            result = astar_search(
                "ura", item["traversable"], np.zeros_like(item["prob"]), start, goal,
                point_cost=lambda point: float(point_cost[point]), max_expansions=MAX_EXPANSIONS,
            )
            successes.append(int(result.success))
            if result.success:
                off_roads.append(1.0 - sum(
                    1 for row, col in result.path if item["gt"][row, col]
                ) / len(result.path))
        per_image.append(summarize_image(successes, off_roads))
    return aggregate(per_image)


def select_ura(rows: list[dict]) -> dict:
    """Maximize safe task utility; use the smallest value within one point."""
    best_safe = max(row["image_mean_safe_at_5"] for row in rows)
    eligible = [
        row for row in rows
        if row["image_mean_safe_at_5"] >= best_safe - 0.01
    ]
    return min(eligible, key=lambda row: row["lambda_u"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--prediction-dir", type=Path, default=Path("artifacts/predictions/calibration")
    )
    parser.add_argument("--n-images", type=int, default=25)
    parser.add_argument("--queries-per-image", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/raw_metrics/planning_parameter_sensitivity_cal.json"),
    )
    args = parser.parse_args()

    root = args.root.resolve()
    prediction_dir = args.prediction_dir if args.prediction_dir.is_absolute() else root / args.prediction_dir
    image_ids = sorted(build_splits(root)["calibration_ids"])[:args.n_images]
    cache = load_cache(root, prediction_dir, image_ids, args.queries_per_image, args.seed)
    print(f"Calibration-only OAT: {len(cache)} images, {args.queries_per_image} queries/image", flush=True)

    sensitivity: dict[str, list[dict]] = {}
    for name, values in SPECS.items():
        rows = []
        for value in values:
            config = dict(NOMINAL)
            config[name] = value
            metrics = evaluate_proposed(cache, config)
            rows.append({name: value, **metrics})
            print(
                f"{name}={value:g}: Success={metrics['image_mean_success']:.4f} "
                f"RouteConform@5={metrics['image_mean_safe_at_5']:.4f} "
                f"off-road={metrics['image_mean_off_road']:.4f}", flush=True,
            )
        sensitivity[name] = rows

    ura_rows = []
    for lambda_u in URA_LAMBDAS:
        metrics = evaluate_ura(cache, lambda_u)
        ura_rows.append({"lambda_u": lambda_u, **metrics})
        print(
            f"URA lambda_u={lambda_u:g}: Success={metrics['image_mean_success']:.4f} "
            f"RouteConform@5={metrics['image_mean_safe_at_5']:.4f} "
            f"off-road={metrics['image_mean_off_road']:.4f}", flush=True,
        )
    selected_ura = select_ura(ura_rows)
    print(f"Selected URA lambda_u={selected_ura['lambda_u']:g}", flush=True)

    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "protocol": {
            "split": "calibration only",
            "selection_unit": "image mean",
            "n_images": len(cache),
            "queries_per_image": args.queries_per_image,
            "seed": args.seed,
            "nominal_proposed_config": NOMINAL,
            "sensitivity": "one factor at a time around nominal configuration",
        },
        "proposed_sensitivity": sensitivity,
        "ura_calibration": {
            "selection_rule": "smallest lambda_u within one percentage point of the best image-mean RouteConform@5",
            "results": ura_rows,
            "selected": selected_ura,
        },
    }, indent=2), encoding="utf-8")
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
