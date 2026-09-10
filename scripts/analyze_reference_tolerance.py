#!/usr/bin/env python
"""Reference-mask tolerance sensitivity for frozen Massachusetts routes.

The script reruns only the two primary planners with the already frozen
predictions, queries, repair gate, planning weights, and 200k expansion cap.
It then evaluates RouteConform@5% against 0/1/2/3-pixel square dilations of
the reference mask. Dilation changes evaluation only; it is never exposed to
either planner.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.dataset import load_mask
from src.data.queries import get_queries_for_image, load_queries
from src.data.splits import build_splits, resolve_image_paths
from src.planning.costs import build_traversability, uncertainty_discounted_score
from src.planning.grid import SearchResult, astar_search
from src.planning.topology_repair import repair_topology


SOFT = "Soft-probability A*"
FULL = "Topology+Risk A*"
MAX_EXPANSIONS = 200_000
LABEL_STRUCTURE = np.ones((3, 3), dtype=np.uint8)


def reachable(labels: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> bool:
    start_label = int(labels[start])
    return start_label != 0 and start_label == int(labels[goal])


def plan(
    name: str,
    traversable: np.ndarray,
    labels: np.ndarray,
    cost_map: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> SearchResult:
    if not reachable(labels, start, goal):
        return SearchResult(name, [], 0.0, 0, False)
    return astar_search(
        name,
        traversable,
        np.zeros_like(cost_map),
        start,
        goal,
        point_cost=lambda point: float(cost_map[point]),
        max_expansions=MAX_EXPANSIONS,
    )


def dilated_masks(reference: np.ndarray, radii: tuple[int, ...]) -> dict[int, np.ndarray]:
    masks = {0: reference.astype(bool)}
    for radius in radii:
        if radius == 0:
            continue
        structure = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
        masks[radius] = ndimage.binary_dilation(reference, structure=structure)
    return masks


def off_reference_ratio(path: list[tuple[int, int]], reference: np.ndarray) -> float:
    if not path:
        return float("nan")
    on_reference = sum(bool(reference[row, col]) for row, col in path)
    return float(1.0 - on_reference / len(path))


def paired_percentile_ci(
    full: np.ndarray,
    soft: np.ndarray,
    bootstrap: int,
    seed: int,
) -> dict[str, object]:
    difference = full - soft
    rng = np.random.default_rng(seed)
    samples = rng.choice(
        difference, size=(bootstrap, len(difference)), replace=True
    ).mean(axis=1)
    return {
        "unit": "image",
        "n_images": int(len(difference)),
        "full_mean": float(np.mean(full)),
        "soft_mean": float(np.mean(soft)),
        "paired_difference": float(np.mean(difference)),
        "bootstrap_replicates": int(bootstrap),
        "bootstrap_seed": int(seed),
        "ci_type": "two-sided percentile interval",
        "bootstrap_95_ci": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--prediction-dir", type=Path, default=Path("artifacts/predictions/test")
    )
    parser.add_argument(
        "--queries", type=Path, default=Path("artifacts/queries/queries.csv")
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("artifacts/raw_metrics/reference_tolerance_per_query.csv"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/raw_metrics/reference_tolerance_stats.json"),
    )
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    prediction_dir = args.prediction_dir if args.prediction_dir.is_absolute() else root / args.prediction_dir
    query_path = args.queries if args.queries.is_absolute() else root / args.queries
    output_csv = args.output_csv if args.output_csv.is_absolute() else root / args.output_csv
    output_json = args.output_json if args.output_json.is_absolute() else root / args.output_json

    image_ids = build_splits(root)["test_ids"]
    if args.limit is not None:
        image_ids = image_ids[: args.limit]
    queries = load_queries(query_path)
    radii = (0, 1, 2, 3)
    rows: list[dict[str, object]] = []

    for image_index, image_id in enumerate(image_ids, start=1):
        probability = np.load(prediction_dir / f"{image_id}_prob.npy").astype(np.float32)
        std_path = prediction_dir / f"{image_id}_std.npy"
        epistemic_std = (
            np.load(std_path).astype(np.float32)
            if std_path.exists()
            else np.zeros_like(probability)
        )
        _, mask_path = resolve_image_paths(root, image_id)
        ground_truth = load_mask(mask_path).astype(bool)
        references = dilated_masks(ground_truth, radii)

        q = uncertainty_discounted_score(probability, epistemic_std, kappa=1.0)
        original = build_traversability(
            probability, threshold=0.5, use_largest_component=False
        )
        repaired, _bridges, _skeleton = repair_topology(original, q, tau_gate=2.0)
        original_labels, _ = ndimage.label(original, structure=LABEL_STRUCTURE)
        repaired_labels, _ = ndimage.label(repaired, structure=LABEL_STRUCTURE)
        repaired_distance = ndimage.distance_transform_edt(repaired).astype(np.float32)

        clipped_p = np.clip(probability, 1e-9, 1.0 - 1e-9)
        clipped_q = np.clip(q, 1e-9, 1.0 - 1e-9)
        soft_cost = -np.log(clipped_p)
        full_cost = (
            2.0 * np.minimum(-np.log(clipped_q), 1.0)
            + np.exp(-repaired_distance / 5.0)
        )

        for query in get_queries_for_image(queries, image_id):
            start = (int(query["start_row"]), int(query["start_col"]))
            goal = (int(query["goal_row"]), int(query["goal_col"]))
            results = (
                plan(SOFT, original, original_labels, soft_cost, start, goal),
                plan(FULL, repaired, repaired_labels, full_cost, start, goal),
            )
            for result in results:
                row: dict[str, object] = {
                    "image_id": image_id,
                    "query_index": int(query["query_index"]),
                    "method": result.name,
                    "success": int(result.success),
                    "expanded_nodes": int(result.expanded_nodes),
                }
                for radius in radii:
                    ratio = off_reference_ratio(result.path, references[radius])
                    row[f"off_reference_r{radius}"] = ratio
                    row[f"routeconform_0.05_r{radius}"] = int(
                        result.success and np.isfinite(ratio) and ratio <= 0.05
                    )
                rows.append(row)
        print(f"[{image_index}/{len(image_ids)}] {image_id}", flush=True)

    expected_queries = sum(
        len(get_queries_for_image(queries, image_id)) for image_id in image_ids
    )
    if len(rows) != 2 * expected_queries:
        raise RuntimeError(
            f"Expected {2 * expected_queries} method-query rows, found {len(rows)}"
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by_method_image: dict[str, dict[str, list[dict[str, object]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        by_method_image[str(row["method"])][str(row["image_id"])].append(row)
    common_images = sorted(
        set(by_method_image[FULL]).intersection(by_method_image[SOFT])
    )
    tolerance_results: dict[str, object] = {}
    for radius in radii:
        field = f"routeconform_0.05_r{radius}"
        full = np.asarray([
            np.mean([float(row[field]) for row in by_method_image[FULL][image_id]])
            for image_id in common_images
        ])
        soft = np.asarray([
            np.mean([float(row[field]) for row in by_method_image[SOFT][image_id]])
            for image_id in common_images
        ])
        query_full = float(np.mean([float(row[field]) for row in rows if row["method"] == FULL]))
        query_soft = float(np.mean([float(row[field]) for row in rows if row["method"] == SOFT]))
        tolerance_results[str(radius)] = {
            "dilation": f"square Chebyshev radius {radius} pixel(s)",
            "query_pooled": {
                "full": query_full,
                "soft": query_soft,
                "difference": query_full - query_soft,
            },
            "image_clustered": paired_percentile_ci(
                full, soft, args.bootstrap, args.seed
            ),
        }

    result = {
        "validation": {
            "n_images": len(image_ids),
            "n_queries": expected_queries,
            "n_rows": len(rows),
            "success": {
                method: float(np.mean([
                    int(row["success"]) for row in rows if row["method"] == method
                ]))
                for method in (SOFT, FULL)
            },
        },
        "metric": (
            "RouteConform@5% = success and no more than 5% of raster path nodes "
            "outside the tolerance-dilated reference road mask"
        ),
        "evaluation_only": True,
        "tolerances": tolerance_results,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved {output_csv}")
    print(f"Saved {output_json}")


if __name__ == "__main__":
    main()
