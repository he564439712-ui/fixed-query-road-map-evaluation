#!/usr/bin/env python
"""Clean-test 2x2x2 planning ablation with controlled cost transforms.

Factors:
1. mask: original vs q-weighted topology repair
2. score used by A*: ensemble mean p vs conservative score q
3. clearance boundary penalty: off vs on

Unlike the legacy 2x2 ablation, p and q receive the same risk weight and cap.
This isolates the incremental effect of q from the boundary-distance term.

The q-weighted repaired mask is held fixed across all repaired variants; the
repair factor therefore measures the complete repair stage, while the p-vs-q
factor measures only the planning-score substitution.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import load_mask
from src.data.queries import get_queries_for_image, load_queries
from src.data.splits import build_splits, resolve_image_paths
from src.planning.costs import (
    build_traversability,
    conservative_confidence_lower_bound,
    make_matched_score_cost,
)
from src.planning.grid import SearchResult, astar_search, path_length
from src.planning.topology_repair import repair_topology


MAX_EXPANSIONS = 200_000
LABEL_STRUCT = np.ones((3, 3), dtype=np.uint8)
SAFE_THRESHOLDS = (0.01, 0.05, 0.10)


def _reachable(labels: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> bool:
    start_id = int(labels[start])
    goal_id = int(labels[goal])
    return start_id != 0 and start_id == goal_id


def _path_off_road(path: list[tuple[int, int]], ground_truth: np.ndarray) -> float:
    if not path:
        return float("nan")
    on_road = sum(bool(ground_truth[p]) for p in path)
    return float(1.0 - on_road / len(path))


def _variant_name(mask_kind: str, score_kind: str, boundary: bool) -> str:
    return f"{mask_kind}+{score_kind}+boundary_{'on' if boundary else 'off'}"


def _process_image(payload: dict) -> list[dict]:
    root = Path(payload["root"])
    prediction_dir = Path(payload["prediction_dir"])
    image_id = payload["image_id"]
    queries = payload["queries"]

    prob = np.load(prediction_dir / f"{image_id}_prob.npy").astype(np.float32)
    std_path = prediction_dir / f"{image_id}_std.npy"
    std = np.load(std_path).astype(np.float32) if std_path.exists() else np.zeros_like(prob)
    _, mask_path = resolve_image_paths(root, image_id)
    ground_truth = load_mask(mask_path).astype(bool)

    q = conservative_confidence_lower_bound(prob, std, payload["kappa"])
    original = build_traversability(prob, payload["threshold"], use_largest_component=False)
    repaired, _, _ = repair_topology(
        original,
        q,
        tau_gate=payload["tau_gate"],
    )

    masks = {"original": original, "repaired": repaired}
    labels = {
        name: ndimage.label(mask, structure=LABEL_STRUCT)[0]
        for name, mask in masks.items()
    }
    distances = {
        name: ndimage.distance_transform_edt(mask).astype(np.float32)
        for name, mask in masks.items()
    }
    scores = {"p": prob, "q": q}

    costs = {}
    for mask_kind, score_kind, boundary in product(
        ("original", "repaired"), ("p", "q"), (False, True)
    ):
        costs[(mask_kind, score_kind, boundary)] = make_matched_score_cost(
            scores[score_kind],
            distances[mask_kind],
            lambda_r=payload["lambda_r"],
            lambda_b=payload["lambda_b"],
            tau_d=payload["tau_d"],
            risk_cap=payload["risk_cap"],
            use_boundary=boundary,
        )

    rows: list[dict] = []
    zero_risk = np.zeros_like(prob, dtype=np.float32)
    for query in queries:
        start = (int(query["start_row"]), int(query["start_col"]))
        goal = (int(query["goal_row"]), int(query["goal_col"]))
        for mask_kind, score_kind, boundary in product(
            ("original", "repaired"), ("p", "q"), (False, True)
        ):
            if not _reachable(labels[mask_kind], start, goal):
                result = SearchResult(
                    name="factorial", path=[], planning_time_ms=0.0,
                    expanded_nodes=0, success=False,
                )
            else:
                result = astar_search(
                    "factorial",
                    masks[mask_kind],
                    zero_risk,
                    start,
                    goal,
                    point_cost=costs[(mask_kind, score_kind, boundary)],
                    max_expansions=MAX_EXPANSIONS,
                )
            off_road = (
                _path_off_road(result.path, ground_truth)
                if result.success
                else float("nan")
            )
            row = {
                "image_id": image_id,
                "query_index": int(query["query_index"]),
                "distance_class": query.get("distance_class", ""),
                "variant": _variant_name(mask_kind, score_kind, boundary),
                "mask": mask_kind,
                "score": score_kind,
                "boundary": int(boundary),
                "success": int(result.success),
                "off_road_ratio": off_road,
                "path_length": path_length(result.path) if result.success else float("nan"),
                "expanded_nodes": int(result.expanded_nodes),
                "planning_time_ms": float(result.planning_time_ms),
            }
            for delta in SAFE_THRESHOLDS:
                key = f"safe_success_{delta:.2f}"
                row[key] = int(result.success and off_road <= delta)
            rows.append(row)
    return rows


def _aggregate(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["variant"]].append(row)

    result = {}
    for variant, values in sorted(grouped.items()):
        successes = [r for r in values if r["success"]]
        result[variant] = {
            "mask": values[0]["mask"],
            "score": values[0]["score"],
            "boundary": bool(values[0]["boundary"]),
            "n_queries": len(values),
            "success": float(np.mean([r["success"] for r in values])),
            "off_road": (
                float(np.mean([r["off_road_ratio"] for r in successes]))
                if successes
                else float("nan")
            ),
            "mean_path_length": (
                float(np.mean([r["path_length"] for r in successes]))
                if successes
                else float("nan")
            ),
            **{
                f"safe_success_{delta:.2f}": float(
                    np.mean([r[f"safe_success_{delta:.2f}"] for r in values])
                )
                for delta in SAFE_THRESHOLDS
            },
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled 2x2x2 planning ablation")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--prediction-dir", type=Path, default=Path("artifacts/predictions/test")
    )
    parser.add_argument(
        "--queries-csv", type=Path, default=Path("artifacts/queries/queries.csv")
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tau-gate", type=float, default=2.0)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--lambda-r", type=float, default=2.0)
    parser.add_argument("--lambda-b", type=float, default=1.0)
    parser.add_argument("--tau-d", type=float, default=5.0)
    parser.add_argument("--risk-cap", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("artifacts/raw_metrics/factorial_ablation.csv"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/raw_metrics/factorial_ablation.json"),
    )
    args = parser.parse_args()

    root = args.root.resolve()
    prediction_dir = (root / args.prediction_dir).resolve()
    queries = load_queries(root / args.queries_csv)
    image_ids = build_splits(root)["test_ids"]
    if args.limit is not None:
        image_ids = image_ids[: args.limit]

    payloads = []
    for image_id in image_ids:
        if not (prediction_dir / f"{image_id}_prob.npy").exists():
            continue
        payloads.append({
            "root": str(root),
            "prediction_dir": str(prediction_dir),
            "image_id": image_id,
            "queries": get_queries_for_image(queries, image_id),
            "threshold": args.threshold,
            "tau_gate": args.tau_gate,
            "kappa": args.kappa,
            "lambda_r": args.lambda_r,
            "lambda_b": args.lambda_b,
            "tau_d": args.tau_d,
            "risk_cap": args.risk_cap,
        })

    print(
        f"2x2x2 ablation: {len(payloads)} images, tau_gate={args.tau_gate}, "
        f"workers={args.workers}",
        flush=True,
    )
    all_rows: list[dict] = []
    if args.workers <= 1:
        for index, payload in enumerate(payloads, 1):
            all_rows.extend(_process_image(payload))
            print(f"[{index}/{len(payloads)}] {payload['image_id']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(_process_image, payload): payload for payload in payloads
            }
            done = 0
            for future in as_completed(futures):
                payload = futures[future]
                all_rows.extend(future.result())
                done += 1
                print(f"[{done}/{len(payloads)}] {payload['image_id']}", flush=True)

    if not all_rows:
        raise RuntimeError("No predictions or queries were evaluated")

    all_rows.sort(key=lambda r: (r["image_id"], r["query_index"], r["variant"]))
    output_csv = args.output_csv if args.output_csv.is_absolute() else root / args.output_csv
    output_json = (
        args.output_json if args.output_json.is_absolute() else root / args.output_json
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    summary = {
        "protocol": {
            "n_images": len(payloads),
            "tau_gate": args.tau_gate,
            "threshold": args.threshold,
            "kappa": args.kappa,
            "lambda_r": args.lambda_r,
            "lambda_b": args.lambda_b,
            "tau_d": args.tau_d,
            "risk_cap": args.risk_cap,
            "note": "p and q use identical risk transforms; repair uses q-weighted selection",
        },
        "results": _aggregate(all_rows),
    }
    output_json.write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
    )

    print()
    print("variant                                      success  off-road  RouteConform@5%")
    print("-" * 74)
    for variant, values in summary["results"].items():
        print(
            f"{variant:<44} {values['success']:7.3f}  "
            f"{values['off_road']:8.4f}  {values['safe_success_0.05']:7.3f}"
        )
    print(f"Saved {output_csv}")
    print(f"Saved {output_json}")


if __name__ == "__main__":
    main()
