#!/usr/bin/env python
"""Generate fixed start-goal queries for the locked DeepGlobe test split.

Adapts scripts/generate_queries.py / src/data/queries.generate_all_queries to
the DeepGlobe data layout (data/train/{id}_sat.jpg + {id}_mask.png) by reusing
the dataset-agnostic generate_queries_for_image on each test GT mask.

Output: artifacts/queries/deepglobe_queries.csv  (same schema as Massachusetts)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import load_mask
from src.data.deepglobe import build_deepglobe_splits, resolve_deepglobe_paths
from src.data.queries import generate_queries_for_image

FIELDNAMES = [
    "query_index", "image_id", "start_row", "start_col",
    "goal_row", "goal_col", "euclidean_distance", "distance_class",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("artifacts/queries/deepglobe_queries.csv"))
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--queries-per-image", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    root = args.root.resolve()
    test_ids = sorted(build_deepglobe_splits(root, seed=args.split_seed)["test_ids"])
    print(f"DeepGlobe test: {len(test_ids)} images, {args.queries_per_image} queries/image",
          flush=True)

    all_queries: list[dict] = []
    for i, image_id in enumerate(test_ids):
        _img_path, mask_path = resolve_deepglobe_paths(root, image_id)
        road_mask = load_mask(mask_path)
        queries = generate_queries_for_image(
            road_mask, num_queries=args.queries_per_image, seed=args.seed + i * 1000,
        )
        for qi, q in enumerate(queries):
            q["image_id"] = image_id
            q["query_index"] = len(all_queries) + qi
        all_queries.extend(queries)
        if len(queries) < args.queries_per_image:
            print(f"WARNING: {image_id} only {len(queries)}/{args.queries_per_image} queries",
                  flush=True)

    out = args.output if args.output.is_absolute() else root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_queries)

    classes = {"short": 0, "medium": 0, "long": 0}
    for q in all_queries:
        classes[q["distance_class"]] += 1
    print(f"Saved {len(all_queries)} queries to {out}")
    print(f"Distribution: short={classes['short']}, medium={classes['medium']}, long={classes['long']}")


if __name__ == "__main__":
    main()
