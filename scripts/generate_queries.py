#!/usr/bin/env python
"""Generate fixed start-goal query set for planning evaluation.

Produces artifacts/queries/queries.csv with up to 20 requested queries per
test image. The current frozen Massachusetts artifact has 945 valid rows
(49 images; five sparse images yield only 13 valid pairs).

Run once and lock — all methods and seeds share this file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.queries import generate_all_queries
from src.data.splits import build_splits


def main():
    parser = argparse.ArgumentParser(
        description="Generate fixed planning queries for all test images."
    )
    parser.add_argument(
        "--root", type=Path, default=Path("."), help="Project root"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/queries/queries.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--per-image", type=int, default=20, help="Queries per test image"
    )
    parser.add_argument(
        "--short-ratio", type=float, default=0.30, help="Short distance ratio"
    )
    parser.add_argument(
        "--medium-ratio", type=float, default=0.37, help="Medium distance ratio"
    )
    parser.add_argument(
        "--snap-radius", type=int, default=8, help="Start/goal snap radius"
    )
    parser.add_argument(
        "--seed", type=int, default=2025, help="Random seed (fixed)"
    )
    args = parser.parse_args()

    splits = build_splits(args.root)
    test_ids = splits["test_ids"]
    print(f"Generating queries for {len(test_ids)} test images...")

    queries = generate_all_queries(
        root=args.root,
        output_path=args.root / args.output,
        test_ids=test_ids,
        queries_per_image=args.per_image,
        short_ratio=args.short_ratio,
        medium_ratio=args.medium_ratio,
        snap_radius=args.snap_radius,
        seed=args.seed,
    )

    print(f"Done — {len(queries)} queries saved to {args.output}")
    print(f"Expected: {len(test_ids) * args.per_image} queries")


if __name__ == "__main__":
    main()
