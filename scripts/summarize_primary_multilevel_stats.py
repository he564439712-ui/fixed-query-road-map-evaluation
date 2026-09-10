#!/usr/bin/env python
"""Summarize locked Massachusetts before--after contrasts by image.

This reporting-only script reads existing per-image artifacts.  It does not
run inference, regenerate queries, alter a frozen product, or inspect labels
beyond those already represented in the saved metrics.  The statistical unit
is the image: bootstrap resampling retains every fixed query from a sampled
image.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def bootstrap_interval(values: np.ndarray, draws: int, seed: int) -> tuple[float, float]:
    """Two-sided percentile interval for a mean paired image contrast."""
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.percentile(means, (2.5, 97.5))
    return float(low), float(high)


def summarize(values: np.ndarray, draws: int, seed: int) -> dict[str, object]:
    return {
        "n_images": int(len(values)),
        "mean_delta": float(values.mean()),
        "delta_bootstrap_95_ci": list(bootstrap_interval(values, draws, seed)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/raw_metrics/primary_multilevel_stats.json"),
    )
    args = parser.parse_args()
    root = args.root.resolve()

    csv_path = root / "artifacts/raw_metrics/massachusetts_test_seg.csv"
    apls_path = root / "artifacts/raw_metrics/massachusetts_apls_reference.json"
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    with apls_path.open(encoding="utf-8") as stream:
        apls_rows = json.load(stream)["per_image"]

    if len(rows) != len(apls_rows):
        raise RuntimeError("Segmentation and APLS artifacts have different image counts.")
    if {row["image_id"] for row in rows} != {row["image_id"] for row in apls_rows}:
        raise RuntimeError("Segmentation and APLS artifacts have different image IDs.")

    as_float = lambda name: np.asarray([float(row[name]) for row in rows], dtype=float)
    metrics = {
        "components_per_image": as_float("components_repaired") - as_float("components_orig"),
        "structural_connectivity": as_float("connectivity_repaired") - as_float("connectivity_orig"),
        "dice": as_float("dice_repaired") - as_float("dice"),
        "cldice": as_float("cldice_repaired") - as_float("cldice"),
        "reference_apls": np.asarray(
            [row["repaired"]["apls"] - row["original"]["apls"] for row in apls_rows],
            dtype=float,
        ),
    }
    result = {
        "analysis": "Paired Original-to-Repaired Massachusetts image contrasts",
        "statistical_unit": "image cluster retaining all fixed queries",
        "bootstrap_replicates": args.draws,
        "bootstrap_seed": args.seed,
        "sources": [str(csv_path.relative_to(root)), str(apls_path.relative_to(root))],
        "metrics": {
            name: summarize(values, args.draws, args.seed)
            for name, values in metrics.items()
        },
    }
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
