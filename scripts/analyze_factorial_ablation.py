#!/usr/bin/env python
"""Image-clustered inference for the controlled 2x2x2 ablation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


def bootstrap_ci(diff: np.ndarray, n_boot: int = 20_000, seed: int = 2025):
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    n = len(diff)
    for index in range(n_boot):
        means[index] = np.mean(diff[rng.integers(0, n, n)])
    return (
        float(np.mean(diff)),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def paired_report(a: np.ndarray, b: np.ndarray) -> dict:
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    diff = a - b
    mean, lo, hi = bootstrap_ci(diff)
    t_result = stats.ttest_rel(a, b)
    try:
        w_result = stats.wilcoxon(a, b)
        wilcoxon_p = float(w_result.pvalue)
    except ValueError:
        wilcoxon_p = float("nan")
    sd = float(np.std(diff, ddof=1))
    return {
        "n_images": int(len(diff)),
        "bootstrap_replicates": 20_000,
        "bootstrap_seed": 2025,
        "ci_type": "two-sided percentile interval",
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "mean_diff": mean,
        "ci95": [lo, hi],
        "paired_t_p": float(t_result.pvalue),
        "wilcoxon_p": wilcoxon_p,
        "cohens_dz": float(mean / sd) if sd > 0 else 0.0,
    }


def variant_name(mask: str, score: str, boundary: bool) -> str:
    return f"{mask}+{score}+boundary_{'on' if boundary else 'off'}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/raw_metrics/factorial_ablation.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/raw_metrics/factorial_ablation_stats.json"),
    )
    args = parser.parse_args()

    rows = list(csv.DictReader(args.input.open(encoding="utf-8")))
    variants = sorted({row["variant"] for row in rows})
    by_variant_query = {
        variant: {
            (row["image_id"], int(row["query_index"])): row
            for row in rows
            if row["variant"] == variant
        }
        for variant in variants
    }
    counts = {variant: len(values) for variant, values in by_variant_query.items()}
    if len(rows) != 7560 or any(value != 945 for value in counts.values()):
        raise ValueError(f"Unexpected factorial row counts: total={len(rows)}, {counts}")

    image_ids = sorted({row["image_id"] for row in rows})
    per_image = defaultdict(dict)
    for variant in variants:
        grouped = defaultdict(list)
        for row in rows:
            if row["variant"] == variant:
                grouped[row["image_id"]].append(row)
        for image_id in image_ids:
            image_rows = grouped[image_id]
            per_image[variant][image_id] = {
                "success": float(np.mean([int(row["success"]) for row in image_rows])),
                "safe_success_0.05": float(
                    np.mean([int(row["safe_success_0.05"]) for row in image_rows])
                ),
            }

    def vector(variant: str, field: str) -> np.ndarray:
        return np.array(
            [per_image[variant][image_id][field] for image_id in image_ids],
            dtype=np.float64,
        )

    def common_success_offroad(a_name: str, b_name: str):
        a_map = by_variant_query[a_name]
        b_map = by_variant_query[b_name]
        a_image = defaultdict(list)
        b_image = defaultdict(list)
        for key in sorted(a_map):
            a_row = a_map[key]
            b_row = b_map[key]
            if a_row["success"] == "1" and b_row["success"] == "1":
                image_id = key[0]
                a_image[image_id].append(float(a_row["off_road_ratio"]))
                b_image[image_id].append(float(b_row["off_road_ratio"]))
        common_images = sorted(set(a_image) & set(b_image))
        a = np.array([np.mean(a_image[i]) for i in common_images], dtype=np.float64)
        b = np.array([np.mean(b_image[i]) for i in common_images], dtype=np.float64)
        return a, b

    contrasts = {}

    def add_contrast(label: str, a_name: str, b_name: str) -> None:
        contrast = {
            "variant_a": a_name,
            "variant_b": b_name,
            "success": paired_report(
                vector(a_name, "success"), vector(b_name, "success")
            ),
            "safe_success_0.05": paired_report(
                vector(a_name, "safe_success_0.05"),
                vector(b_name, "safe_success_0.05"),
            ),
        }
        a_off, b_off = common_success_offroad(a_name, b_name)
        contrast["off_road_common_success"] = paired_report(a_off, b_off)
        contrasts[label] = contrast

    for mask in ("original", "repaired"):
        for boundary in (False, True):
            add_contrast(
                f"q_minus_p__{mask}__boundary_{int(boundary)}",
                variant_name(mask, "q", boundary),
                variant_name(mask, "p", boundary),
            )

    for mask in ("original", "repaired"):
        for score in ("p", "q"):
            add_contrast(
                f"boundary_on_minus_off__{mask}__{score}",
                variant_name(mask, score, True),
                variant_name(mask, score, False),
            )

    for score in ("p", "q"):
        for boundary in (False, True):
            add_contrast(
                f"repair_minus_original__{score}__boundary_{int(boundary)}",
                variant_name("repaired", score, boundary),
                variant_name("original", score, boundary),
            )

    result = {
        "validation": {"total_rows": len(rows), "rows_per_variant": counts},
        "contrast_sign": "A minus B; negative off-road is better",
        "contrasts": contrasts,
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    for label, values in contrasts.items():
        success = values["success"]
        off = values["off_road_common_success"]
        safe = values["safe_success_0.05"]
        print(label)
        print(
            f"  success: {success['mean_diff']:+.4f} "
            f"CI[{success['ci95'][0]:+.4f},{success['ci95'][1]:+.4f}] "
            f"p={success['paired_t_p']:.3g}"
        )
        print(
            f"  off-road(common): {off['mean_diff']:+.4f} "
            f"CI[{off['ci95'][0]:+.4f},{off['ci95'][1]:+.4f}] "
            f"p={off['paired_t_p']:.3g}"
        )
        print(
            f"  RouteConform@5%: {safe['mean_diff']:+.4f} "
            f"CI[{safe['ci95'][0]:+.4f},{safe['ci95'][1]:+.4f}] "
            f"p={safe['paired_t_p']:.3g}"
        )

    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
