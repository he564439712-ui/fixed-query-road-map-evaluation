#!/usr/bin/env python
"""Image-clustered statistics for the frozen DeepGlobe planning transfer."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

PROPOSED = "Topology+Risk A*"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 2025


def paired_report(diff: np.ndarray) -> dict:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap = np.asarray([
        np.mean(diff[rng.integers(0, len(diff), len(diff))])
        for _ in range(BOOTSTRAP_REPLICATES)
    ])
    try:
        wilcoxon_p = float(stats.wilcoxon(diff).pvalue)
    except ValueError:
        wilcoxon_p = float("nan")
    return {
        "n_images": int(len(diff)),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "ci_type": "two-sided percentile interval",
        "mean_difference": float(np.mean(diff)),
        "ci95": [float(np.percentile(bootstrap, 2.5)),
                 float(np.percentile(bootstrap, 97.5))],
        "paired_t_p": float(stats.ttest_1samp(diff, 0.0).pvalue),
        "wilcoxon_p": wilcoxon_p,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path,
                        default=Path("artifacts/raw_metrics/deepglobe_planning_tau2.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/raw_metrics/deepglobe_planning_tau2_stats.json"))
    args = parser.parse_args()
    rows = list(csv.DictReader(args.input.open(encoding="utf-8")))
    counts = {method: sum(row["method"] == method for row in rows)
              for method in sorted({row["method"] for row in rows})}
    if len(rows) != 63720 or any(count != 10620 for count in counts.values()):
        raise ValueError(f"Unexpected DeepGlobe row counts: total={len(rows)}, {counts}")

    by_method = {
        method: {(row["image_id"], int(row["query_index"])): row
                 for row in rows if row["method"] == method}
        for method in counts
    }
    baselines = sorted(set(counts) - {PROPOSED, "GT Oracle A*"})
    contrasts = {}
    for baseline in baselines:
        per_image = defaultdict(lambda: {"p_s": [], "b_s": [], "p_safe": [],
                                         "b_safe": [], "p_off": [], "b_off": []})
        keys = sorted(by_method[PROPOSED].keys() & by_method[baseline].keys())
        for key in keys:
            p = by_method[PROPOSED][key]
            b = by_method[baseline][key]
            p_success, b_success = int(p["success"]), int(b["success"])
            p_off, b_off = float(p["off_road_ratio"]), float(b["off_road_ratio"])
            d = per_image[key[0]]
            d["p_s"].append(p_success)
            d["b_s"].append(b_success)
            d["p_safe"].append(int(p_success and np.isfinite(p_off) and p_off <= 0.05))
            d["b_safe"].append(int(b_success and np.isfinite(b_off) and b_off <= 0.05))
            if p_success and b_success and np.isfinite(p_off) and np.isfinite(b_off):
                d["p_off"].append(p_off)
                d["b_off"].append(b_off)

        def difference(left, right):
            return np.asarray([
                np.mean(per_image[image_id][left]) - np.mean(per_image[image_id][right])
                for image_id in sorted(per_image)
                if per_image[image_id][left] and per_image[image_id][right]
            ], dtype=np.float64)

        contrasts[baseline] = {
            "contrast": f"{PROPOSED} minus {baseline}",
            "success": paired_report(difference("p_s", "b_s")),
            "safe_success_0.05": paired_report(difference("p_safe", "b_safe")),
            "off_road_common_success": paired_report(difference("p_off", "b_off")),
        }

    result = {
        "validation": {"total_rows": len(rows), "rows_per_method": counts,
                       "n_images": len({row["image_id"] for row in rows})},
        "contrast_sign": "Topology+Risk minus baseline; negative off-road is better",
        "contrasts": contrasts,
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for baseline, values in contrasts.items():
        success = values["success"]
        safe = values["safe_success_0.05"]
        off = values["off_road_common_success"]
        print(f"{baseline}: success {success['mean_difference']:+.4f} "
              f"CI{success['ci95']}; RouteConform@5 {safe['mean_difference']:+.4f}; "
              f"off-road {off['mean_difference']:+.4f}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
