#!/usr/bin/env python
"""Run the complete six-method planning benchmark on all 13 corruptions.

Each condition is evaluated by ``evaluate_planning.py`` using the frozen
``tau_gate=2.0`` protocol and all 945 fixed test queries.  Condition-level
files are preserved, then merged with a condition column.  The JSON summary
includes image-clustered paired contrasts against Topology+Risk A*.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.corruptions import CORRUPTION_CONDITIONS

TAU_GATE = 2.0
PROPOSED = "Topology+Risk A*"


def run_condition(root: Path, condition: str) -> tuple[str, int, str]:
    out_dir = root / "artifacts/raw_metrics/corrupted_planning_tau2"
    log_dir = root / "artifacts/logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    output_csv = out_dir / f"{condition}.csv"
    log_path = log_dir / f"corrupted_planning_tau2_{condition}.log"
    command = [
        sys.executable,
        str(root / "scripts/evaluate_planning.py"),
        "--prediction-dir", f"artifacts/predictions/corruptions/test/{condition}",
        "--queries-csv", "artifacts/queries/queries.csv",
        "--tau-gate", str(TAU_GATE),
        "--output-csv", str(output_csv),
    ]
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
    tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-12:])
    return condition, process.returncode, tail


def paired_image_report(rows, baseline):
    variants = (PROPOSED, baseline)
    per_query = {
        method: {
            (row["condition"], row["image_id"], int(row["query_index"])): row
            for row in rows if row["method"] == method and row["condition"] != "clean"
        }
        for method in variants
    }
    keys = sorted(per_query[PROPOSED].keys() & per_query[baseline].keys())
    per_image = defaultdict(lambda: {"p_s": [], "b_s": [], "p_safe": [], "b_safe": [],
                                    "p_off": [], "b_off": []})
    for key in keys:
        p = per_query[PROPOSED][key]
        b = per_query[baseline][key]
        image_id = key[1]
        p_success, b_success = int(p["success"]), int(b["success"])
        p_off = float(p["off_road_ratio"])
        b_off = float(b["off_road_ratio"])
        d = per_image[image_id]
        d["p_s"].append(p_success)
        d["b_s"].append(b_success)
        d["p_safe"].append(int(p_success and np.isfinite(p_off) and p_off <= 0.05))
        d["b_safe"].append(int(b_success and np.isfinite(b_off) and b_off <= 0.05))
        if p_success and b_success and np.isfinite(p_off) and np.isfinite(b_off):
            d["p_off"].append(p_off)
            d["b_off"].append(b_off)

    def vector(left, right):
        values = []
        for image_id in sorted(per_image):
            d = per_image[image_id]
            if d[left] and d[right]:
                values.append(float(np.mean(d[left]) - np.mean(d[right])))
        return np.asarray(values, dtype=np.float64)

    def report(diff):
        if not len(diff):
            return {"n_images": 0}
        rng = np.random.default_rng(2025)
        boot = np.asarray([
            np.mean(diff[rng.integers(0, len(diff), len(diff))]) for _ in range(10000)
        ])
        try:
            wilcoxon_p = float(stats.wilcoxon(diff).pvalue)
        except ValueError:
            wilcoxon_p = float("nan")
        return {
            "n_images": int(len(diff)),
            "mean_difference": float(np.mean(diff)),
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "paired_t_p": float(stats.ttest_1samp(diff, 0.0).pvalue),
            "wilcoxon_p": wilcoxon_p,
        }

    return {
        "contrast": f"{PROPOSED} minus {baseline}",
        "unit": "image; each image averaged over the 12 degraded conditions",
        "success": report(vector("p_s", "b_s")),
        "safe_success_0.05": report(vector("p_safe", "b_safe")),
        "off_road_common_success": report(vector("p_off", "b_off")),
    }


def merge_and_summarize(root: Path, conditions):
    rows = []
    source_dir = root / "artifacts/raw_metrics/corrupted_planning_tau2"
    for condition in conditions:
        with (source_dir / f"{condition}.csv").open(encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                rows.append({"condition": condition, **row})
    rows.sort(key=lambda row: (row["condition"], row["image_id"],
                               int(row["query_index"]), row["method"]))
    merged_path = root / "artifacts/raw_metrics/corrupted_planning_tau2.csv"
    with merged_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    results = {}
    for condition in conditions:
        results[condition] = {}
        for method in sorted({row["method"] for row in rows}):
            values = [row for row in rows
                      if row["condition"] == condition and row["method"] == method]
            successful = [row for row in values if row["success"] == "1"]
            off = [float(row["off_road_ratio"]) for row in successful
                   if np.isfinite(float(row["off_road_ratio"]))]
            detour = [float(row["relative_detour"]) for row in successful
                      if np.isfinite(float(row["relative_detour"]))]
            results[condition][method] = {
                "n_queries": len(values),
                "success": float(np.mean([int(row["success"]) for row in values])),
                "safe_success_0.05": float(np.mean([
                    int(row["success"] == "1" and
                        np.isfinite(float(row["off_road_ratio"])) and
                        float(row["off_road_ratio"]) <= 0.05)
                    for row in values
                ])),
                "off_road_successes": float(np.mean(off)) if off else float("nan"),
                "detour_median_successes": float(np.median(detour)) if detour else float("nan"),
            }

    baselines = sorted({row["method"] for row in rows} - {PROPOSED, "GT Oracle A*"})
    summary = {
        "protocol": {
            "tau_gate": TAU_GATE,
            "n_conditions": len(conditions),
            "conditions": list(conditions),
            "n_images": len({row["image_id"] for row in rows}),
            "queries_per_condition": len(rows) // len(conditions) // 6,
            "total_rows": len(rows),
        },
        "results": results,
        "degraded_image_clustered_contrasts": {
            baseline: paired_image_report(rows, baseline) for baseline in baselines
        },
    }
    json_path = root / "artifacts/raw_metrics/corrupted_planning_tau2.json"
    json_path.write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    return merged_path, json_path, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--conditions", nargs="+", default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    conditions = args.conditions or list(CORRUPTION_CONDITIONS)
    print(f"Corrupted planning: {len(conditions)} conditions, tau_gate={TAU_GATE}, "
          f"workers={args.workers}", flush=True)
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_condition, root, condition): condition
                   for condition in conditions}
        for future in as_completed(futures):
            condition, returncode, tail = future.result()
            print(f"[{condition}] exit={returncode}", flush=True)
            if returncode:
                failures.append(condition)
                print(tail, flush=True)
    if failures:
        raise RuntimeError(f"Conditions failed: {failures}")
    merged_path, json_path, summary = merge_and_summarize(root, conditions)
    print(f"Merged: {merged_path}")
    print(f"Summary: {json_path}")
    print("\ncondition           proposed success   RouteConform@5%   off-road")
    for condition in conditions:
        row = summary["results"][condition][PROPOSED]
        print(f"{condition:<20} {row['success']:8.3f} {row['safe_success_0.05']:10.3f} "
              f"{row['off_road_successes']:10.4f}")


if __name__ == "__main__":
    main()
