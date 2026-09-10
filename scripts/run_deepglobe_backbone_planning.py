#!/usr/bin/env python
"""Run frozen DeepGlobe planning for an arbitrary segmentation backbone.

This wrapper deliberately reuses the Massachusetts planning operating point;
it does not tune a backbone-specific threshold or cost weight.  First create
the dense predictions with ``predict_deepglobe_ensemble.py`` and then run, for
example:

    python scripts/run_deepglobe_backbone_planning.py \
      --prediction-dir artifacts/predictions/deepglobe_deeplabv3_test \
      --output-csv artifacts/raw_metrics/deepglobe_deeplabv3_planning_tau2.csv

For a single DeepLabV3 checkpoint, the prediction script writes an all-zero
standard-deviation array.  Therefore q=p in the Topology+Risk cost: this is a
backbone-robustness test of repair and boundary clearance, not an additional
test of ensemble uncertainty.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
N_IMAGES = 624


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--tau-gate", type=float, default=2.0)
    parser.add_argument("--n-shards", type=int, default=6)
    args = parser.parse_args()

    if args.n_shards < 1 or N_IMAGES % args.n_shards:
        raise ValueError(f"n-shards must divide {N_IMAGES}, got {args.n_shards}")
    pred_dir = args.prediction_dir if args.prediction_dir.is_absolute() else ROOT / args.prediction_dir
    if not pred_dir.is_dir():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")
    output_csv = args.output_csv if args.output_csv.is_absolute() else ROOT / args.output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    shard_size = N_IMAGES // args.n_shards
    shard_paths: list[Path] = []
    processes = []
    for shard_index in range(args.n_shards):
        shard = output_csv.with_name(f"{output_csv.stem}_shard{shard_index}.csv")
        log = ROOT / "artifacts" / "logs" / f"{output_csv.stem}_shard{shard_index}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        shard_paths.append(shard)
        cmd = [
            sys.executable, str(ROOT / "scripts" / "evaluate_planning.py"),
            "--dataset", "deepglobe",
            "--prediction-dir", str(pred_dir),
            "--queries-csv", "artifacts/queries/deepglobe_queries.csv",
            "--tau-gate", str(args.tau_gate),
            "--skip", str(shard_index * shard_size),
            "--limit", str(shard_size),
            "--output-csv", str(shard),
        ]
        print(f"launching shard {shard_index}: {shard.name}", flush=True)
        with log.open("w", encoding="utf-8") as handle:
            processes.append((shard_index, log, subprocess.Popen(cmd, cwd=ROOT, stdout=handle,
                                                                  stderr=subprocess.STDOUT)))

    failed = []
    for shard_index, log, process in processes:
        return_code = process.wait()
        print(f"shard {shard_index} exit={return_code}", flush=True)
        if return_code:
            failed.append(shard_index)
            print("\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]),
                  flush=True)
    if failed:
        raise RuntimeError(f"Planning shards failed: {failed}")

    rows: list[dict[str, str]] = []
    fieldnames = None
    for shard in shard_paths:
        with shard.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames if fieldnames is None else fieldnames
            rows.extend(reader)
    if fieldnames is None:
        raise RuntimeError("No planning rows were produced")
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    methods = sorted({row["method"] for row in rows})
    expected = 10620
    counts = {method: sum(row["method"] == method for row in rows) for method in methods}
    if any(count != expected for count in counts.values()):
        raise RuntimeError(f"Unexpected row counts: {counts}")
    print(f"merged {len(rows)} rows -> {output_csv}")
    print(f"{'method':<24} {'success':>8} {'conform@5':>10} {'off-road':>9}")
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        success = np.asarray([int(row["success"]) for row in subset], dtype=float)
        off = np.asarray([float(row["off_road_ratio"]) for row in subset], dtype=float)
        safe = success * np.isfinite(off) * (off <= 0.05)
        mean_off = np.nanmean(off)
        print(f"{method:<24} {success.mean():8.3f} {safe.mean():8.3f} {mean_off:9.4f}")


if __name__ == "__main__":
    main()
