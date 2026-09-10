#!/usr/bin/env python
"""Run the DeepGlobe planning evaluation in 6 parallel shards, then merge.

Each shard runs evaluate_planning.py --dataset deepglobe over a disjoint slice
of the 624 test images with the frozen tau_gate=2.0 protocol; the per-query
CSVs are merged into artifacts/raw_metrics/deepglobe_planning_tau2.csv.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
N_SHARDS = 6
IMAGES = 624
CHUNK = IMAGES // N_SHARDS  # 104

BASE = [
    sys.executable, str(ROOT / "scripts/evaluate_planning.py"),
    "--dataset", "deepglobe",
    "--prediction-dir", "artifacts/predictions/deepglobe_test",
    "--queries-csv", "artifacts/queries/deepglobe_queries.csv",
    "--tau-gate", "2.0",
]

procs = []
for i in range(N_SHARDS):
    skip = i * CHUNK
    limit = min(CHUNK, IMAGES - skip)
    out_csv = ROOT / f"artifacts/raw_metrics/deepglobe_planning_tau2_shard{i}.csv"
    log = ROOT / f"artifacts/logs/deepglobe_planning_tau2_shard{i}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [*BASE, "--skip", str(skip), "--limit", str(limit),
           "--output-csv", str(out_csv)]
    print(f"launching shard {i}: skip={skip} limit={limit} -> {out_csv.name}", flush=True)
    with log.open("w") as fh:
        procs.append((i, subprocess.Popen(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT)))

failed = []
for i, p in procs:
    rc = p.wait()
    print(f"shard {i} exit={rc}", flush=True)
    if rc != 0:
        failed.append(i)
        log = ROOT / f"artifacts/logs/deepglobe_planning_tau2_shard{i}.log"
        print(f"  --- shard {i} log tail ---", flush=True)
        print("\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]), flush=True)

if failed:
    raise RuntimeError(f"DeepGlobe planning shards failed: {failed}")

# ---- merge shards ----
merged_path = ROOT / "artifacts/raw_metrics/deepglobe_planning_tau2.csv"
fieldnames = None
rows = 0
with merged_path.open("w", newline="", encoding="utf-8") as out:
    writer = None
    for i in range(N_SHARDS):
        shard = ROOT / f"artifacts/raw_metrics/deepglobe_planning_tau2_shard{i}.csv"
        if not shard.exists():
            continue
        with shard.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = reader.fieldnames
                writer = csv.DictWriter(out, fieldnames=fieldnames)
                writer.writeheader()
            for row in reader:
                writer.writerow(row)
                rows += 1
print(f"\nmerged {rows} rows -> {merged_path}", flush=True)

# ---- quick summary over merged ----
import numpy as np
with merged_path.open("r", encoding="utf-8") as f:
    all_rows = list(csv.DictReader(f))
methods = sorted({r["method"] for r in all_rows})
print(f"\n=== DeepGlobe planning summary ({rows} rows) ===")
print(f"{'method':<24} {'success':>8} {'off_road':>9} {'detour_med':>11} {'n':>6}")
for m in methods:
    rs = [r for r in all_rows if r["method"] == m]
    succ = np.mean([int(r["success"]) for r in rs])
    offs = [float(r["off_road_ratio"]) for r in rs if r["off_road_ratio"] not in ("nan", "")]
    dets = [float(r["relative_detour"]) for r in rs if r["relative_detour"] not in ("nan", "")]
    print(f"{m:<24} {succ:8.3f} {(np.mean(offs) if offs else float('nan')):9.4f} "
          f"{(np.median(dets) if dets else float('nan')):11.4f} {len(rs):6d}")
