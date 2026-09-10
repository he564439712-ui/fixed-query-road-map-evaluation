#!/usr/bin/env python
"""Segmentation + topology metrics on the Massachusetts Roads test set (49 images)
from saved ensemble predictions. CPU-only, no inference rerun needed.

Inputs (already on disk):
  artifacts/predictions/test/{id}_prob.npy   ensemble mean probability
  artifacts/predictions/test/{id}_std.npy    epistemic std
  tiff/test_labels/{id}.tif                  ground-truth mask
  artifacts/queries/queries.csv              945 fixed start--goal pairs

Outputs:
  artifacts/raw_metrics/massachusetts_test_seg.csv
  artifacts/raw_metrics/massachusetts_test_seg_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import load_mask
from src.data.queries import get_queries_for_image, load_queries
from src.data.splits import build_splits, resolve_image_paths
from src.eval.segmentation import (
    compute_cldice,
    compute_dice,
    compute_iou,
    compute_precision,
    compute_recall,
    segmentation_metrics,
)
from src.planning.costs import conservative_confidence_lower_bound
from src.planning.topology_repair import repair_topology

CONNECTIVITY_STRUCT = np.ones((3, 3), dtype=np.uint8)
METRICS = ("dice", "iou", "precision", "recall", "cldice", "apls")


def bootstrap_interval(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        means[index] = values[rng.integers(0, n, size=n)].mean()
    return tuple(float(value) for value in np.percentile(means, (2.5, 97.5)))


def connectivity_ratio(mask: np.ndarray, queries: list[dict]) -> float:
    """Fraction of the image's fixed queries whose endpoints lie in the same
    connected component of ``mask`` (endpoint off the mask counts as
    disconnected). This is a pure-reachability topology metric, independent of
    the A* search cap used by planning success."""
    if not queries:
        return float("nan")
    labels, _ = ndimage.label(mask, structure=CONNECTIVITY_STRUCT)
    connected = 0
    for q in queries:
        s = labels[q["start_row"], q["start_col"]]
        g = labels[q["goal_row"], q["goal_col"]]
        if s > 0 and g > 0 and s == g:
            connected += 1
    return connected / len(queries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--prediction-dir", type=Path, default=Path("artifacts/predictions/test"))
    parser.add_argument("--queries", type=Path, default=Path("artifacts/queries/queries.csv"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--tau-gate", "--tau", dest="tau_gate", type=float, default=2.0)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-apls",
        action="store_true",
        help="Skip the legacy APLS-inspired approximation when only Dice, clDice, and connectivity are needed.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/raw_metrics"))
    args = parser.parse_args()

    root = args.root.resolve()
    pred_dir = args.prediction_dir if args.prediction_dir.is_absolute() else root / args.prediction_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = build_splits(root)
    test_ids = splits["test_ids"]
    queries = load_queries(args.queries if args.queries.is_absolute() else root / args.queries)
    queries_by_img = {img: get_queries_for_image(queries, img) for img in test_ids}

    rows: list[dict] = []
    conn_orig_sum = 0.0
    conn_repaired_sum = 0.0
    n_queries_total = 0

    print(f"Massachusetts test: {len(test_ids)} images, {len(queries)} queries "
          f"(kappa={args.kappa}, tau_gate={args.tau_gate})", flush=True)

    for i, image_id in enumerate(test_ids, start=1):
        prob = np.load(pred_dir / f"{image_id}_prob.npy").astype(np.float32)
        std = np.load(pred_dir / f"{image_id}_std.npy").astype(np.float32)
        q = conservative_confidence_lower_bound(prob, std, kappa=args.kappa)
        gt = load_mask(resolve_image_paths(root, image_id)[1]).astype(bool)
        mask = prob >= args.threshold
        repaired, bridges, _ = repair_topology(mask, q, tau_gate=args.tau_gate, w_p=1.0)

        if args.skip_apls:
            m = {
                "dice": compute_dice(mask, gt),
                "iou": compute_iou(mask, gt),
                "precision": compute_precision(mask, gt),
                "recall": compute_recall(mask, gt),
                "cldice": compute_cldice(mask, gt)[0],
                "apls": float("nan"),
            }
        else:
            m = segmentation_metrics(prob, gt, args.threshold)
        dice_repaired = compute_dice(repaired, gt)
        cldice_repaired = compute_cldice(repaired, gt)[0]

        _, n_comp_orig = ndimage.label(mask, structure=CONNECTIVITY_STRUCT)
        _, n_comp_repaired = ndimage.label(repaired, structure=CONNECTIVITY_STRUCT)

        qs = queries_by_img.get(image_id, [])
        co = connectivity_ratio(mask, qs)
        cr = connectivity_ratio(repaired, qs)
        conn_orig_sum += co * len(qs)
        conn_repaired_sum += cr * len(qs)
        n_queries_total += len(qs)

        rows.append({
            "image_id": image_id,
            **{k: m[k] for k in METRICS},
            "dice_repaired": dice_repaired,
            "cldice_repaired": cldice_repaired,
            "components_orig": int(n_comp_orig),
            "components_repaired": int(n_comp_repaired),
            "component_reduction": int(n_comp_orig) - int(n_comp_repaired),
            "num_bridges": len(bridges),
            "connectivity_orig": co,
            "connectivity_repaired": cr,
        })
        if i == 1 or i % 10 == 0 or i == len(test_ids):
            print(f"[{i}/{len(test_ids)}] {image_id}: dice={m['dice']:.4f} "
                  f"cldice={m['cldice']:.4f} comp {n_comp_orig}->{n_comp_repaired} "
                  f"bridges={len(bridges)} conn {co:.3f}->{cr:.3f}", flush=True)

    # ---- summary with bootstrap CIs over the 49 per-image values ----
    summary: dict = {
        "dataset": "Massachusetts Roads",
        "split": "test (49 images)",
        "n_images": len(rows),
        "n_queries": n_queries_total,
        "kappa": args.kappa,
        "tau_gate": args.tau_gate,
        "threshold": args.threshold,
        "bootstrap_samples": args.bootstrap_samples,
        "metrics": {},
        "connectivity": {
            "original": {"connected": int(conn_orig_sum), "total": n_queries_total,
                         "ratio": conn_orig_sum / n_queries_total},
            "repaired": {"connected": int(conn_repaired_sum), "total": n_queries_total,
                         "ratio": conn_repaired_sum / n_queries_total},
        },
    }
    per_image_fields = [
        "dice", "iou", "precision", "recall", "cldice", "apls",
        "dice_repaired",
        "cldice_repaired", "components_orig", "components_repaired",
        "component_reduction", "num_bridges",
    ]
    for mi, field in enumerate(per_image_fields):
        values = np.asarray([float(r[field]) for r in rows])
        lower, upper = bootstrap_interval(values, args.bootstrap_samples, args.seed + mi)
        summary["metrics"][field] = {"mean": float(values.mean()), "ci95": [lower, upper]}

    csv_path = output_dir / "massachusetts_test_seg.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = output_dir / "massachusetts_test_seg_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Summary (per-image mean, 95% bootstrap CI) ===")
    for field in per_image_fields:
        v = summary["metrics"][field]
        print(f"  {field:22s} {v['mean']:.4f}  [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}]")
    c = summary["connectivity"]
    print(f"  connectivity orig→repaired: {c['original']['ratio']:.4f} -> "
          f"{c['repaired']['ratio']:.4f}  ({c['original']['connected']}/{c['original']['total']} "
          f"-> {c['repaired']['connected']}/{c['repaired']['total']})")
    print(f"Saved: {csv_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
