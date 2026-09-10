#!/usr/bin/env python
"""M3 topology repair evaluation on the TEST split.

For each image: threshold ensemble probability -> repair_topology with the
conservative lower bound q = clip(p - kappa*std, eps, 1-eps) as the confidence
map -> evaluate bridge quality and APLS improvement vs GT.

Usage:
    python scripts/evaluate_topology.py \
        --prediction-dir artifacts/predictions/test \
        --split test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import load_mask
from src.data.splits import build_splits, resolve_image_paths
from src.eval.topology import topology_metrics
from src.planning.costs import conservative_confidence_lower_bound
from src.planning.topology_repair import repair_topology


def main():
    parser = argparse.ArgumentParser(description="Evaluate topology repair.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--prediction-dir", type=Path, default=Path("artifacts/predictions/test"))
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-radius", type=float, default=50.0)
    parser.add_argument("--tau-gate", "--tau", dest="tau_gate", type=float, default=2.0)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--w-p", type=float, default=1.0)
    parser.add_argument("--w-l", type=float, default=0.5)
    parser.add_argument("--w-theta", type=float, default=1.0)
    parser.add_argument("--w-w", type=float, default=0.5)
    parser.add_argument("--w-g", type=float, default=1.0)
    args = parser.parse_args()

    splits = build_splits(args.root)
    image_ids = splits[f"{args.split}_ids"]
    pred_dir = args.root / args.prediction_dir

    metrics_list = []
    n_skipped = 0
    for i, image_id in enumerate(image_ids):
        prob_file = pred_dir / f"{image_id}_prob.npy"
        std_file = pred_dir / f"{image_id}_std.npy"
        if not prob_file.exists():
            n_skipped += 1
            print(f"[{i+1}/{len(image_ids)}] {image_id}: MISSING, skip")
            continue
        prob = np.load(prob_file).astype(np.float32)
        std = np.load(std_file).astype(np.float32) if std_file.exists() else np.zeros_like(prob)
        _, mask_path = resolve_image_paths(args.root, image_id)
        gt = load_mask(mask_path).astype(bool)

        q = conservative_confidence_lower_bound(prob, std, args.kappa)
        binary = prob >= args.threshold
        repaired, bridges, skeleton = repair_topology(
            binary, q,
            max_radius=args.max_radius, tau_gate=args.tau_gate,
            w_p=args.w_p, w_l=args.w_l, w_theta=args.w_theta,
            w_w=args.w_w, w_g=args.w_g,
        )
        m = topology_metrics(binary, repaired, bridges, gt)
        m["image_id"] = image_id
        metrics_list.append(m)
        print(f"[{i+1}/{len(image_ids)}] {image_id}: "
              f"bridges={m['num_bridges']:.0f} prec={m['bridge_precision']:.2f} "
              f"apls {m['apls_original']:.3f}->{m['apls_repaired']:.3f} "
              f"comps {m['original_components']:.0f}->{m['repaired_components']:.0f}")

    if not metrics_list:
        print("No predictions found.")
        return

    def avg(key):
        return float(np.nanmean([m[key] for m in metrics_list]))

    print(f"\n=== Topology repair on '{args.split}' (n={len(metrics_list)}, skipped={n_skipped}) ===")
    print(f"Bridges per image      : {avg('num_bridges'):.1f}")
    print(f"Bridge precision       : {avg('bridge_precision'):.4f}  (target >= 0.80)")
    print(f"Correct bridges/image  : {avg('correct_bridges'):.1f}")
    print(f"Component reduction    : {avg('component_reduction'):.1f}")
    print(f"APLS original          : {avg('apls_original'):.4f}")
    print(f"APLS repaired          : {avg('apls_repaired'):.4f}")
    print(f"APLS improvement       : {avg('apls_improvement'):+.4f}  (target > 0)")


if __name__ == "__main__":
    main()
