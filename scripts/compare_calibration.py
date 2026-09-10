#!/usr/bin/env python
"""M2 summary table: single-model vs Deep Ensemble vs +temperature scaling
calibration quality, plus uncertainty-error detection.

Produces the comparison table that drives the M2 claims:
  ECE / Brier / NLL per member, for the ensemble, and after temperature
  scaling; relative reduction of ensemble vs average single model; and the
  epistemic-std error-detection AUROC/AUPRC.

Usage:
    python scripts/compare_calibration.py \
        --prediction-dir artifacts/predictions/test \
        --temperature artifacts/checkpoints/temperature.json \
        --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import load_mask
from src.data.splits import build_splits, resolve_image_paths
from src.eval.calibration import calibration_metrics
from scripts.evaluate_calibration import apply_temperature


def agg_metrics(pred_dir: Path, root: Path, image_ids: list[str], T: float):
    """Aggregate calibration metrics over images (with optional temperature)."""
    ece, brier, nll = [], [], []
    for image_id in image_ids:
        prob_file = pred_dir / f"{image_id}_prob.npy"
        if not prob_file.exists():
            continue
        prob = np.load(prob_file).astype(np.float32)
        if T != 1.0:
            prob = apply_temperature(prob, T)
        _, mask_path = resolve_image_paths(root, image_id)
        gt = load_mask(mask_path)
        m = calibration_metrics(prob, gt, None)
        ece.append(m["ece"]); brier.append(m["brier"]); nll.append(m["nll"])
    n = len(ece)
    return {
        "n": n,
        "ece": float(np.mean(ece)),
        "brier": float(np.mean(brier)),
        "nll": float(np.mean(nll)),
    }


def main():
    parser = argparse.ArgumentParser(description="M2 calibration comparison table.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--prediction-dir", type=Path, default=Path("artifacts/predictions/test"))
    parser.add_argument("--temperature", type=Path, default=Path("artifacts/checkpoints/temperature.json"))
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    splits = build_splits(args.root)
    image_ids = splits[f"{args.split}_ids"] if args.split != "val" else splits["val_ids"]
    pred_dir = args.root / args.prediction_dir

    temp_path = args.root / args.temperature
    T = float(json.loads(temp_path.read_text())["temperature"]) if temp_path.exists() else 1.0
    print(f"Temperature T = {T:.4f}\n")

    # Members
    member_results = []
    for i in range(3):
        r = agg_metrics(pred_dir / f"member{i}", args.root, image_ids, 1.0)
        member_results.append(r)
        print(f"member{i}  : ECE={r['ece']:.4f}  Brier={r['brier']:.4f}  NLL={r['nll']:.4f}")

    avg_single = {
        "ece": float(np.mean([r["ece"] for r in member_results])),
        "brier": float(np.mean([r["brier"] for r in member_results])),
        "nll": float(np.mean([r["nll"] for r in member_results])),
    }
    print(f"avg single: ECE={avg_single['ece']:.4f}  Brier={avg_single['brier']:.4f}  NLL={avg_single['nll']:.4f}")

    ens = agg_metrics(pred_dir, args.root, image_ids, 1.0)
    ens_ts = agg_metrics(pred_dir, args.root, image_ids, T)
    print(f"ensemble  : ECE={ens['ece']:.4f}  Brier={ens['brier']:.4f}  NLL={ens['nll']:.4f}")
    print(f"ensemble+T: ECE={ens_ts['ece']:.4f}  Brier={ens_ts['brier']:.4f}  NLL={ens_ts['nll']:.4f}")

    def rel(a, b):
        return (a - b) / max(a, 1e-9) * 100.0

    print(f"\nReductions:")
    print(f"  ensemble vs avg single:  ECE {rel(avg_single['ece'], ens['ece']):+.1f}%  "
          f"Brier {rel(avg_single['brier'], ens['brier']):+.1f}%  NLL {rel(avg_single['nll'], ens['nll']):+.1f}%")
    print(f"  +TS vs ensemble:         ECE {rel(ens['ece'], ens_ts['ece']):+.1f}%  "
          f"Brier {rel(ens['brier'], ens_ts['brier']):+.1f}%  NLL {rel(ens['nll'], ens_ts['nll']):+.1f}%")

    # Uncertainty error detection on the ensemble
    aurocs, auprcs = [], []
    for image_id in image_ids:
        prob_file = pred_dir / f"{image_id}_prob.npy"
        std_file = pred_dir / f"{image_id}_std.npy"
        if not (prob_file.exists() and std_file.exists()):
            continue
        prob = np.load(prob_file).astype(np.float32)
        std = np.load(std_file).astype(np.float32)
        _, mask_path = resolve_image_paths(args.root, image_id)
        gt = load_mask(mask_path)
        m = calibration_metrics(prob, gt, std)
        if "auroc_error" in m:
            aurocs.append(m["auroc_error"]); auprcs.append(m["auprc_error"])
    if aurocs:
        print(f"\nUncertainty error detection (epistemic std):")
        print(f"  AUROC = {np.mean(aurocs):.4f}   AUPRC = {np.mean(auprcs):.4f}")


if __name__ == "__main__":
    main()
