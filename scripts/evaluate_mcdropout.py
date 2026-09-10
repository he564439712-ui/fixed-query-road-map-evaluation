#!/usr/bin/env python
"""MC Dropout calibration evaluation.

Compares, on the TEST split:
  - single forward pass (dropout off) of the dropout-trained model
  - MC-averaged prediction (T stochastic passes)
  - MC-averaged + temperature scaling
Plus epistemic uncertainty error-detection AUROC/AUPRC from MC std.

Usage:
    python scripts/evaluate_mcdropout.py \
        --prediction-dir artifacts/predictions/mcdropout/test \
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


def agg(pred_dir: Path, root: Path, image_ids: list[str], key: str, T: float = 1.0):
    ece, brier, nll = [], [], []
    for image_id in image_ids:
        f = pred_dir / f"{image_id}_{key}.npy"
        if not f.exists():
            continue
        prob = np.load(f).astype(np.float32)
        if T != 1.0:
            prob = apply_temperature(prob, T)
        _, mask_path = resolve_image_paths(root, image_id)
        gt = load_mask(mask_path)
        m = calibration_metrics(prob, gt, None)
        ece.append(m["ece"]); brier.append(m["brier"]); nll.append(m["nll"])
    return (float(np.mean(ece)), float(np.mean(brier)), float(np.mean(nll)))


def main():
    parser = argparse.ArgumentParser(description="MC Dropout calibration.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--prediction-dir", type=Path, default=Path("artifacts/predictions/mcdropout/test"))
    parser.add_argument("--temperature", type=Path, default=Path("artifacts/checkpoints/temperature.json"))
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    splits = build_splits(args.root)
    image_ids = splits[f"{args.split}_ids"] if args.split != "val" else splits["val_ids"]
    pred_dir = args.root / args.prediction_dir

    temp_path = args.root / args.temperature
    T = float(json.loads(temp_path.read_text())["temperature"]) if temp_path.exists() else 1.0
    print(f"Temperature T = {T:.4f}\n")

    single = agg(pred_dir, args.root, image_ids, "single")
    mc = agg(pred_dir, args.root, image_ids, "prob")
    mc_ts = agg(pred_dir, args.root, image_ids, "prob", T)
    print("                 ECE      Brier    NLL")
    print(f"single pass :    {single[0]:.4f}   {single[1]:.4f}   {single[2]:.4f}")
    print(f"MC averaged :    {mc[0]:.4f}   {mc[1]:.4f}   {mc[2]:.4f}")
    print(f"MC + TS     :    {mc_ts[0]:.4f}   {mc_ts[1]:.4f}   {mc_ts[2]:.4f}")

    def rel(a, b):
        return (a - b) / max(a, 1e-9) * 100.0

    print(f"\nReductions:")
    print(f"  MC vs single:      ECE {rel(single[0], mc[0]):+.1f}%  Brier {rel(single[1], mc[1]):+.1f}%  NLL {rel(single[2], mc[2]):+.1f}%")
    print(f"  MC+TS vs single:   ECE {rel(single[0], mc_ts[0]):+.1f}%  Brier {rel(single[1], mc_ts[1]):+.1f}%  NLL {rel(single[2], mc_ts[2]):+.1f}%")

    # Uncertainty error detection (MC std)
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
        print(f"\nMC uncertainty error detection:")
        print(f"  AUROC = {np.mean(aurocs):.4f}   AUPRC = {np.mean(auprcs):.4f}")


if __name__ == "__main__":
    main()
