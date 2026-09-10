#!/usr/bin/env python
"""M2 calibration evaluation: compare calibration quality before/after
temperature scaling, and uncertainty-error detection quality.

Protocol:
  - Temperature T is optimized on the CALIBRATION split (scripts/calibrate.py).
  - Calibration metrics are reported on the TEST split (post-hoc), comparing:
      * before: raw ensemble mean probability
      * after : temperature-scaled probability (logit/T -> sigmoid)

Usage:
    python scripts/evaluate_calibration.py \
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
from src.eval.calibration import (
    calibration_metrics,
    expected_calibration_error,
)


def apply_temperature(prob: np.ndarray, T: float, eps: float = 1e-9) -> np.ndarray:
    """Temperature-scale a probability map: logit/T -> sigmoid."""
    p = np.clip(prob, eps, 1.0 - eps)
    logit = np.log(p / (1.0 - p))
    scaled = logit / max(T, 1e-6)
    return (1.0 / (1.0 + np.exp(-scaled))).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Evaluate calibration quality.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--prediction-dir", type=Path,
                        default=Path("artifacts/predictions/test"))
    parser.add_argument("--temperature", type=Path,
                        default=Path("artifacts/checkpoints/temperature.json"))
    parser.add_argument("--split", default="test",
                        choices=["calibration", "test", "val"])
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    splits = build_splits(args.root)
    image_ids = splits[f"{args.split}_ids"] if args.split != "val" else splits["val_ids"]
    pred_dir = args.root / args.prediction_dir

    # Load temperature
    temp_path = args.root / args.temperature
    if temp_path.exists():
        T = float(json.loads(temp_path.read_text())["temperature"])
        print(f"Temperature T = {T:.4f} (from {temp_path})")
    else:
        T = 1.0
        print(f"WARNING: {temp_path} not found — using T=1.0 (no scaling)")

    pre_ece, post_ece = [], []
    pre_brier, post_brier = [], []
    pre_nll, post_nll = [], []
    aurocs, auprcs = [], []

    for image_id in image_ids:
        prob_file = pred_dir / f"{image_id}_prob.npy"
        std_file = pred_dir / f"{image_id}_std.npy"
        if not prob_file.exists():
            print(f"  WARNING: missing prediction for {image_id}, skipping")
            continue
        prob = np.load(prob_file).astype(np.float32)
        std = np.load(std_file).astype(np.float32) if std_file.exists() else None
        _, mask_path = resolve_image_paths(args.root, image_id)
        gt = load_mask(mask_path)

        # Before temperature scaling
        m_pre = calibration_metrics(prob, gt, std, threshold=args.threshold)
        # After temperature scaling
        prob_cal = apply_temperature(prob, T)
        m_post = calibration_metrics(prob_cal, gt, None, threshold=args.threshold)

        pre_ece.append(m_pre["ece"]); pre_brier.append(m_pre["brier"]); pre_nll.append(m_pre["nll"])
        post_ece.append(m_post["ece"]); post_brier.append(m_post["brier"]); post_nll.append(m_post["nll"])
        if "auroc_error" in m_pre:
            aurocs.append(m_pre["auroc_error"]); auprcs.append(m_pre["auprc_error"])

    # Aggregate (mean over images)
    n = len(pre_ece)
    if n == 0:
        print("No predictions found.")
        return

    def agg(xs):
        return float(np.mean(xs))

    pre_ece_m, post_ece_m = agg(pre_ece), agg(post_ece)
    ece_rel = (pre_ece_m - post_ece_m) / max(pre_ece_m, 1e-9) * 100.0

    print(f"\n=== Calibration evaluation on '{args.split}' (n={n} images) ===")
    print(f"               before TS   after TS    Δ")
    print(f"ECE  :         {pre_ece_m:.4f}    {post_ece_m:.4f}    {ece_rel:+.1f}%")
    print(f"Brier:         {agg(pre_brier):.4f}    {agg(post_brier):.4f}")
    print(f"NLL  :         {agg(pre_nll):.4f}    {agg(post_nll):.4f}")
    if aurocs:
        print(f"\nUncertainty error detection (epistemic std):")
        print(f"  AUROC = {agg(aurocs):.4f}   AUPRC = {agg(auprcs):.4f}")
    print(f"\nTarget: ECE relative reduction >= 20%")
    print(f"  Achieved: {ece_rel:.1f}%")


if __name__ == "__main__":
    main()
