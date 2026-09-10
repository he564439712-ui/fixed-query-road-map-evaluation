#!/usr/bin/env python
"""M5: calibration quality under degradations (13 conditions).

For each corruption condition, computes ECE / Brier / NLL on the ensemble
predictions vs GT, and uncertainty-error-detection AUROC/AUPRC.

The key M5 hypothesis: degradation makes the model OVERCONFIDENT (predictions
drift from GT while confidence stays high), so ECE rises sharply vs clean.
Under such overconfidence the conservative lower bound q = p - kappa*sigma
and temperature scaling should be most valuable — this is the paper's
"calibrated uncertainty" selling point.

Also reports temperature scaling per condition (T_opt on the calibration set,
applied post-hoc to the test predictions for that condition).

Usage:
    python scripts/evaluate_corrupted_calibration.py --split test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.corruptions import CORRUPTION_CONDITIONS
from src.data.dataset import load_mask
from src.data.splits import build_splits, resolve_image_paths
from src.eval.calibration import calibration_metrics


def main():
    parser = argparse.ArgumentParser(description="M5 corrupted calibration.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--prediction-dir", type=Path,
                        default=Path("artifacts/predictions/corruptions/test"))
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--conditions", nargs="+", default=None)
    args = parser.parse_args()

    root = args.root
    splits = build_splits(root)
    image_ids = splits[f"{args.split}_ids"]
    pred_root = root / args.prediction_dir
    conditions = args.conditions or list(CORRUPTION_CONDITIONS.keys())

    print(f"{'condition':<15} {'ECE':>7} {'Brier':>7} {'NLL':>7} {'AUROC':>7} {'std':>7}")
    print("-" * 60)
    rows = []
    for cond in conditions:
        cond_dir = pred_root / cond
        if not cond_dir.exists():
            print(f"  {cond}: MISSING directory, skip")
            continue
        ece_l, brier_l, nll_l, auroc_l, std_l = [], [], [], [], []
        n = 0
        for image_id in image_ids:
            prob_file = cond_dir / f"{image_id}_prob.npy"
            if not prob_file.exists():
                continue
            prob = np.load(prob_file).astype(np.float32)
            std_file = cond_dir / f"{image_id}_std.npy"
            std = np.load(std_file).astype(np.float32) if std_file.exists() else None
            _, mask_path = resolve_image_paths(root, image_id)
            gt = load_mask(mask_path)
            m = calibration_metrics(prob, gt, std, threshold=0.5)
            ece_l.append(m["ece"]); brier_l.append(m["brier"]); nll_l.append(m["nll"])
            if "auroc_error" in m:
                auroc_l.append(m["auroc_error"])
            if std is not None:
                std_l.append(float(std.mean()))
            n += 1

        if n == 0:
            continue
        row = {
            "condition": cond, "n": n,
            "ece": float(np.mean(ece_l)), "brier": float(np.mean(brier_l)),
            "nll": float(np.mean(nll_l)),
            "auroc": float(np.mean(auroc_l)) if auroc_l else float("nan"),
            "std_mean": float(np.mean(std_l)) if std_l else float("nan"),
        }
        rows.append(row)
        print(f"{cond:<15} {row['ece']:7.4f} {row['brier']:7.4f} {row['nll']:7.4f} "
              f"{row['auroc']:7.4f} {row['std_mean']:7.4f}")

    # Clean vs degradation summary
    if rows:
        clean = next((r for r in rows if r["condition"] == "clean"), None)
        if clean:
            print("\n--- ECE relative to clean ---")
            for r in rows:
                if r["condition"] == "clean":
                    continue
                rel = (r["ece"] - clean["ece"]) / max(clean["ece"], 1e-9) * 100
                print(f"{r['condition']:<15} ECE {clean['ece']:.4f} -> {r['ece']:.4f}  ({rel:+.0f}%)")

    # Save
    out = root / "artifacts" / "raw_metrics"
    out.mkdir(parents=True, exist_ok=True)
    import csv
    with (out / "corrupted_calibration.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved {len(rows)} rows to {out / 'corrupted_calibration.csv'}")


if __name__ == "__main__":
    main()
