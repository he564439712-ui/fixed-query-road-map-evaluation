#!/usr/bin/env python
"""M5: per-condition temperature scaling under degradations.

For each corruption condition, optimize a temperature T_c on the CALIBRATION
split (held-out), then apply it to the TEST split predictions and measure the
calibration improvement.

The M5 hypothesis: on clean data T~1 (model already calibrated, TS marginal).
Under degradation the model becomes overconfident (ECE jumps, e.g. +169% for
blur_medium), so T_c >> 1 and temperature scaling should recover most of the
ECE — this is the "calibrated uncertainty" contribution the paper claims.

Usage:
    python scripts/calibrate_corrupted.py --conditions blur_medium
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.corruptions import CORRUPTION_CONDITIONS
from src.data.dataset import load_mask
from src.data.splits import build_splits, resolve_image_paths
from src.eval.calibration import calibration_metrics, expected_calibration_error
from src.models.calibration import TemperatureScaler


def apply_temperature(prob: np.ndarray, T: float, eps: float = 1e-9) -> np.ndarray:
    p = np.clip(prob, eps, 1.0 - eps)
    logit = np.log(p / (1.0 - p))
    scaled = logit / max(T, 1e-6)
    return (1.0 / (1.0 + np.exp(-scaled))).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="M5 per-condition temperature scaling.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--prediction-dir", type=Path,
                        default=Path("artifacts/predictions/corruptions"))
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--output", type=Path, default=Path("artifacts/raw_metrics/corrupted_temperature.csv"))
    parser.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    args = parser.parse_args()

    root = args.root
    splits = build_splits(root)
    test_ids = splits["test_ids"]
    cal_ids = splits["calibration_ids"]
    conditions = args.conditions or list(CORRUPTION_CONDITIONS.keys())
    pred_root = root / args.prediction_dir

    print(f"{'condition':<16} {'T_opt':>7} {'ECE_before':>11} {'ECE_after':>10} "
          f"{'rel':>7} {'NLL_before':>11} {'NLL_after':>10}")
    print("-" * 80)
    rows = []
    for cond in conditions:
        # --- optimize T on calibration split (held-out) ---
        # Use ALL pixels + grid search over T. LBFGS on a subsample is biased
        # (stratified sampling changed the NLL surface so T=1.0 won even
        # though the full-pixel optimum for blur_medium is ~1.5); random
        # subsampling is background-dominated (clean gave T=1.41 vs the true
        # ~1.0). Grid search over the full pixel set is exact and cheap.
        cal_flat_p, cal_flat_g = [], []
        cal_dir = pred_root / "calibration" / cond
        if not cal_dir.exists():
            print(f"  {cond}: no calibration predictions, skip")
            continue
        for image_id in cal_ids:
            prob_file = cal_dir / f"{image_id}_prob.npy"
            if not prob_file.exists():
                continue
            p = np.load(prob_file).astype(np.float32)
            _, mask_path = resolve_image_paths(root, image_id)
            gt = load_mask(mask_path)
            cal_flat_p.append(p.reshape(-1).astype(np.float64))
            cal_flat_g.append(gt.reshape(-1).astype(np.float64))
        if not cal_flat_p:
            print(f"  {cond}: no calibration predictions, skip")
            continue
        P = np.concatenate(cal_flat_p)
        G = np.concatenate(cal_flat_g)

        def nll_at(T):
            p = np.clip(P, 1e-12, 1.0 - 1e-12)
            logit = np.log(p / (1.0 - p))
            pc = np.clip(1.0 / (1.0 + np.exp(-logit / T)), 1e-12, 1.0 - 1e-12)
            return float(-(G * np.log(pc) + (1.0 - G) * np.log(1.0 - pc)).mean())

        cands = [0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 1.8, 2.2]
        best_T, best_nll = 1.0, nll_at(1.0)
        for tc in cands:
            n = nll_at(tc)
            if n < best_nll:
                best_nll, best_T = n, tc
        T = best_T

        # --- evaluate on test split: before vs after ---
        ece_b, ece_a, nll_b, nll_a, brier_b, brier_a = [], [], [], [], [], []
        test_dir = pred_root / "test" / cond
        for image_id in test_ids:
            prob_file = test_dir / f"{image_id}_prob.npy"
            if not prob_file.exists():
                continue
            p = np.load(prob_file).astype(np.float32)
            _, mask_path = resolve_image_paths(root, image_id)
            gt = load_mask(mask_path)
            p_flat = p.flatten().astype(np.float64)
            gt_flat = gt.flatten().astype(np.float64)
            m_b = calibration_metrics(p, gt)
            p_ts = apply_temperature(p, T)
            m_a = calibration_metrics(p_ts, gt)
            ece_b.append(m_b["ece"]); ece_a.append(m_a["ece"])
            nll_b.append(m_b["nll"]); nll_a.append(m_a["nll"])
            brier_b.append(m_b["brier"]); brier_a.append(m_a["brier"])

        n = len(ece_b)
        if n == 0:
            continue
        eb, ea = float(np.mean(ece_b)), float(np.mean(ece_a))
        nb_, na_ = float(np.mean(nll_b)), float(np.mean(nll_a))
        rel = (ea - eb) / max(eb, 1e-9) * 100
        rows.append({"condition": cond, "T": T, "n_images": n,
                     "ece_before": eb, "ece_after": ea, "ece_rel_pct": rel,
                     "nll_before": nb_, "nll_after": na_,
                     "brier_before": float(np.mean(brier_b)), "brier_after": float(np.mean(brier_a))})
        print(f"{cond:<16} {T:7.3f} {eb:11.4f} {ea:10.4f} {rel:6.1f}% {nb_:11.4f} {na_:10.4f}")

    out = root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
