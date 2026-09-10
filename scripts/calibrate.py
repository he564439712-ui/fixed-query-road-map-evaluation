#!/usr/bin/env python
"""Temperature scaling calibration on the 110-image calibration set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.splits import build_splits, resolve_image_paths
from src.data.dataset import load_mask
from src.models.calibration import TemperatureScaler


def main():
    parser = argparse.ArgumentParser(description="Calibrate model probabilities.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--prediction-dir", type=Path, required=True,
                        help="Directory containing _logits.npy or _prob.npy files")
    parser.add_argument("--use-logits", action="store_true",
                        help="Inputs are logits (pre-sigmoid)")
    parser.add_argument("--init-temp", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("artifacts/checkpoints/temperature.json"))
    args = parser.parse_args()

    splits = build_splits(args.root)
    cal_ids = splits["calibration_ids"]
    print(f"Calibrating on {len(cal_ids)} images")

    # Load predictions and labels
    logits_list = []
    prob_list = []
    label_list = []

    if args.use_logits:
        pred_dir = args.root / args.prediction_dir
        for image_id in cal_ids:
            logits_file = pred_dir / f"{image_id}_logits.npy"
            if not logits_file.exists():
                print(f"WARNING: missing {logits_file}, skipping {image_id}")
                continue
            logits = np.load(logits_file)
            _, mask_path = resolve_image_paths(args.root, image_id)
            label = load_mask(mask_path)
            logits_list.append(logits)
            label_list.append(label)
        print(f"Loaded {len(logits_list)} logit-mask pairs")
    else:
        pred_dir = args.root / args.prediction_dir
        for image_id in cal_ids:
            prob_file = pred_dir / f"{image_id}_prob.npy"
            if not prob_file.exists():
                print(f"WARNING: missing {prob_file}, skipping {image_id}")
                continue
            prob = np.load(prob_file)
            _, mask_path = resolve_image_paths(args.root, image_id)
            label = load_mask(mask_path)
            prob_list.append(prob)
            label_list.append(label)
        # Convert probs to logits for temperature scaling
        eps = 1e-9
        for prob in prob_list:
            p = np.clip(prob, eps, 1.0 - eps)
            logits_list.append(np.log(p / (1.0 - p)))
        print(f"Loaded {len(logits_list)} prob-mask pairs (converted to logits)")

    # Optimize temperature
    scaler = TemperatureScaler(init_temperature=args.init_temp)
    T = scaler.calibrate(logits_list, label_list)
    print(f"Optimized temperature: T = {T:.4f}")

    # Save
    output_path = args.root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump({"temperature": T, "init_temperature": args.init_temp}, f, indent=2)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
