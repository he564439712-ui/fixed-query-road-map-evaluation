#!/usr/bin/env python
"""Evaluate a DeepGlobe road-segmentation checkpoint on its locked test split."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import load_image, load_mask
from src.data.deepglobe import build_deepglobe_splits, resolve_deepglobe_paths
from src.data.tiling import predict_full_image
from src.eval.segmentation import (
    compute_cldice,
    compute_dice,
    compute_iou,
    compute_precision,
    compute_recall,
)
from src.models.segmentation import create_segmentation_model


METRICS = ("dice", "iou", "precision", "recall", "cldice")


def bootstrap_interval(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        means[index] = values[rng.integers(0, n, size=n)].mean()
    return tuple(float(value) for value in np.percentile(means, (2.5, 97.5)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", default="unet", choices=("unet", "deeplabv3_resnet50"))
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/raw_metrics"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    root = args.root.resolve()
    device = torch.device(args.device)
    checkpoint_path = args.checkpoint if args.checkpoint.is_absolute() else root / args.checkpoint
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model = create_segmentation_model(args.model, base_channels=args.base_channels)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    image_ids = build_deepglobe_splits(root, seed=args.seed)["test_ids"]
    print(f"Device: {device}; model: {args.model}; test images: {len(image_ids)}", flush=True)
    print(f"Checkpoint: {checkpoint_path}", flush=True)

    rows: list[dict[str, float | str]] = []
    for index, image_id in enumerate(image_ids, start=1):
        image_path, mask_path = resolve_deepglobe_paths(root, image_id)
        image = load_image(image_path, as_tensor=True).unsqueeze(0)
        probability = predict_full_image(
            model, image, tile_size=args.tile_size, overlap=args.overlap,
            batch_size=args.batch_size, device=str(device),
        )
        prediction = probability >= args.threshold
        ground_truth = load_mask(mask_path).astype(bool)
        cldice, _, _ = compute_cldice(prediction, ground_truth)
        rows.append({
            "image_id": image_id,
            "dice": compute_dice(prediction, ground_truth),
            "iou": compute_iou(prediction, ground_truth),
            "precision": compute_precision(prediction, ground_truth),
            "recall": compute_recall(prediction, ground_truth),
            "cldice": cldice,
        })
        if index == 1 or index % 25 == 0 or index == len(image_ids):
            print(f"[{index}/{len(image_ids)}] mean Dice={np.mean([r['dice'] for r in rows]):.4f}", flush=True)

    output_model = "unet" if args.model == "unet" else "deeplabv3"
    csv_path = output_dir / f"deepglobe_{output_model}_test_per_image.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("image_id", *METRICS))
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, object] = {
        "dataset": "DeepGlobe Road Extraction",
        "split": "held-out deterministic test split",
        "n_images": len(rows),
        "model": args.model,
        "checkpoint": str(args.checkpoint),
        "threshold": args.threshold,
        "tile_size": args.tile_size,
        "overlap": args.overlap,
        "bootstrap_samples": args.bootstrap_samples,
        "metrics": {},
    }
    for metric_index, metric in enumerate(METRICS):
        values = np.asarray([float(row[metric]) for row in rows])
        lower, upper = bootstrap_interval(values, args.bootstrap_samples, args.seed + metric_index)
        summary["metrics"][metric] = {"mean": float(values.mean()), "ci95": [lower, upper]}

    summary_path = output_dir / f"deepglobe_{output_model}_test_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved: {csv_path}", flush=True)
    print(f"Saved: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
