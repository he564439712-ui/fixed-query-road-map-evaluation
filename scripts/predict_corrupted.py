#!/usr/bin/env python
"""M5: Deep Ensemble inference under image degradations (13 conditions).

For each test image and each corruption condition (clean + 4 types x 3
levels), corrupt the input image and run the 3-member ensemble via sliding
window. Saves per-condition ensemble predictions so downstream calibration
and planning evaluation can be run per condition.

Output layout:
  artifacts/predictions/corruptions/{condition}/{image_id}_{prob,std,logits}.npy

Usage (GPU, ~20 min on RTX 5060):
    python scripts/predict_corrupted.py --device cuda
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.corruptions import (
    CORRUPTION_CONDITIONS,
    apply_corruption,
    deterministic_corruption_seed,
)
from src.data.dataset import load_image
from src.data.splits import build_splits, resolve_image_paths
from src.data.tiling import predict_full_image
from src.models.unet import UNet


def main():
    parser = argparse.ArgumentParser(description="M5 corrupted-input ensemble inference.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--checkpoints", nargs="+", default=[
        "artifacts/checkpoints/unet_seed42/best_model.pt",
        "artifacts/checkpoints/unet_seed123/best_model.pt",
        "artifacts/checkpoints/unet_seed456/best_model.pt",
    ])
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/predictions/corruptions"))
    parser.add_argument("--split", default="test", choices=["test", "val", "calibration"])
    parser.add_argument("--conditions", nargs="+", default=None,
                        help="subset of condition names (default: all 13)")
    parser.add_argument("--limit", type=int, default=None, help="first N images (debug)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")
    if args.device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    root = args.root
    splits = build_splits(root)
    image_ids = splits[f"{args.split}_ids"]
    if args.limit:
        image_ids = image_ids[: args.limit]

    conditions = args.conditions or list(CORRUPTION_CONDITIONS.keys())
    print(f"{len(image_ids)} images x {len(conditions)} conditions")

    # Load ensemble once
    models = []
    for ckpt in args.checkpoints:
        m = UNet(base_channels=args.base_channels)
        m.load_state_dict(torch.load(root / ckpt, map_location=device))
        m = m.to(device).eval()
        models.append(m)
    print(f"Loaded {len(models)} models")

    base_seed = 42
    t_start = time.perf_counter()
    n_total = 0
    split_out = root / args.output_dir / args.split
    for cond in conditions:
        cond_dir = split_out / cond
        cond_dir.mkdir(parents=True, exist_ok=True)
        for img_idx, image_id in enumerate(image_ids):
            img_path, _ = resolve_image_paths(root, image_id)
            image = load_image(img_path, as_tensor=True).unsqueeze(0)  # (1,C,H,W) [0,1]

            if cond != "clean":
                # Corrupt on CPU tensor, then push to device for inference
                img_np = image.squeeze(0).numpy().astype(np.float32)
                corr_type, params = CORRUPTION_CONDITIONS[cond]
                corruption_seed = deterministic_corruption_seed(
                    image_id, cond, base_seed
                )
                img_corr = apply_corruption(
                    img_np, corr_type, params, seed=corruption_seed
                )
                image = torch.from_numpy(img_corr).unsqueeze(0)

            member_probs = []
            for m in models:
                p = predict_full_image(
                    m, image, tile_size=args.tile_size, overlap=args.overlap,
                    batch_size=args.batch_size, device=str(device),
                )
                member_probs.append(p)

            stacked = np.stack(member_probs, axis=0)  # (M,H,W)
            mean_prob = np.mean(stacked, axis=0).astype(np.float32)
            std = np.std(stacked, axis=0).astype(np.float32)
            eps = 1e-9
            pc = np.clip(mean_prob, eps, 1 - eps)
            mean_logit = np.log(pc / (1.0 - pc)).astype(np.float32)

            np.save(cond_dir / f"{image_id}_prob.npy", mean_prob)
            np.save(cond_dir / f"{image_id}_std.npy", std)
            np.save(cond_dir / f"{image_id}_logits.npy", mean_logit)
            n_total += 1

        print(f"  {cond}: {len(image_ids)} done", flush=True)

    elapsed = time.perf_counter() - t_start
    print(f"\nSaved {n_total} images x conditions to {root / args.output_dir}")
    print(f"Elapsed: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
