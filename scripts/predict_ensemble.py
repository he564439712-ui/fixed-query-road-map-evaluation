#!/usr/bin/env python
"""Deep Ensemble inference: run N models on a split and save
ensemble mean probability + epistemic uncertainty (std across models).

Outputs per image (saved to {output_dir}/{split}/):
  {image_id}_prob.npy  — ensemble mean probability (H, W) float32
  {image_id}_std.npy   — epistemic std across models (H, W) float32
  {image_id}_logits.npy — mean logit (derived from mean prob) for temp scaling

Per-model probabilities are also saved to {output_dir}/{split}/member{i}/
so member-level metrics (per-model Dice, etc.) remain possible.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import load_image
from src.data.splits import build_splits
from src.data.tiling import predict_full_image
from src.models.unet import UNet


def main():
    parser = argparse.ArgumentParser(description="Deep Ensemble inference.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="paths to model checkpoints (relative to root)")
    parser.add_argument("--split", default="calibration",
                        choices=["calibration", "test", "val"])
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/predictions"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}, {len(args.checkpoints)} models")

    # Load all models
    models = []
    for ckpt in args.checkpoints:
        model = UNet(base_channels=args.base_channels)
        model.load_state_dict(torch.load(args.root / ckpt, map_location=device))
        model = model.to(device)
        model.eval()
        models.append(model)
        print(f"  loaded {ckpt}")

    splits = build_splits(args.root)
    split_map = {
        "seg_train": splits["seg_train_ids"],
        "calibration": splits["calibration_ids"],
        "val": splits["val_ids"],
        "test": splits["test_ids"],
    }
    image_ids = split_map[args.split]
    print(f"Processing {len(image_ids)} images from split '{args.split}'")

    out_dir = args.root / args.output_dir / args.split
    member_dirs = []
    for i in range(len(models)):
        d = out_dir / f"member{i}"
        d.mkdir(parents=True, exist_ok=True)
        member_dirs.append(d)
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_idx, image_id in enumerate(image_ids):
        print(f"[{img_idx+1}/{len(image_ids)}] {image_id} ...", end=" ", flush=True)
        img_path, _mask_path = _resolve(args.root, image_id)
        image = load_image(img_path, as_tensor=True).unsqueeze(0)

        member_probs = []
        for i, model in enumerate(models):
            p = predict_full_image(
                model, image,
                tile_size=args.tile_size, overlap=args.overlap,
                batch_size=args.batch_size, device=str(device),
            )
            member_probs.append(p)
            np.save(member_dirs[i] / f"{image_id}_prob.npy", p)

        stacked = np.stack(member_probs, axis=0)  # (M, H, W)
        mean_prob = np.mean(stacked, axis=0).astype(np.float32)
        std = np.std(stacked, axis=0).astype(np.float32)
        # Logit from mean prob (for temperature scaling)
        eps = 1e-9
        mean_logit = np.log(np.clip(mean_prob, eps, 1 - eps) / (1 - np.clip(mean_prob, eps, 1 - eps)))
        mean_logit = mean_logit.astype(np.float32)

        np.save(out_dir / f"{image_id}_prob.npy", mean_prob)
        np.save(out_dir / f"{image_id}_std.npy", std)
        np.save(out_dir / f"{image_id}_logits.npy", mean_logit)
        print(f"done (std={float(std.mean()):.4f})")

    print(f"Ensemble predictions saved to {out_dir}")


def _resolve(root: Path, image_id: str) -> tuple[Path, Path]:
    from src.data.splits import resolve_image_paths
    return resolve_image_paths(root, image_id)


if __name__ == "__main__":
    main()
