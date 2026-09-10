#!/usr/bin/env python
"""Run model inference on full-resolution images using sliding windows.

Outputs probability maps as .npy files in artifacts/predictions/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import load_image, load_mask
from src.data.splits import build_splits, resolve_image_paths
from src.data.tiling import predict_full_image
from src.models.unet import UNet


def main():
    parser = argparse.ArgumentParser(description="Run inference on full images.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["seg_train", "calibration", "val", "test"])
    parser.add_argument("--image-ids", nargs="*", default=None)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/predictions"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    # Load model
    model = UNet(base_channels=args.base_channels)
    checkpoint_path = args.root / args.checkpoint
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")

    # Determine image list
    splits = build_splits(args.root)
    split_map = {
        "seg_train": splits["seg_train_ids"],
        "calibration": splits["calibration_ids"],
        "val": splits["val_ids"],
        "test": splits["test_ids"],
    }
    if args.image_ids:
        image_ids = args.image_ids
    else:
        image_ids = split_map[args.split]
    print(f"Processing {len(image_ids)} images from split '{args.split}'")

    output_dir = args.root / args.output_dir / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, image_id in enumerate(image_ids):
        img_path, mask_path = resolve_image_paths(args.root, image_id)
        print(f"[{i+1}/{len(image_ids)}] {image_id} ...", end=" ", flush=True)

        image = load_image(img_path, as_tensor=True)
        image = image.unsqueeze(0)  # add batch dim

        prob = predict_full_image(
            model,
            image,
            tile_size=args.tile_size,
            overlap=args.overlap,
            batch_size=args.batch_size,
            device=str(device),
        )

        np.save(output_dir / f"{image_id}_prob.npy", prob)
        print(f"done ({prob.shape})")

    print(f"Predictions saved to {output_dir}")


if __name__ == "__main__":
    main()
