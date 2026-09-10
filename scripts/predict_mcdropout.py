#!/usr/bin/env python
"""Monte Carlo Dropout inference: sliding-window full-image prediction with
T stochastic forward passes per tile (dropout kept active).

For each image outputs:
  {image_id}_prob.npy  — mean probability over T stochastic passes (H, W)
  {image_id}_std.npy   — epistemic std over T passes (H, W)
  {image_id}_logits.npy— mean logit (for temperature scaling)

Also saves the single-pass (dropout-off) prediction as {image_id}_single.npy
so we can compare "single forward vs MC-averaged" calibration.

Usage:
    python scripts/predict_mcdropout.py \
        --checkpoint artifacts/checkpoints_dropout/unet_seed42/best_model.pt \
        --split test --num-samples 30
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
from src.data.tiling import stitch_tiles, tile_locations
from src.models.unet import DropoutUNet


@torch.no_grad()
def mc_predict_full_image(
    model: torch.nn.Module,
    image: torch.Tensor,
    num_samples: int,
    tile_size: int,
    overlap: int,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sliding-window MC Dropout prediction.

    Returns (mean_prob, std, single_pass_prob) as (H, W) arrays.
    """
    _, _, h, w = image.shape
    coords = tile_locations(int(h), int(w), tile_size, overlap)
    dev = torch.device(device)

    mean_tiles, std_tiles, single_tiles = [], [], []

    for i in range(0, len(coords), batch_size):
        batch_coords = coords[i : i + batch_size]
        batch_tiles = []
        for rs, cs, re, ce in batch_coords:
            batch_tiles.append(image[:, :, rs:re, cs:ce])
        batch = torch.cat(batch_tiles, dim=0).to(dev)

        # Dropout ON: T stochastic passes
        model.train()
        sample_probs = []
        for _ in range(num_samples):
            probs = torch.sigmoid(model(batch)).cpu().numpy()[:, 0]
            sample_probs.append(probs)
        stack = np.stack(sample_probs, axis=0)  # (T, B, tile, tile)
        mean_b = stack.mean(axis=0)
        std_b = stack.std(axis=0)
        mean_tiles.extend([m for m in mean_b])
        std_tiles.extend([s for s in std_b])

        # Dropout OFF: single deterministic pass
        model.eval()
        single = torch.sigmoid(model(batch)).cpu().numpy()[:, 0]
        single_tiles.extend([s for s in single])

    model.eval()
    mean_prob = stitch_tiles(mean_tiles, coords, int(h), int(w), overlap)
    std = stitch_tiles(std_tiles, coords, int(h), int(w), overlap)
    single = stitch_tiles(single_tiles, coords, int(h), int(w), overlap)
    return mean_prob.astype(np.float32), std.astype(np.float32), single.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="MC Dropout inference.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["calibration", "test", "val"])
    parser.add_argument("--dropout-p", type=float, default=0.2)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=30)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/predictions/mcdropout"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}, dropout_p={args.dropout_p}, num_samples={args.num_samples}")

    model = DropoutUNet(base_channels=args.base_channels, dropout_p=args.dropout_p)
    model.load_state_dict(torch.load(args.root / args.checkpoint, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"Loaded {args.checkpoint}")

    splits = build_splits(args.root)
    split_map = {
        "seg_train": splits["seg_train_ids"],
        "calibration": splits["calibration_ids"],
        "val": splits["val_ids"],
        "test": splits["test_ids"],
    }
    image_ids = split_map[args.split]
    print(f"Processing {len(image_ids)} images from '{args.split}'")

    out_dir = args.root / args.output_dir / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_idx, image_id in enumerate(image_ids):
        print(f"[{img_idx+1}/{len(image_ids)}] {image_id} ...", end=" ", flush=True)
        img_path, _ = _resolve(args.root, image_id)
        image = load_image(img_path, as_tensor=True).unsqueeze(0)

        mean_prob, std, single = mc_predict_full_image(
            model, image, args.num_samples,
            args.tile_size, args.overlap, args.batch_size, str(device),
        )
        eps = 1e-9
        logit = np.log(np.clip(mean_prob, eps, 1 - eps) / (1 - np.clip(mean_prob, eps, 1 - eps)))
        np.save(out_dir / f"{image_id}_prob.npy", mean_prob)
        np.save(out_dir / f"{image_id}_std.npy", std)
        np.save(out_dir / f"{image_id}_single.npy", single)
        np.save(out_dir / f"{image_id}_logits.npy", logit.astype(np.float32))
        print(f"done (std={float(std.mean()):.4f})")

    print(f"MC Dropout predictions saved to {out_dir}")


def _resolve(root: Path, image_id: str):
    from src.data.splits import resolve_image_paths
    return resolve_image_paths(root, image_id)


if __name__ == "__main__":
    main()
