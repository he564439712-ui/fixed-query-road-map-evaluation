#!/usr/bin/env python
"""Deep Ensemble inference on the locked DeepGlobe test split (624 images).

Loads N DeepGlobe checkpoints (create_segmentation_model supports unet and
deeplabv3_resnet50), runs sliding-window inference, and saves per image:
  artifacts/predictions/deepglobe_test/{id}_prob.npy  — ensemble mean prob
  artifacts/predictions/deepglobe_test/{id}_std.npy   — epistemic std across members
  artifacts/predictions/deepglobe_test/member{i}/{id}_prob.npy

Usage (RTX 5060 thermal env):
    conda run -n thermal python scripts/predict_deepglobe_ensemble.py \
        --checkpoints artifacts/checkpoints/deepglobe_unet_seed42_fast/best_model.pt \
                       artifacts/checkpoints/deepglobe_unet_seed123/best_model.pt \
                       artifacts/checkpoints/deepglobe_unet_seed456/best_model.pt \
        --model unet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import load_image
from src.data.deepglobe import build_deepglobe_splits, resolve_deepglobe_paths
from src.data.tiling import predict_full_image
from src.models.segmentation import create_segmentation_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--model", default="unet", choices=("unet", "deeplabv3_resnet50"))
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/predictions/deepglobe_test"))
    parser.add_argument("--limit", type=int, default=None, help="only first N test images (smoke test)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    root = args.root.resolve()
    device = torch.device(args.device)
    print(f"Device: {device}, {len(args.checkpoints)} models ({args.model})", flush=True)

    models = []
    for ckpt in args.checkpoints:
        path = ckpt if ckpt.is_absolute() else root / ckpt
        model = create_segmentation_model(args.model, base_channels=args.base_channels)
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device).eval()
        models.append(model)
        print(f"  loaded {ckpt}", flush=True)

    image_ids = build_deepglobe_splits(root, seed=args.split_seed)["test_ids"]
    if args.limit:
        image_ids = image_ids[: args.limit]
    print(f"DeepGlobe test split: {len(image_ids)} images", flush=True)

    out_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    member_dirs = []
    for i in range(len(models)):
        d = out_dir / f"member{i}"
        d.mkdir(parents=True, exist_ok=True)
        member_dirs.append(d)

    for img_idx, image_id in enumerate(image_ids):
        img_path, _mask_path = resolve_deepglobe_paths(root, image_id)
        image = load_image(img_path, as_tensor=True).unsqueeze(0)

        member_probs = []
        for i, model in enumerate(models):
            p = predict_full_image(model, image, tile_size=args.tile_size,
                                   overlap=args.overlap, batch_size=args.batch_size,
                                   device=str(device))
            member_probs.append(p)
            np.save(member_dirs[i] / f"{image_id}_prob.npy", p)

        stacked = np.stack(member_probs, axis=0)
        mean_prob = np.mean(stacked, axis=0).astype(np.float32)
        std = np.std(stacked, axis=0).astype(np.float32)
        np.save(out_dir / f"{image_id}_prob.npy", mean_prob)
        np.save(out_dir / f"{image_id}_std.npy", std)

        if (img_idx + 1) % 25 == 0 or img_idx + 1 == len(image_ids):
            print(f"[{img_idx+1}/{len(image_ids)}] {image_id} (std mean {std.mean():.4f})", flush=True)

    print(f"Ensemble predictions saved to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
