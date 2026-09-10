#!/usr/bin/env python
"""Predict one deterministic held-out seg_train fold with its OOF model."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import load_image
from src.data.splits import build_splits, resolve_image_paths
from src.data.tiling import predict_full_image
from src.models.segmentation import create_segmentation_model


def heldout_ids(root: Path, fold: int, fold_count: int) -> list[str]:
    ids = build_splits(root)["seg_train_ids"]
    ranked = sorted(ids, key=lambda item: hashlib.sha256(item.encode()).hexdigest())
    return [item for index, item in enumerate(ranked) if index % fold_count == fold]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--model", default="unet")
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/predictions/oof_train"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.fold < args.fold_count:
        parser.error("--fold must be in [0, --fold-count)")
    if not torch.cuda.is_available():
        raise RuntimeError("OOF prediction requires CUDA")

    root = args.root.resolve()
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else root / args.checkpoint
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    ids = heldout_ids(root, args.fold, args.fold_count)
    model = create_segmentation_model(args.model, base_channels=args.base_channels)
    state = torch.load(checkpoint, map_location="cuda")
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    model = model.cuda().eval()
    started = time.perf_counter()
    records = []
    for index, image_id in enumerate(ids):
        destination = output / f"{image_id}_prob.npy"
        if destination.exists() and not args.overwrite:
            raise FileExistsError(f"OOF prediction exists: {destination}")
        image_path, _ = resolve_image_paths(root, image_id)
        image = load_image(image_path, as_tensor=True).unsqueeze(0)
        probability = predict_full_image(
            model, image, tile_size=args.tile_size, overlap=args.overlap,
            batch_size=args.batch_size, device="cuda",
        ).astype(np.float32)
        np.save(destination, probability)
        records.append({"image_id": image_id, "shape": list(probability.shape)})
        print(f"fold={args.fold} [{index + 1}/{len(ids)}] {image_id}", flush=True)
    payload = {
        "experiment": "OOF segmentation material for image-guided bridge learning",
        "fold": args.fold,
        "fold_count": args.fold_count,
        "image_count": len(ids),
        "image_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "output_dir": str(output),
        "wall_seconds": time.perf_counter() - started,
        "records": records,
    }
    (output / f"fold{args.fold}_manifest.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
