#!/usr/bin/env python
"""Train U-Net models for road segmentation.

Supports:
  - Standard U-Net training on 998-image segmentation training set
  - Multiple seeds for Deep Ensemble
  - MC Dropout variant
  - Mixed precision (AMP)
  - Gradient accumulation
  - Early stopping
  - Training log and checkpoint saving
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import RoadPatchDataset
from src.data.deepglobe import DeepGlobeRoadPatchDataset, build_deepglobe_splits
from src.data.splits import build_splits
from src.models.losses import BCEDiceLoss, RouteDemandBCEDiceLoss
from src.models.segmentation import MODEL_NAMES, count_trainable_parameters, create_segmentation_model
from src.models.unet import DropoutUNet


class ImageGroupSampler(Sampler[int]):
    """Shuffle at the image level; keep one image's patches consecutive.

    RoadPatchDataset extracts ``patches_per_image`` patches per image. With a
    global shuffle the same image's patches are scattered across the whole
    epoch, so its full 1500×1500 decode is repeated ~patches_per_image times
    (the dominant data-loading cost). By shuffling image order and keeping each
    image's patches adjacent, the decode LRU cache hits for all of a group.
    """

    def __init__(self, n_images: int, patches_per_image: int, seed: int = 42):
        self.n_images = n_images
        self.patches_per_image = patches_per_image
        self.seed = seed
        self._epoch = 0

    def __iter__(self):
        self._epoch += 1
        rng = np.random.default_rng(self.seed + self._epoch * 1000)
        for image_idx in rng.permutation(self.n_images):
            base = int(image_idx) * self.patches_per_image
            for p in range(self.patches_per_image):
                yield base + p

    def __len__(self) -> int:
        return self.n_images * self.patches_per_image


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: BCEDiceLoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    accumulation_steps: int,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_bce = 0.0
    total_dice = 0.0
    total_loss = 0.0
    n_samples = 0

    optimizer.zero_grad()
    for step, batch in enumerate(loader):
        images, masks = batch[:2]
        demand = batch[2].to(device) if len(batch) == 3 else None
        images = images.to(device)
        masks = masks.to(device)

        if scaler is not None:
            with torch.autocast(device_type=device.type):
                logits = model(images)
                loss, components = loss_fn(logits, masks, demand) if demand is not None else loss_fn(logits, masks)
                loss = loss / accumulation_steps
            scaler.scale(loss).backward()
        else:
            logits = model(images)
            loss, components = loss_fn(logits, masks, demand) if demand is not None else loss_fn(logits, masks)
            loss = loss / accumulation_steps
            loss.backward()

        should_step = (step + 1) % accumulation_steps == 0 or (step + 1) == len(loader)
        if should_step:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_bce += components["bce"] * len(images)
        total_dice += components["dice"] * len(images)
        total_loss += components["total"] * len(images)
        n_samples += len(images)

    return {
        "loss": total_loss / max(n_samples, 1),
        "bce": total_bce / max(n_samples, 1),
        "dice_loss": total_dice / max(n_samples, 1),
    }


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: BCEDiceLoss,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_bce = 0.0
    total_dice = 0.0
    n_samples = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        _, components = loss_fn(logits, masks)
        total_bce += components["bce"] * len(images)
        total_dice += components["dice"] * len(images)
        total_loss += components["total"] * len(images)
        n_samples += len(images)

    return {
        "val_loss": total_loss / max(n_samples, 1),
        "val_bce": total_bce / max(n_samples, 1),
        "val_dice_loss": total_dice / max(n_samples, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Train U-Net for road segmentation.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/checkpoints"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--dataset", choices=["massachusetts", "deepglobe"], default="massachusetts")
    parser.add_argument("--model", choices=MODEL_NAMES, default="unet")
    parser.add_argument("--pretrained-backbone", action="store_true")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--oof-fold", type=int, default=None,
        help="exclude one deterministic seg_train fold for out-of-fold prediction",
    )
    parser.add_argument("--oof-fold-count", type=int, default=5)
    parser.add_argument("--train-image-limit", type=int, default=None,
                        help="deterministically keep this many training images for a Gate")
    parser.add_argument("--route-demand-dir", type=Path, default=None,
                        help="directory containing <image_id>_demand.npy maps")
    parser.add_argument("--route-demand-alpha", type=float, default=2.0)
    parser.add_argument("--resume", type=Path, default=None,
                        help="resume from a last_state.pt training-state checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--dropout-p", type=float, default=0.0)
    # Defaults tuned for a 24 GB GPU (RTX 4090): batch 8 × 512² U-Net uses
    # ~10 GB comfortably; accumulation 2 keeps effective batch 16, matching
    # the plan and the local 8 GB config (batch 2 × accum 8) for reproducibility.
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--patches-per-image", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Optimisations, tuned by measurement on RTX 4090 / 5060:
    #   - cudnn.benchmark stays OFF: it bloats activation memory ~3× and slows
    #     steps (measured 156 ms → 89 ms after disabling). Not worth it here.
    #   - float32_matmul_precision='high' enables TF32 matmuls (harmless, small win).
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    if args.dropout_p > 0 and args.model != "unet":
        parser.error("--dropout-p is supported only for --model unet")

    # Build image-level, mutually disjoint splits.
    if args.dataset == "deepglobe":
        splits = build_deepglobe_splits(args.root, seed=args.split_seed)
        dataset_class = DeepGlobeRoadPatchDataset
    else:
        splits = build_splits(args.root)
        dataset_class = RoadPatchDataset
    train_ids = splits["seg_train_ids"]
    val_ids = splits["val_ids"]

    oof_heldout_ids: list[str] = []
    if args.oof_fold is not None:
        if args.dataset != "massachusetts":
            parser.error("--oof-fold currently supports only Massachusetts")
        if not 0 <= args.oof_fold < args.oof_fold_count:
            parser.error("--oof-fold must be in [0, --oof-fold-count)")
        ranked = sorted(
            train_ids,
            key=lambda image_id: hashlib.sha256(image_id.encode("utf-8")).hexdigest(),
        )
        oof_heldout_ids = [
            image_id for index, image_id in enumerate(ranked)
            if index % args.oof_fold_count == args.oof_fold
        ]
        heldout = set(oof_heldout_ids)
        train_ids = [image_id for image_id in train_ids if image_id not in heldout]
        if heldout & set(train_ids) or heldout & set(val_ids):
            raise RuntimeError("OOF held-out image leakage")

    if args.train_image_limit is not None:
        if args.train_image_limit <= 0:
            parser.error("--train-image-limit must be positive")
        train_ids = sorted(
            train_ids,
            key=lambda image_id: hashlib.sha256(
                f"route-gate:{image_id}".encode("utf-8")
            ).hexdigest(),
        )[:args.train_image_limit]

    print(f"Dataset: {args.dataset}, model: {args.model}")
    print(f"Train images: {len(train_ids)}, Val images: {len(val_ids)}")
    print(f"Device: {device}, AMP: {args.amp}")

    # Datasets
    train_dataset = dataset_class(
        image_ids=train_ids,
        root=args.root,
        patch_size=args.patch_size,
        patches_per_image=args.patches_per_image,
        augment=True,
        seed=args.seed,
        route_demand_dir=args.route_demand_dir,
    )
    val_dataset = dataset_class(
        image_ids=val_ids,
        root=args.root,
        patch_size=args.patch_size,
        patches_per_image=args.patches_per_image // 2,
        augment=False,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=ImageGroupSampler(
            len(train_ids), args.patches_per_image, seed=args.seed
        ),
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=args.num_workers > 0,
    )

    # Model
    if args.dropout_p > 0:
        model = DropoutUNet(
            base_channels=args.base_channels, dropout_p=args.dropout_p
        )
    else:
        model = create_segmentation_model(
            args.model,
            base_channels=args.base_channels,
            pretrained_backbone=args.pretrained_backbone,
        )
    model = model.to(device)
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")

    loss_fn = (
        RouteDemandBCEDiceLoss(demand_alpha=args.route_demand_alpha)
        if args.route_demand_dir is not None
        else BCEDiceLoss()
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler = (
        torch.amp.GradScaler("cuda") if args.amp and device.type == "cuda" else None
    )

    run_name = args.run_name
    if run_name is None:
        if args.oof_fold is not None:
            run_name = f"oof_{args.model}_fold{args.oof_fold}_seed{args.seed}"
        elif args.dataset == "massachusetts" and args.model == "unet":
            run_name = f"unet_seed{args.seed}"
        else:
            run_name = f"{args.dataset}_{args.model}_seed{args.seed}"
    output_dir = args.root / args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0
    train_log: list[dict] = []
    start_epoch = 1

    if args.resume is not None:
        resume_path = args.resume if args.resume.is_absolute() else args.root / args.resume
        state = torch.load(resume_path, map_location=device)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        if scaler is not None and state.get("scaler_state") is not None:
            scaler.load_state_dict(state["scaler_state"])
        best_val_loss = float(state["best_val_loss"])
        patience_counter = int(state["patience_counter"])
        train_log = list(state.get("train_log", []))
        start_epoch = int(state["epoch"]) + 1
        print(f"Resumed {resume_path} at epoch {start_epoch}")

    run_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    run_config["train_images"] = len(train_ids)
    run_config["val_images"] = len(val_ids)
    run_config["oof_heldout_images"] = len(oof_heldout_ids)
    run_config["oof_heldout_sha256"] = hashlib.sha256(
        "\n".join(oof_heldout_ids).encode("utf-8")
    ).hexdigest() if oof_heldout_ids else None
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)
    if oof_heldout_ids:
        (output_dir / "oof_heldout_ids.txt").write_text(
            "\n".join(oof_heldout_ids) + "\n", encoding="utf-8"
        )

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.perf_counter()
        train_metrics = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scaler,
            args.accumulation_steps, device,
        )
        val_metrics = validate(model, val_loader, loss_fn, device)
        scheduler.step()

        row = {
            "epoch": epoch,
            **train_metrics,
            **val_metrics,
            "lr": float(scheduler.get_last_lr()[0]),
            "seconds": time.perf_counter() - t0,
        }
        train_log.append(row)

        print(
            f"epoch {epoch:3d} | loss={row['loss']:.4f} val_loss={row['val_loss']:.4f} "
            f"lr={row['lr']:.2e} | {row['seconds']:.1f}s"
        )

        # Early stopping
        if val_metrics["val_loss"] < best_val_loss - 1e-4:
            best_val_loss = val_metrics["val_loss"]
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
        else:
            patience_counter += 1

        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict() if scaler is not None else None,
                "best_val_loss": best_val_loss,
                "patience_counter": patience_counter,
                "train_log": train_log,
            },
            output_dir / "last_state.pt",
        )
        with (output_dir / "train_log.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(train_log[0].keys()))
            writer.writeheader()
            writer.writerows(train_log)

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Save final model and log
    torch.save(model.state_dict(), output_dir / "last_model.pt")

    print(f"Training done. Best val_loss={best_val_loss:.4f}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
