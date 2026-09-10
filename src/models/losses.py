"""Loss functions for road segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Soft Dice loss.

    Args:
        logits: (B, 1, H, W) raw logits
        targets: (B, 1, H, W) binary targets
        epsilon: numerical stability

    Returns:
        scalar loss (1 - Dice)
    """
    probs = torch.sigmoid(logits)
    numerator = 2.0 * torch.sum(probs * targets, dim=(1, 2, 3)) + epsilon
    denominator = torch.sum(probs + targets, dim=(1, 2, 3)) + epsilon
    return 1.0 - torch.mean(numerator / denominator)


def combined_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """BCE + Dice combined loss.

    Returns (total_loss, component_dict).
    """
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight
    )
    d_loss = dice_loss(logits, targets)
    total = bce_weight * bce + dice_weight * d_loss

    components = {
        "bce": float(bce.item()),
        "dice": float(d_loss.item()),
        "total": float(total.item()),
    }
    return total, components


class BCEDiceLoss(nn.Module):
    """BCE + Dice loss as a nn.Module."""

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        pos_weight_max: float = 8.0,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.pos_weight_max = pos_weight_max

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        positive_ratio = float(torch.mean(targets))
        pos_weight_val = min(
            self.pos_weight_max,
            max(1.0, (1.0 - positive_ratio) / max(positive_ratio, 1e-4)),
        )
        pos_weight = torch.tensor(
            [pos_weight_val], dtype=logits.dtype, device=logits.device
        )
        return combined_loss(
            logits, targets, pos_weight, self.bce_weight, self.dice_weight
        )


class RouteDemandBCEDiceLoss(BCEDiceLoss):
    """BCE + Dice with label-derived route-demand emphasis on road pixels.

    ``demand`` must be a normalized map in [0, 1].  A value of one gives the
    corresponding positive pixel an additional ``demand_alpha`` loss weight;
    background retains unit weight.  This keeps the supervision local and
    makes the baseline exactly recoverable by setting ``demand_alpha=0``.
    """

    def __init__(self, demand_alpha: float = 2.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.demand_alpha = demand_alpha

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        demand: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if demand is None:
            return super().forward(logits, targets)
        positive_ratio = float(torch.mean(targets))
        pos_weight_val = min(
            self.pos_weight_max,
            max(1.0, (1.0 - positive_ratio) / max(positive_ratio, 1e-4)),
        )
        pos_weight = torch.tensor([pos_weight_val], dtype=logits.dtype, device=logits.device)
        pixel_weight = 1.0 + self.demand_alpha * demand * targets
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight, reduction="none"
        )
        bce = torch.sum(bce * pixel_weight) / torch.sum(pixel_weight)
        probs = torch.sigmoid(logits)
        numerator = 2.0 * torch.sum(probs * targets * pixel_weight, dim=(1, 2, 3)) + 1e-6
        denominator = torch.sum((probs + targets) * pixel_weight, dim=(1, 2, 3)) + 1e-6
        d_loss = 1.0 - torch.mean(numerator / denominator)
        total = self.bce_weight * bce + self.dice_weight * d_loss
        return total, {"bce": float(bce.item()), "dice": float(d_loss.item()), "total": float(total.item())}
