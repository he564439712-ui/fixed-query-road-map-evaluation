"""Segmentation model factory used by backbone-generalization experiments."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.unet import UNet


MODEL_NAMES = ("unet", "deeplabv3_resnet50")


class TorchvisionSegmentationWrapper(nn.Module):
    """Expose torchvision segmentation models through a logits-only API."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        if not isinstance(output, dict) or "out" not in output:
            raise TypeError("Expected a torchvision segmentation output dictionary")
        return output["out"]


def create_segmentation_model(
    name: str,
    *,
    base_channels: int = 64,
    pretrained_backbone: bool = False,
) -> nn.Module:
    """Create a binary road-segmentation model that returns one logits map."""
    if name == "unet":
        return UNet(base_channels=base_channels)
    if name == "deeplabv3_resnet50":
        from torchvision.models import ResNet50_Weights
        from torchvision.models.segmentation import deeplabv3_resnet50

        backbone_weights = (
            ResNet50_Weights.IMAGENET1K_V2 if pretrained_backbone else None
        )
        model = deeplabv3_resnet50(
            weights=None,
            weights_backbone=backbone_weights,
            num_classes=1,
            aux_loss=False,
        )
        return TorchvisionSegmentationWrapper(model)
    raise ValueError(f"Unknown model '{name}'. Expected one of: {', '.join(MODEL_NAMES)}")


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
