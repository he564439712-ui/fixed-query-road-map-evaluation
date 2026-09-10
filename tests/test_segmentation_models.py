from __future__ import annotations

import pytest
import torch

from src.models.segmentation import create_segmentation_model


@pytest.mark.parametrize("name", ["unet", "deeplabv3_resnet50"])
def test_segmentation_model_returns_binary_logits(name: str):
    model = create_segmentation_model(name, base_channels=8)
    model.eval()

    with torch.no_grad():
        logits = model(torch.zeros(1, 3, 64, 64))

    assert tuple(logits.shape) == (1, 1, 64, 64)
