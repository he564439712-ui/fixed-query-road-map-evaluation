"""U-Net for road segmentation.

Standard U-Net with configurable depth and channels.
Supports dropout for MC Dropout uncertainty estimation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two consecutive Conv → BN → ReLU blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout_p > 0:
            layers.insert(2, nn.Dropout2d(dropout_p))
            layers.append(nn.Dropout2d(dropout_p))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UNetEncoder(nn.Module):
    """Contracting path: repeated DoubleConv → MaxPool."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 64,
        depth: int = 4,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        ch_in = in_channels
        ch_out = base_channels
        for _ in range(depth):
            self.encoders.append(DoubleConv(ch_in, ch_out, dropout_p))
            self.pools.append(nn.MaxPool2d(2))
            ch_in = ch_out
            ch_out *= 2

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        skip_features: list[torch.Tensor] = []
        for i in range(self.depth):
            x = self.encoders[i](x)
            skip_features.append(x)
            x = self.pools[i](x)
        return x, skip_features


class UNetDecoder(nn.Module):
    """Expansive path: repeated Upsample → Concat → DoubleConv."""

    def __init__(
        self,
        base_channels: int = 64,
        depth: int = 4,
        bilinear: bool = True,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.bilinear = bilinear

        self.upsamplers = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.dropout = nn.Dropout2d(dropout_p) if dropout_p > 0 else nn.Identity()

        ch = base_channels * (2 ** (depth - 1))
        for i in range(depth - 1):
            ch_skip = base_channels * (2 ** (depth - 2 - i))
            if bilinear:
                self.upsamplers.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                        nn.Conv2d(ch, ch // 2, kernel_size=1),
                    )
                )
            else:
                self.upsamplers.append(
                    nn.ConvTranspose2d(ch, ch // 2, kernel_size=2, stride=2)
                )
            self.decoders.append(DoubleConv(ch // 2 + ch_skip, ch // 2, 0.0))
            ch //= 2

    def forward(
        self, x: torch.Tensor, skip_features: list[torch.Tensor]
    ) -> torch.Tensor:
        # Decoder has depth-1 upsampling stages, so it consumes the
        # shallower depth-1 encoder skips. The deepest encoder output
        # (skip_features[depth-1]) feeds the bottleneck directly and is
        # NOT concatenated again. Reverse and drop the deepest one:
        #   [enc2, enc1, enc0] for depth=4.
        skip_features = skip_features[-2::-1]  # depth-1 items, deepest-first
        for i in range(self.depth - 1):
            x = self.upsamplers[i](x)
            x = self.dropout(x)
            skip = skip_features[i]
            # Handle size mismatch
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
            x = torch.cat([x, skip], dim=1)
            x = self.decoders[i](x)
        return x


class UNet(nn.Module):
    """Standard U-Net for road segmentation.

    Args:
        in_channels: input image channels (3 for RGB)
        out_channels: output channels (1 for binary road)
        base_channels: first-level channel count
        depth: number of pooling layers
        dropout_p: dropout rate (0 = no dropout, >0 for MC Dropout)
        bilinear: use bilinear upsampling instead of transposed conv
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 64,
        depth: int = 4,
        dropout_p: float = 0.0,
        bilinear: bool = True,
    ) -> None:
        super().__init__()
        self.base_channels = base_channels
        self.depth = depth
        self.dropout_p = dropout_p

        self.encoder = UNetEncoder(
            in_channels, base_channels, depth, dropout_p
        )
        self.decoder = UNetDecoder(
            base_channels, depth, bilinear, dropout_p
        )
        bottleneck_ch = base_channels * (2 ** (depth - 1))
        self.bottleneck = DoubleConv(
            bottleneck_ch, bottleneck_ch, dropout_p
        )
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits.

        During training: dropout is active (nn.Dropout2d in train mode).
        During inference with dropout_p > 0: call model.train() for MC samples.
        """
        x, skip_features = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, skip_features)
        return self.out_conv(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Single forward pass returning probability."""
        self.eval()
        return torch.sigmoid(self.forward(x))

    @torch.no_grad()
    def predict_mc_samples(
        self, x: torch.Tensor, num_samples: int = 30
    ) -> torch.Tensor:
        """Monte Carlo Dropout: collect T stochastic forward passes.

        Returns (T, 1, H, W) logits tensor.
        """
        self.train()  # keep dropout active
        samples = []
        for _ in range(num_samples):
            samples.append(self.forward(x))
        return torch.stack(samples, dim=0)

    @torch.no_grad()
    def predict_mc_statistics(
        self, x: torch.Tensor, num_samples: int = 30
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """MC Dropout: return (mean_prob, epistemic_std)."""
        logits_samples = self.predict_mc_samples(x, num_samples)
        prob_samples = torch.sigmoid(logits_samples)
        mean_prob = prob_samples.mean(dim=0)
        std = prob_samples.std(dim=0)
        return mean_prob, std


class DropoutUNet(UNet):
    """U-Net with dropout enabled for MC Dropout.

    Thin wrapper that defaults dropout_p > 0.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 64,
        depth: int = 4,
        dropout_p: float = 0.2,
        bilinear: bool = True,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            depth=depth,
            dropout_p=dropout_p,
            bilinear=bilinear,
        )
