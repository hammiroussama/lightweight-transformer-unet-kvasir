from __future__ import annotations

import torch
from torch import nn

from .blocks import DoubleConv, Down, TransformerBottleneck, Up


class ConfigurableUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
        use_attention: bool = False,
        use_transformer: bool = False,
        transformer_heads: int = 8,
        transformer_depth: int = 2,
        transformer_mlp_ratio: float = 2.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4, c5 = [base_channels * (2**i) for i in range(5)]
        self.inc = DoubleConv(in_channels, c1)
        self.down1 = Down(c1, c2)
        self.down2 = Down(c2, c3)
        self.down3 = Down(c3, c4)
        self.down4 = Down(c4, c5)
        self.bottleneck = (
            TransformerBottleneck(
                c5,
                num_heads=transformer_heads,
                depth=transformer_depth,
                mlp_ratio=transformer_mlp_ratio,
                dropout=dropout,
            )
            if use_transformer
            else nn.Identity()
        )
        self.up1 = Up(c5, c4, c4, use_attention)
        self.up2 = Up(c4, c3, c3, use_attention)
        self.up3 = Up(c3, c2, c2, use_attention)
        self.up4 = Up(c2, c1, c1, use_attention)
        self.outc = nn.Conv2d(c1, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneck(self.down4(x4))
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)
