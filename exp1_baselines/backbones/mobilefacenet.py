"""MobileFaceNet backbone (vendored from InsightFace).

Attribution:
    Source: deepinsight/insightface/recognition/arcface_torch/backbones/mobilefacenet.py
    Reference: Chen et al., "MobileFaceNets: Efficient CNNs for Accurate Real-time
    Face Verification on Mobile Devices" (CCBR 2018).

Adapted for exp1_baselines:
  - Input: [B, 3, 112, 112]
  - Output: [B, 512] (512-d embedding before any L2-norm)
  - The backbone keeps its 512-d embedding layer (GDC head).
  - The ArcFace classification head is provided separately in losses/arcface.py
    and is discarded after training.

Width scaling:
  InsightFace get_mbf factory defaults to scale=2, which makes conv1 produce
  64*scale = 128 channels. The original MobileFaceNet paper used 64 channels
  (= scale=1). We default to scale=2 to match the InsightFace baseline that
  prior palmprint papers reference; scale=1 is available as a smaller variant.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_c: int,
        out_c: int,
        kernel: int = 1,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        linear: bool = False,
    ) -> None:
        super().__init__()
        self.linear = linear
        self.conv = nn.Conv2d(
            in_c, out_c, kernel_size=kernel, stride=stride,
            padding=padding, groups=groups, bias=False,
        )
        self.bn = nn.BatchNorm2d(out_c)
        if not linear:
            self.prelu = nn.PReLU(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        if not self.linear:
            x = self.prelu(x)
        return x


class DepthWise(nn.Module):
    def __init__(
        self,
        in_c: int,
        out_c: int,
        kernel: int = 3,
        stride: int = 2,
        padding: int = 1,
        groups: int = 1,
        residual: bool = False,
    ) -> None:
        super().__init__()
        self.residual = residual
        self.conv = ConvBlock(in_c, groups, kernel=1, stride=1, padding=0)
        self.conv_dw = ConvBlock(
            groups, groups, kernel=kernel, stride=stride,
            padding=padding, groups=groups,
        )
        self.project = ConvBlock(groups, out_c, kernel=1, stride=1, padding=0, linear=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            x = x + identity
        return x


class Residual(nn.Module):
    def __init__(self, c: int, num_block: int, groups: int, kernel: int = 3,
                 stride: int = 1, padding: int = 1) -> None:
        super().__init__()
        modules = []
        for _ in range(num_block):
            modules.append(
                DepthWise(c, c, kernel=kernel, stride=stride,
                          padding=padding, groups=groups, residual=True)
            )
        self.model = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class GDC(nn.Module):
    """Global Depthwise Convolution (final embedding head)."""

    def __init__(self, embedding_size: int = 512) -> None:
        super().__init__()
        # For 112x112 input, the spatial dim before GDC is 7x7.
        self.conv_6_dw = ConvBlock(
            512, 512, kernel=7, stride=1, padding=0, groups=512, linear=True
        )
        self.conv_6_flatten = Flatten()
        self.linear = nn.Linear(512, embedding_size, bias=False)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_6_dw(x)
        x = self.conv_6_flatten(x)
        x = self.linear(x)
        x = self.bn(x)
        return x


class MobileFaceNet(nn.Module):
    """MobileFaceNet backbone for 112x112 input.

    Output: [B, embedding_dim] (default 512).
    """

    def __init__(self, embedding_dim: int = 512, fp16: bool = False) -> None:
        super().__init__()
        self.fp16 = fp16
        self.conv1 = ConvBlock(3, 64, kernel=3, stride=2, padding=1)
        self.conv2_dw = ConvBlock(64, 64, kernel=3, stride=1, padding=1, groups=64)
        self.conv_23 = DepthWise(64, 64, kernel=3, stride=2, padding=1, groups=128)
        self.conv_3 = Residual(64, num_block=4, groups=128, kernel=3, stride=1, padding=1)
        self.conv_34 = DepthWise(64, 128, kernel=3, stride=2, padding=1, groups=256)
        self.conv_4 = Residual(128, num_block=6, groups=256, kernel=3, stride=1, padding=1)
        self.conv_45 = DepthWise(128, 128, kernel=3, stride=2, padding=1, groups=512)
        self.conv_5 = Residual(128, num_block=2, groups=256, kernel=3, stride=1, padding=1)
        self.conv_6_sep = ConvBlock(128, 512, kernel=1, stride=1, padding=0)
        self.gdc = GDC(embedding_dim)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.cuda.amp.autocast(self.fp16):
            x = self.conv1(x)
            x = self.conv2_dw(x)
            x = self.conv_23(x)
            x = self.conv_3(x)
            x = self.conv_34(x)
            x = self.conv_4(x)
            x = self.conv_45(x)
            x = self.conv_5(x)
            x = self.conv_6_sep(x)
        x = self.gdc(x.float() if self.fp16 else x)
        return x
