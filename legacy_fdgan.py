"""
Compatibility model for the Hugging Face FDGAN-generator.pth checkpoint.

This is not the modern reimplementation in model.py. It mirrors the older
FD-GAN-style generator layout used by the checkpoint at:
    Ramssesdlsm/FDGAN/FDGAN-generator.pth
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ConvRelu(nn.Module):
    """Conv2d followed by ReLU, with state_dict keys under .block."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, padding: int = 0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=1, padding=padding, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvBnRelu(nn.Module):
    """Conv/ConvTranspose followed by BatchNorm and ReLU, with keys under .block."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        padding: int = 0,
        transpose: bool = False,
    ):
        super().__init__()
        conv_cls = nn.ConvTranspose2d if transpose else nn.Conv2d
        self.block = nn.Sequential(
            conv_cls(in_ch, out_ch, kernel_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Projection(nn.Module):
    """Wrapper used to produce checkpoint keys like side_branch1.proj.block.0."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = ConvRelu(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class DenseUpBlock(nn.Module):
    """Bottleneck dense block plus 1x1 up-transition used by the HF checkpoint."""

    def __init__(self, in_ch: int, growth_ch: int, out_ch: int):
        super().__init__()
        self.dense = nn.Sequential(
            ConvBnRelu(in_ch, growth_ch * 4, 1),
            ConvBnRelu(growth_ch * 4, growth_ch, 3, padding=1),
        )
        self.up_trans = ConvBnRelu(in_ch + growth_ch, out_ch, 1, transpose=True)

    def forward(self, x: torch.Tensor, size: tuple[int, int] | None = None) -> torch.Tensor:
        y = self.dense(x)
        y = torch.cat([x, y], dim=1)
        y = self.up_trans(y)
        if size is None:
            return F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=False)
        return F.interpolate(y, size=size, mode="bilinear", align_corners=False)


class LegacyEncoder(nn.Module):
    """DenseNet-121 feature blocks exposed with checkpoint-compatible names."""

    def __init__(self):
        super().__init__()
        backbone = models.densenet121(weights=None)
        self.features = backbone.features

        self.block1 = nn.Sequential(self.features.denseblock1, self.features.transition1)
        self.block2 = nn.Sequential(self.features.denseblock2, self.features.transition2)
        self.block3 = nn.Sequential(self.features.denseblock3, self.features.transition3)


class LegacyFDGAN(nn.Module):
    """
    Older FD-GAN-style generator compatible with Ramssesdlsm/FDGAN.

    Input/output tensors use the same normalization as infer.py: RGB in [-1, 1].
    """

    def __init__(self):
        super().__init__()
        self.conv_in = ConvRelu(3, 64, 3, padding=1)
        self.encoder = LegacyEncoder()

        self.side_branch1 = Projection(64, 32)
        self.side_branch2 = Projection(256, 128)

        self.fusion_x1 = ConvRelu(160, 128, 3, padding=1)
        self.fusion_bottleneck = ConvRelu(640, 512, 3, padding=1)

        self.block4 = DenseUpBlock(512, 256, 256)
        self.block5 = DenseUpBlock(512, 128, 128)
        self.block6 = DenseUpBlock(256, 64, 64)

        self.final_head = nn.Sequential(
            ConvRelu(64, 32, 3, padding=1),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1, bias=True),
        )
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[2:]

        x0 = self.conv_in(x)

        side1 = self.side_branch1(F.avg_pool2d(x0, kernel_size=2, stride=2))
        x1 = self.encoder.block1(x0)
        if side1.shape[2:] != x1.shape[2:]:
            side1 = F.interpolate(side1, size=x1.shape[2:], mode="bilinear", align_corners=False)
        x1 = self.fusion_x1(torch.cat([side1, x1], dim=1))

        x2 = self.encoder.block2(x1)
        x3 = self.encoder.block3(x2)

        side2 = self.side_branch2(F.avg_pool2d(x2, kernel_size=2, stride=2))
        if side2.shape[2:] != x3.shape[2:]:
            side2 = F.interpolate(side2, size=x3.shape[2:], mode="bilinear", align_corners=False)
        bottleneck = self.fusion_bottleneck(torch.cat([x3, side2], dim=1))

        d4 = self.block4(bottleneck, size=x2.shape[2:])
        d5 = self.block5(torch.cat([d4, x2], dim=1), size=x1.shape[2:])
        d6 = self.block6(torch.cat([d5, x1], dim=1), size=input_size)

        return self.tanh(self.final_head(d6))


def is_legacy_fdgan_state_dict(state: dict[str, torch.Tensor]) -> bool:
    required = {
        "conv_in.block.0.weight",
        "encoder.block1.0.denselayer1.conv1.weight",
        "fusion_bottleneck.block.0.weight",
        "block4.dense.0.block.0.weight",
        "final_head.1.weight",
    }
    return required.issubset(state.keys())
