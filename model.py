"""FD-GAN generator ported to modern PyTorch.

This module follows the generator described in "FD-GAN: Generative
Adversarial Networks with Fusion-discriminator for Single Image Dehazing"
and the public WeilanAnnn/FD-GAN implementation:

* end-to-end hazy -> haze-free mapping, no explicit transmission or
  atmospheric-light estimation;
* DenseNet-121 dense blocks reused in the encoder;
* dense decoder blocks with nearest-neighbor upsampling to avoid
  checkerboard artifacts;
* Tanh RGB output in [-1, 1].
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class BottleneckBlock(nn.Module):
    """Dense decoder bottleneck used by the official FD-GAN code."""

    def __init__(self, in_channels: int, growth_channels: int, drop_rate: float = 0.0):
        super().__init__()
        inter_channels = growth_channels * 4
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, 1, bias=False)
        self.conv2 = nn.Conv2d(inter_channels, growth_channels, 3, padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.relu(x))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(self.relu(out))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return torch.cat([x, out], dim=1)


class TransitionUp(nn.Module):
    """1x1 projection followed by nearest-neighbor upsampling."""

    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 2):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, 1, bias=False)
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(self.relu(x))
        return F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")


def _densenet121_features(pretrained: bool) -> nn.Module:
    if pretrained:
        weights = models.DenseNet121_Weights.DEFAULT
    else:
        weights = None
    return models.densenet121(weights=weights).features


class FDGANGenerator(nn.Module):
    """Paper-aligned FD-GAN densely connected encoder-decoder.

    The public FD-GAN repo uses DenseNet-121's dense blocks as a feature
    extractor while avoiding the initial max-pool so spatial detail is kept.
    This port preserves that topology and supports arbitrary H/W divisible
    by 4, with final interpolation guarding exact output size.
    """

    def __init__(self, pretrained_encoder: bool = True):
        super().__init__()
        features = _densenet121_features(pretrained_encoder)

        self.conv_refin1 = nn.Conv2d(3, 64, 3, padding=1)
        self.relu0 = features.relu0

        self.dense_block1 = features.denseblock1
        self.trans_block1 = features.transition1
        self.dense_block2 = features.denseblock2
        self.trans_block2 = features.transition2
        self.dense_block3 = features.denseblock3
        self.trans_block3 = features.transition3

        self.conv_refin2 = nn.Conv2d(64, 32, 1)
        self.conv_refine4 = nn.Conv2d(160, 128, 3, padding=1)
        self.conv_refin5 = nn.Conv2d(256, 128, 1)
        self.conv_refin6 = nn.Conv2d(640, 512, 3, padding=1)

        self.dense_block4 = BottleneckBlock(512, 256)
        self.trans_block4 = TransitionUp(768, 128)
        self.dense_block5 = BottleneckBlock(384, 128)
        self.trans_block5 = TransitionUp(512, 64)
        self.dense_block6 = BottleneckBlock(64, 32)
        self.trans_block6 = TransitionUp(96, 16)

        self.conv_refin3 = nn.Conv2d(16, 3, 3, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]

        x0 = self.relu0(self.conv_refin1(x))
        x01 = self.conv_refin2(F.avg_pool2d(x0, 2))

        x1 = self.trans_block1(self.dense_block1(x0))
        x10 = self.conv_refine4(torch.cat([x01, x1], dim=1))

        x2 = self.trans_block2(self.dense_block2(x10))
        x3 = self.trans_block3(self.dense_block3(x2))
        x22 = self.conv_refin5(F.avg_pool2d(x2, 2))

        x4_in = self.conv_refin6(torch.cat([x3, x22], dim=1))
        x4 = self.trans_block4(self.dense_block4(x4_in))

        x5 = self.trans_block5(self.dense_block5(torch.cat([x4, x2], dim=1)))
        x6 = self.trans_block6(self.dense_block6(x5))

        out = self.tanh(self.conv_refin3(x6))
        if out.shape[-2:] != input_size:
            out = F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)
        return out


# Backward-compatible alias for older scripts in this repository.
ModernFDGAN = FDGANGenerator

