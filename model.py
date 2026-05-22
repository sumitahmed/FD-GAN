"""
Modern FD-GAN Inspired Dehazing Model

Architecture:
    - Encoder: DenseNet-121 backbone (pretrained on ImageNet)
    - Decoder: U-Net style upsampling with skip connections
    - Output:  Tanh-activated RGB image in [-1, 1]

Inspired by:
    FD-GAN: Generative Adversarial Networks with Fusion-Discriminator
    for Single Image Dehazing (AAAI 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ConvBlock(nn.Module):
    """Double convolution block with BatchNorm and ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """Upsample ×2 → optional skip-connection concat → ConvBlock."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        # in_ch from previous decoder stage + skip_ch from encoder
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        if skip is not None:
            # Handle slight spatial mismatches from odd input sizes
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)

        return self.conv(x)


class ModernFDGAN(nn.Module):
    """
    DenseNet-121 encoder → U-Net decoder with skip connections.

    Encoder feature map progression (for 256×256 input):
        stem  (conv0+bn+relu+pool) → 64ch,  64×64   (÷4)
        enc1  (denseblock1+trans1) → 128ch, 32×32   (÷8)
        enc2  (denseblock2+trans2) → 256ch, 16×16   (÷16)
        enc3  (denseblock3+trans3) → 512ch,  8×8    (÷32)
        bottleneck (denseblock4)   → 1024ch, 8×8    (÷32)

    Decoder restores spatial resolution with skip connections:
        up1: 1024 + skip(512)  → 512,  16×16
        up2:  512 + skip(256)  → 256,  32×32
        up3:  256 + skip(128)  → 128,  64×64
        up4:  128 + skip(64)   →  64, 128×128
        up5:   64 + 0          →  32, 256×256  (no skip — matches input)

    Output head: 32 → 16 → 3  (Tanh activation, range [-1, 1])
    """

    def __init__(self):
        super().__init__()

        backbone = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        features = backbone.features

        # ── Encoder stages ──────────────────────────────────────────
        self.stem = nn.Sequential(
            features.conv0,   # 3  → 64,  stride=2  (÷2)
            features.norm0,
            features.relu0,
            features.pool0,   # maxpool stride=2     (÷4 total)
        )

        self.enc1 = nn.Sequential(features.denseblock1, features.transition1)
        self.enc2 = nn.Sequential(features.denseblock2, features.transition2)
        self.enc3 = nn.Sequential(features.denseblock3, features.transition3)

        self.bottleneck = nn.Sequential(features.denseblock4, features.norm5)

        # ── Decoder stages (with skip connections) ──────────────────
        #         input_ch  skip_ch  output_ch
        self.up1 = UpBlock(1024,     512,      512)
        self.up2 = UpBlock(512,      256,      256)
        self.up3 = UpBlock(256,      128,      128)
        self.up4 = UpBlock(128,       64,       64)
        self.up5 = UpBlock(64,         0,       32)   # No skip, restores to full res

        # ── Final residual projection ──────────────────────────────
        # Predicts a residual correction (not the full image).
        # output = input + residual  →  untrained residual ≈ 0  →  output ≈ input
        self.head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 1, bias=True),  # 1×1 conv → 3ch residual
        )

        # Zero-init the last conv so initial residual ≈ 0
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[2:]  # (H, W)

        # ── Encoder ─────────────────────────────────────────────────
        s0 = self.stem(x)       # 64ch,  H/4  × W/4
        s1 = self.enc1(s0)      # 128ch, H/8  × W/8
        s2 = self.enc2(s1)      # 256ch, H/16 × W/16
        s3 = self.enc3(s2)      # 512ch, H/32 × W/32

        b = self.bottleneck(s3) # 1024ch, H/32 × W/32

        # ── Decoder with skip connections ───────────────────────────
        d = self.up1(b,  s3)    # 512ch, H/16
        d = self.up2(d,  s2)    # 256ch, H/8
        d = self.up3(d,  s1)    # 128ch, H/4
        d = self.up4(d,  s0)    #  64ch, H/2
        d = self.up5(d)         #  32ch, H/1

        # Guard: ensure exact spatial match with input
        if d.shape[2:] != input_size:
            d = F.interpolate(d, size=input_size, mode="bilinear", align_corners=False)

        # ── Residual learning: output = input + learned_residual ────
        residual = self.head(d)
        return torch.clamp(x + residual, -1.0, 1.0)