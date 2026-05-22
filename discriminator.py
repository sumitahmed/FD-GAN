"""Fusion-discriminator for FD-GAN.

The FD-GAN paper feeds the discriminator a concatenation of an RGB image,
its low-frequency component, and its high-frequency component:

    [X, X_LF, X_HF]

LF is extracted with a Gaussian filter. HF is extracted by converting RGB to
grayscale and applying a 3x3 Laplacian operator. This module implements that
frequency fusion in differentiable PyTorch and uses a PatchGAN classifier.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_gaussian_kernel(kernel_size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(xx.square() + yy.square()) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum().clamp_min(1e-12)
    return kernel.view(1, 1, kernel_size, kernel_size)


def low_frequency(x: torch.Tensor, kernel_size: int = 15, sigma: float = 3.0) -> torch.Tensor:
    """Gaussian low-frequency component, matching the paper setting."""
    if kernel_size % 2 == 0:
        raise ValueError("Gaussian kernel_size must be odd")
    kernel = _make_gaussian_kernel(kernel_size, sigma, x.device, x.dtype)
    kernel = kernel.repeat(x.shape[1], 1, 1, 1)
    padding = kernel_size // 2
    x_pad = F.pad(x, (padding, padding, padding, padding), mode="reflect")
    return F.conv2d(x_pad, kernel, groups=x.shape[1])


def rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
    weights = torch.tensor([0.299, 0.587, 0.114], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x * weights).sum(dim=1, keepdim=True)


def high_frequency(x: torch.Tensor) -> torch.Tensor:
    """Laplacian high-frequency component on grayscale image."""
    gray = rgb_to_gray(x)
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)
    gray = F.pad(gray, (1, 1, 1, 1), mode="reflect")
    hf = F.conv2d(gray, kernel)
    return hf.clamp(-1.0, 1.0)


def fusion_tensor(x: torch.Tensor, mode: str = "full") -> torch.Tensor:
    """Build discriminator input for Fusion-full, Fusion-LF, Fusion-HF, or RGB."""
    mode = mode.lower()
    if mode == "rgb":
        return x
    if mode == "lf":
        return torch.cat([x, low_frequency(x)], dim=1)
    if mode == "hf":
        return torch.cat([x, high_frequency(x)], dim=1)
    if mode == "full":
        return torch.cat([x, low_frequency(x), high_frequency(x)], dim=1)
    raise ValueError(f"Unknown fusion mode: {mode}")


def fusion_channels(mode: str = "full") -> int:
    return {"rgb": 3, "lf": 6, "hf": 4, "full": 7}[mode.lower()]


class FusionDiscriminator(nn.Module):
    """PatchGAN discriminator over fused RGB/LF/HF samples."""

    def __init__(
        self,
        fusion_mode: str = "full",
        ndf: int = 64,
        n_layers: int = 3,
        use_sigmoid: bool = False,
    ):
        super().__init__()
        self.fusion_mode = fusion_mode.lower()
        input_channels = fusion_channels(self.fusion_mode)

        layers: list[nn.Module] = [
            nn.Conv2d(input_channels, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers.extend(
                [
                    nn.Conv2d(ndf * nf_prev, ndf * nf_mult, 4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(ndf * nf_mult),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )

        nf_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers.extend(
            [
                nn.Conv2d(ndf * nf_prev, ndf * nf_mult, 4, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * nf_mult, 1, 4, stride=1, padding=1),
            ]
        )
        if use_sigmoid:
            layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, 0.0, 0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight, 1.0, 0.02)
            nn.init.zeros_(module.bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(fusion_tensor(image, self.fusion_mode))


# Compatibility name used by older tests/scripts.
class NLayerDiscriminator(FusionDiscriminator):
    def __init__(self, in_channels: int | None = None, ndf: int = 64, n_layers: int = 3):
        super().__init__(fusion_mode="full", ndf=ndf, n_layers=n_layers)

    def forward(self, hazy: torch.Tensor, image: torch.Tensor | None = None) -> torch.Tensor:
        return super().forward(hazy if image is None else image)
