"""
PatchGAN Discriminator for FD-GAN.

Outputs a spatial map of real/fake predictions rather than a single scalar.
This encourages the generator to produce locally realistic textures.

Architecture mirrors pix2pix / CycleGAN discriminators:
    70x70 PatchGAN (3 downsampling layers).
"""

import torch
import torch.nn as nn


class NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator with configurable depth.

    Args:
        in_channels: Input channels (6 for concatenated hazy+output pair).
        ndf:         Base number of discriminator filters.
        n_layers:    Number of downsampling conv layers.
    """

    def __init__(self, in_channels: int = 6, ndf: int = 64, n_layers: int = 3):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        ch_mult = 1
        for i in range(1, n_layers):
            ch_prev = ch_mult
            ch_mult = min(2 ** i, 8)
            layers += [
                nn.Conv2d(ndf * ch_prev, ndf * ch_mult, 4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ndf * ch_mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        # Second-to-last layer: stride=1
        ch_prev = ch_mult
        ch_mult = min(2 ** n_layers, 8)
        layers += [
            nn.Conv2d(ndf * ch_prev, ndf * ch_mult, 4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * ch_mult),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Final layer: 1-channel prediction map
        layers += [
            nn.Conv2d(ndf * ch_mult, 1, 4, stride=1, padding=1),
        ]

        self.model = nn.Sequential(*layers)

        # Initialize weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, hazy: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hazy:  Hazy input image  (B, 3, H, W) in [-1, 1]
            image: Real or generated (B, 3, H, W) in [-1, 1]

        Returns:
            Patch prediction map (B, 1, H', W')
        """
        x = torch.cat([hazy, image], dim=1)  # (B, 6, H, W)
        return self.model(x)
