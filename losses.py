"""
Loss functions for FD-GAN training.

Components:
    - L1 pixel loss          (reconstruction fidelity)
    - VGG perceptual loss    (high-level structural similarity)
    - LSGAN adversarial loss (stable GAN training)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class VGGPerceptualLoss(nn.Module):
    """Perceptual loss using VGG-16 feature maps.

    Compares features at relu1_2, relu2_2, relu3_3, relu4_3.
    Weights are frozen — only used for feature extraction.
    """

    def __init__(self):
        super().__init__()

        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features

        # Extract features at specific ReLU layers
        self.slice1 = nn.Sequential(*list(vgg[:4]))    # relu1_2
        self.slice2 = nn.Sequential(*list(vgg[4:9]))   # relu2_2
        self.slice3 = nn.Sequential(*list(vgg[9:16]))  # relu3_3
        self.slice4 = nn.Sequential(*list(vgg[16:23])) # relu4_3

        # Freeze all VGG parameters
        for param in self.parameters():
            param.requires_grad = False

        # ImageNet normalization (input should be [0, 1])
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Convert from [-1, 1] to ImageNet-normalized."""
        x = (x + 1.0) / 2.0  # -> [0, 1]
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = self._normalize(pred)
        target = self._normalize(target)

        loss = 0.0
        x, y = pred, target

        for layer in [self.slice1, self.slice2, self.slice3, self.slice4]:
            x = layer(x)
            y = layer(y)
            loss += F.l1_loss(x, y)

        return loss


class GANLoss(nn.Module):
    """LSGAN loss (Least Squares GAN).

    More stable than vanilla BCE GAN loss, avoids vanishing gradients.
    """

    def __init__(self, real_label: float = 1.0, fake_label: float = 0.0):
        super().__init__()
        self.register_buffer("real_val", torch.tensor(real_label))
        self.register_buffer("fake_val", torch.tensor(fake_label))

    def forward(self, prediction: torch.Tensor, is_real: bool) -> torch.Tensor:
        target = self.real_val if is_real else self.fake_val
        target = target.expand_as(prediction)
        return F.mse_loss(prediction, target)
