"""Loss functions used by the FD-GAN paper.

Equation mapping:
* Eq. 4: L1 pixel-wise loss
* Eq. 6: SSIM loss = 1 - SSIM
* Eq. 7: VGG16 relu1_2 perceptual L1 loss
* Eq. 8: adversarial loss from Fusion-discriminator
* Eq. 9: weighted sum with alpha = (2, 1, 2, 0.1)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def denormalize(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1.0) * 0.5).clamp(0.0, 1.0)


def ssim_index(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11, eps: float = 1e-8) -> torch.Tensor:
    pred = denormalize(pred)
    target = denormalize(target)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    padding = window_size // 2

    mu_x = F.avg_pool2d(pred, window_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(target, window_size, stride=1, padding=padding)
    sigma_x = F.avg_pool2d(pred * pred, window_size, stride=1, padding=padding) - mu_x.square()
    sigma_y = F.avg_pool2d(target * target, window_size, stride=1, padding=padding) - mu_y.square()
    sigma_xy = F.avg_pool2d(pred * target, window_size, stride=1, padding=padding) - mu_x * mu_y

    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2) + eps
    )
    return score.mean()


class SSIMLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1.0 - ssim_index(pred, target)


class VGGRelu12Loss(nn.Module):
    """VGG16 relu1_2 feature L1 loss from the FD-GAN paper."""

    def __init__(self):
        super().__init__()
        features = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        self.slice = nn.Sequential(*list(features[:4]))
        for param in self.parameters():
            param.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = denormalize(x)
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self.slice(self._normalize(pred)), self.slice(self._normalize(target)))


class GANLoss(nn.Module):
    """Adversarial loss.

    FD-GAN writes the objective in BCE/log form. The default uses
    BCEWithLogitsLoss for numerical stability while leaving the discriminator
    final layer unsigmoided.
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, prediction: torch.Tensor, is_real: bool) -> torch.Tensor:
        target = torch.ones_like(prediction) if is_real else torch.zeros_like(prediction)
        return self.loss(prediction, target)


# Backward-compatible alias.
VGGPerceptualLoss = VGGRelu12Loss

