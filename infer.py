"""
Inference script for Modern FD-GAN Inspired Dehazing.

Usage:
    python infer.py --input path/to/hazy_image.jpg
    python infer.py --input path/to/hazy_image.jpg --output results/dehazed.png
    python infer.py --input path/to/hazy_image.jpg --checkpoint checkpoints/model.pth
"""

import os
import argparse
import time

import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T

from model import ModernFDGAN


# ── Image I/O ───────────────────────────────────────────────────────

def load_image(path: str) -> tuple[torch.Tensor, tuple[int, int]]:
    """Load an image, normalize to [-1, 1], return (1,3,H,W) tensor and original (W,H)."""
    image = Image.open(path).convert("RGB")
    original_size = image.size  # (W, H)

    transform = T.Compose([
        T.ToTensor(),                                  # [0, 1]
        T.Normalize(mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]),               # [-1, 1]
    ])

    tensor = transform(image).unsqueeze(0)  # (1, 3, H, W)
    return tensor, original_size


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Convert a (1,3,H,W) tensor in [-1,1] to a PIL RGB image."""
    tensor = tensor.squeeze(0).detach().cpu()
    tensor = (tensor + 1.0) / 2.0        # → [0, 1]
    tensor = torch.clamp(tensor, 0.0, 1.0)
    array = (tensor.numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FD-GAN Inspired Image Dehazing — Inference"
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to the hazy input image"
    )
    parser.add_argument(
        "--output", default="outputs/result.png",
        help="Path to save the dehazed output (default: outputs/result.png)"
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to a trained model checkpoint (.pth). "
             "If not provided, uses randomly-initialized weights (demo mode)."
    )
    args = parser.parse_args()

    # ── Device ──────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Model ───────────────────────────────────────────────────
    model = ModernFDGAN().to(device)

    if args.checkpoint and os.path.isfile(args.checkpoint):
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("No checkpoint loaded — running with pretrained DenseNet encoder (demo mode)")

    model.eval()

    # ── Load input ──────────────────────────────────────────────
    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input image not found: {args.input}")

    image_tensor, original_size = load_image(args.input)
    image_tensor = image_tensor.to(device)
    print(f"Input shape: {image_tensor.shape}  (original {original_size[0]}×{original_size[1]})")

    # ── Inference ───────────────────────────────────────────────
    t0 = time.perf_counter()

    with torch.no_grad():
        output = model(image_tensor)

    elapsed = time.perf_counter() - t0
    print(f"Output shape: {output.shape}  ({elapsed:.3f}s)")

    # ── Validate output ─────────────────────────────────────────
    assert output.shape[1] == 3, (
        f"Expected 3-channel RGB output, got {output.shape[1]} channels. "
        f"Full shape: {output.shape}"
    )

    # ── Save ────────────────────────────────────────────────────
    result = tensor_to_image(output)

    # Resize to original dimensions if they don't match
    if result.size != original_size:
        result = result.resize(original_size, Image.LANCZOS)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    result.save(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()