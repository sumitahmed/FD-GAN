"""
Inference script for ModernFDGAN and legacy Hugging Face FDGAN checkpoints.

Examples:
    python infer.py --input sample_images/hazy.jpg --output outputs/result.png
    python infer.py --checkpoint best_gen.pth --input hazy.jpg --output dehazed.png
    python infer.py --checkpoint hf --input hazy.jpg --output dehazed.png
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T

from legacy_fdgan import LegacyFDGAN, is_legacy_fdgan_state_dict
from model import ModernFDGAN


def load_image(path: str) -> tuple[torch.Tensor, tuple[int, int]]:
    """Load an RGB image as a normalized (1, 3, H, W) tensor in [-1, 1]."""
    image = Image.open(path).convert("RGB")
    original_size = image.size

    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return transform(image).unsqueeze(0), original_size


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Convert a (1, 3, H, W) tensor in [-1, 1] to a PIL RGB image."""
    tensor = tensor.squeeze(0).detach().cpu()
    tensor = (tensor + 1.0) / 2.0
    tensor = torch.clamp(tensor, 0.0, 1.0)
    array = (tensor.numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def resolve_checkpoint(checkpoint: str | None) -> str | None:
    """Return a local checkpoint path. Use --checkpoint hf for the HF FDGAN file."""
    if checkpoint is None:
        return None

    if checkpoint.lower() not in {"hf", "huggingface", "ramssesdlsm/fdgan"}:
        return checkpoint

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for --checkpoint hf. "
            "Install it with: pip install huggingface_hub"
        ) from exc

    return hf_hub_download(
        repo_id="Ramssesdlsm/FDGAN",
        filename="FDGAN-generator.pth",
    )


def load_checkpoint(path: str, device: torch.device) -> dict[str, torch.Tensor]:
    """Load a generator state_dict from common checkpoint formats."""
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        state = torch.load(path, map_location=device, weights_only=False)

    if isinstance(state, dict) and "generator" in state:
        state = state["generator"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model" in state:
        state = state["model"]

    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(state)!r}")

    return state


def build_model(checkpoint_path: str | None, device: torch.device) -> torch.nn.Module:
    """Build the matching architecture and load weights when a checkpoint is given."""
    if checkpoint_path is None:
        print("No checkpoint loaded - running ModernFDGAN in demo mode.")
        return ModernFDGAN().to(device)

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state = load_checkpoint(checkpoint_path, device)
    if is_legacy_fdgan_state_dict(state):
        model = LegacyFDGAN().to(device)
        model.load_state_dict(state, strict=True)
        print("Loaded architecture: LegacyFDGAN")
    else:
        model = ModernFDGAN().to(device)
        model.load_state_dict(state, strict=True)
        print("Loaded architecture: ModernFDGAN")

    print(f"Loaded checkpoint: {checkpoint_path}")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="FD-GAN image dehazing inference")
    parser.add_argument("--input", required=True, help="Path to the hazy input image")
    parser.add_argument(
        "--output",
        default="outputs/result.png",
        help="Path to save the dehazed RGB output",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path, or 'hf' for Ramssesdlsm/FDGAN from Hugging Face",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input image not found: {args.input}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint_path = resolve_checkpoint(args.checkpoint)
    model = build_model(checkpoint_path, device)
    model.eval()

    image_tensor, original_size = load_image(args.input)
    image_tensor = image_tensor.to(device)
    print(f"Input shape: {tuple(image_tensor.shape)}  original={original_size}")

    start = time.perf_counter()
    with torch.no_grad():
        output = model(image_tensor)
    elapsed = time.perf_counter() - start

    if output.ndim != 4 or output.shape[0] != 1 or output.shape[1] != 3:
        raise RuntimeError(f"Expected output shape (1, 3, H, W), got {tuple(output.shape)}")

    print(f"Output shape: {tuple(output.shape)}  time={elapsed:.3f}s")

    result = tensor_to_image(output)
    if result.size != original_size:
        result = result.resize(original_size, Image.Resampling.LANCZOS)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    result.save(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
