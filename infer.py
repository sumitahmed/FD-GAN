"""Inference for the modern PyTorch FD-GAN generator."""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

from model import FDGANGenerator


def load_image(path: str) -> tuple[Image.Image, torch.Tensor, tuple[int, int]]:
    image = Image.open(path).convert("RGB")
    original_size = image.size
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return image, transform(image).unsqueeze(0), original_size


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.squeeze(0).detach().cpu()
    tensor = ((tensor + 1.0) * 0.5).clamp(0.0, 1.0)
    array = (tensor.numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def load_generator(path: str, device: torch.device) -> FDGANGenerator:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "generator" in state:
        state = state["generator"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model" in state:
        state = state["model"]
    if any(k.startswith("module.") for k in state):
        state = {k.removeprefix("module."): v for k, v in state.items()}

    model = FDGANGenerator(pretrained_encoder=False).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="FD-GAN dehazing inference")
    parser.add_argument("--checkpoint", required=True, help="Path to FDGAN generator checkpoint")
    parser.add_argument("--input", required=True, help="Path to hazy RGB image")
    parser.add_argument("--output", default="outputs/result.png", help="Output image path")
    parser.add_argument("--gt", default=None, help="Path to ground truth image (optional)")
    parser.add_argument(
        "--triptych",
        default=None,
        help="Path to save hazy|dehazed|ground-truth composite (optional)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(args.input)
    if args.gt and not os.path.isfile(args.gt):
        raise FileNotFoundError(args.gt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = load_generator(args.checkpoint, device)

    hazy_image, image, original_size = load_image(args.input)
    image = image.to(device)
    print(f"Input shape: {tuple(image.shape)} original={original_size}")

    start = time.perf_counter()
    with torch.no_grad():
        output = model(image)
    elapsed = time.perf_counter() - start

    result = tensor_to_image(output)
    if result.size != original_size:
        result = result.resize(original_size, Image.Resampling.LANCZOS)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    result.save(args.output)
    print(f"Output shape: {tuple(output.shape)} time={elapsed:.3f}s")
    print(f"Saved: {args.output}")

    if args.gt:
        gt_image = Image.open(args.gt).convert("RGB")
        target_size = result.size
        if hazy_image.size != target_size:
            hazy_image = hazy_image.resize(target_size, Image.Resampling.LANCZOS)
        if gt_image.size != target_size:
            gt_image = gt_image.resize(target_size, Image.Resampling.LANCZOS)

        triptych = Image.new("RGB", (target_size[0] * 3, target_size[1]))
        triptych.paste(hazy_image, (0, 0))
        triptych.paste(result, (target_size[0], 0))
        triptych.paste(gt_image, (target_size[0] * 2, 0))

        triptych_path = args.triptych
        if not triptych_path:
            root, ext = os.path.splitext(args.output)
            if not ext:
                ext = ".png"
            triptych_path = f"{root}_triptych{ext}"

        triptych_dir = os.path.dirname(triptych_path)
        if triptych_dir:
            os.makedirs(triptych_dir, exist_ok=True)
        triptych.save(triptych_path)
        print(f"Saved triptych: {triptych_path}")


if __name__ == "__main__":
    main()

