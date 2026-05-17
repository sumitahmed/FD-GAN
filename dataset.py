"""
Paired dataset loader for image dehazing.

Supports two folder layouts:

1. Simple paired (same filenames):
    data/train/hazy/001.png   <->   data/train/clean/001.png
    data/train/hazy/002.png   <->   data/train/clean/002.png

2. RESIDE ITS format:
    data/train/hazy/1_1_0.90179.png   ->   data/train/clean/1.png
    (hazy filename starts with clean image ID before first underscore)
"""

import os
import random
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class DehazingDataset(Dataset):
    """Load paired hazy/clean images for dehazing training.

    Args:
        hazy_dir:   Path to folder of hazy images.
        clean_dir:  Path to folder of corresponding clean images.
        crop_size:  Random crop size for training (None = use full image).
        augment:    Enable random horizontal/vertical flips.
    """

    def __init__(
        self,
        hazy_dir: str,
        clean_dir: str,
        crop_size: int = 256,
        augment: bool = True,
    ):
        self.hazy_dir = Path(hazy_dir)
        self.clean_dir = Path(clean_dir)
        self.crop_size = crop_size
        self.augment = augment

        # Supported image extensions
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

        # Collect hazy images recursively. Official RESIDE archives and Kaggle
        # mirrors are not always flattened the same way.
        hazy_files = sorted([
            f for f in self.hazy_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in exts
        ])

        # Build pairs
        self.pairs = []
        clean_files = {
            f.stem: f
            for f in self.clean_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in exts
        }

        for hf in hazy_files:
            # Try 1: exact filename match
            if hf.stem in clean_files:
                self.pairs.append((hf, clean_files[hf.stem]))
                continue

            # Try 2: RESIDE ITS format — hazy name = "{id}_{depth}_{beta}"
            clean_id = hf.stem.split("_")[0]
            if clean_id in clean_files:
                self.pairs.append((hf, clean_files[clean_id]))
                continue

        if len(self.pairs) == 0:
            raise RuntimeError(
                f"No paired images found!\n"
                f"  Hazy dir:  {self.hazy_dir} ({len(hazy_files)} images)\n"
                f"  Clean dir: {self.clean_dir} ({len(clean_files)} images)\n"
                f"Make sure filenames match or follow RESIDE naming convention."
            )

        print(f"Dataset: {len(self.pairs)} pairs from {self.hazy_dir.name}/")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        hazy_path, clean_path = self.pairs[idx]

        hazy = Image.open(hazy_path).convert("RGB")
        clean = Image.open(clean_path).convert("RGB")

        # Resize clean to match hazy if dimensions differ
        if hazy.size != clean.size:
            clean = clean.resize(hazy.size, Image.LANCZOS)

        # Ensure images are at least crop_size before random cropping.
        if self.crop_size is not None:
            if hazy.width < self.crop_size or hazy.height < self.crop_size:
                scale = self.crop_size / min(hazy.width, hazy.height)
                new_size = (
                    max(self.crop_size, int(round(hazy.width * scale))),
                    max(self.crop_size, int(round(hazy.height * scale))),
                )
                hazy = hazy.resize(new_size, Image.BICUBIC)
                clean = clean.resize(new_size, Image.BICUBIC)

            i, j, h, w = T.RandomCrop.get_params(hazy, (self.crop_size, self.crop_size))
            hazy = TF.crop(hazy, i, j, h, w)
            clean = TF.crop(clean, i, j, h, w)

        # Augmentation
        if self.augment:
            if random.random() > 0.5:
                hazy = TF.hflip(hazy)
                clean = TF.hflip(clean)
            if random.random() > 0.5:
                hazy = TF.vflip(hazy)
                clean = TF.vflip(clean)

        # To tensor and normalize to [-1, 1]
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        hazy_t = transform(hazy)
        clean_t = transform(clean)

        return {"hazy": hazy_t, "clean": clean_t, "name": hazy_path.stem}
