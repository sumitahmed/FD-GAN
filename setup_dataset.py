"""
Setup RESIDE-6K dataset after downloading from Kaggle.

Downloads from: https://www.kaggle.com/datasets/thedevastator/reside-6k

After downloading and extracting, run:
    python setup_dataset.py

This auto-detects the folder structure (GT/ or clear/) and
creates symlinks/copies into the expected data/train/hazy + data/train/clean layout.
"""

import os
import sys
import shutil
from pathlib import Path


DATA_DIR = Path(__file__).parent / "data"
EXPECTED_HAZY = DATA_DIR / "train" / "hazy"
EXPECTED_CLEAN = DATA_DIR / "train" / "clean"
EXPECTED_TEST_HAZY = DATA_DIR / "test" / "hazy"
EXPECTED_TEST_CLEAN = DATA_DIR / "test" / "clean"

# Possible names for the ground-truth folder
GT_NAMES = ["GT", "gt", "clear", "clean", "ground_truth", "groundtruth"]


def find_gt_folder(parent: Path) -> Path | None:
    """Find the ground-truth subfolder under a parent directory."""
    for name in GT_NAMES:
        candidate = parent / name
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def count_images(folder: Path) -> int:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif"}
    return sum(1 for f in folder.iterdir() if f.suffix.lower() in exts) if folder.exists() else 0


def setup_split(src_parent: Path, dst_hazy: Path, dst_clean: Path, split_name: str):
    """Setup one split (train or test)."""
    src_hazy = src_parent / "hazy"
    src_clean = find_gt_folder(src_parent)

    if not src_hazy.exists():
        print(f"  WARNING: {src_hazy} not found, skipping {split_name}")
        return

    if src_clean is None:
        print(f"  WARNING: No GT folder found in {src_parent}")
        print(f"  Looked for: {', '.join(GT_NAMES)}")
        return

    n_hazy = count_images(src_hazy)
    n_clean = count_images(src_clean)
    print(f"  Found {split_name}: {n_hazy} hazy, {n_clean} clean (from '{src_clean.name}/')")

    # Create destination dirs
    dst_hazy.mkdir(parents=True, exist_ok=True)
    dst_clean.mkdir(parents=True, exist_ok=True)

    # Check if already populated
    if count_images(dst_hazy) > 0 and count_images(dst_clean) > 0:
        print(f"  Already set up ({count_images(dst_hazy)} hazy, {count_images(dst_clean)} clean)")
        return

    # Copy images (or create symlinks on Linux)
    print(f"  Copying to {dst_hazy.parent}/ ...")

    for f in src_hazy.iterdir():
        if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif"}:
            shutil.copy2(f, dst_hazy / f.name)

    for f in src_clean.iterdir():
        if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif"}:
            shutil.copy2(f, dst_clean / f.name)

    print(f"  Done: {count_images(dst_hazy)} hazy, {count_images(dst_clean)} clean")


def find_reside_root() -> Path | None:
    """Search for RESIDE-6K folder in common locations."""
    search_paths = [
        Path(__file__).parent,                          # Same folder as script
        Path(__file__).parent / "data",                 # data/ subfolder
        Path(__file__).parent / "data_raw",             # data_raw/ subfolder
        Path(__file__).parent / "RESIDE-6K",            # Direct extraction
        Path.home() / "Downloads",                      # Downloads folder
        Path.home() / "Downloads" / "archive",          # Kaggle default
    ]

    for base in search_paths:
        if not base.exists():
            continue

        # Check if this IS the RESIDE root (has train/ and test/)
        if (base / "train" / "hazy").exists():
            gt = find_gt_folder(base / "train")
            if gt:
                return base

        # Check subfolders
        for child in base.iterdir():
            if child.is_dir():
                train_path = child / "train"
                if train_path.exists() and (train_path / "hazy").exists():
                    if find_gt_folder(train_path):
                        return child

    return None


def main():
    print("=" * 60)
    print("  RESIDE-6K Dataset Setup")
    print("=" * 60)

    # Check if already done
    if count_images(EXPECTED_HAZY) > 0 and count_images(EXPECTED_CLEAN) > 0:
        print(f"\nDataset already set up!")
        print(f"  Train: {count_images(EXPECTED_HAZY)} hazy, {count_images(EXPECTED_CLEAN)} clean")
        if EXPECTED_TEST_HAZY.exists():
            print(f"  Test:  {count_images(EXPECTED_TEST_HAZY)} hazy, {count_images(EXPECTED_TEST_CLEAN)} clean")
        return

    # Find RESIDE-6K
    print("\nSearching for RESIDE-6K dataset...")
    root = find_reside_root()

    if root is None:
        print("\nCould not find RESIDE-6K dataset!")
        print()
        print("Please download from Kaggle:")
        print("  https://www.kaggle.com/datasets/thedevastator/reside-6k")
        print()
        print("Then either:")
        print("  1. Extract to this folder (FD-GAN2/RESIDE-6K/)")
        print("  2. Extract to FD-GAN2/data/")
        print("  3. Run this script again")
        print()
        print("Expected structure after extraction:")
        print("  RESIDE-6K/")
        print("    train/")
        print("      hazy/     (hazy images)")
        print("      GT/       (ground truth clean images)")
        print("    test/")
        print("      hazy/")
        print("      GT/")
        return

    print(f"Found dataset at: {root}")
    print()

    # Setup train split
    train_src = root / "train"
    if train_src.exists():
        setup_split(train_src, EXPECTED_HAZY, EXPECTED_CLEAN, "train")
    else:
        # Maybe the root IS the train folder
        setup_split(root, EXPECTED_HAZY, EXPECTED_CLEAN, "train")

    # Setup test split
    test_src = root / "test"
    if test_src.exists():
        setup_split(test_src, EXPECTED_TEST_HAZY, EXPECTED_TEST_CLEAN, "test")

    print()
    print("=" * 60)
    print("  Setup complete! Ready to train:")
    print(f"  python train.py --hazy_dir {EXPECTED_HAZY} --clean_dir {EXPECTED_CLEAN}")
    print("=" * 60)


if __name__ == "__main__":
    main()
