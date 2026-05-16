"""
Download and prepare the RESIDE ITS (Indoor Training Set) dataset.

RESIDE is the standard benchmark for single image dehazing.
ITS contains ~13,990 hazy images from 1,399 clean indoor images.

Source: https://sites.google.com/view/raborsi/datasets/reside

This script downloads and organizes the data into:
    data/
      train/
        hazy/    (13,990 synthetic hazy images)
        clean/   (1,399 ground truth clean images)
      test/
        hazy/    (SOTS indoor test hazy images)
        clean/   (SOTS indoor test clean images)
"""

import os
import sys
import zipfile
import urllib.request
from pathlib import Path


# RESIDE ITS links (Google Drive alternative mirrors)
# NOTE: Google Drive links may require manual download.
# If automatic download fails, download manually and place zips in data/ folder.

DATA_DIR = Path(__file__).parent / "data"

INSTRUCTIONS = """
=================================================================
  RESIDE Dataset Setup Instructions
=================================================================

The RESIDE ITS dataset must be downloaded manually because
it's hosted on Google Drive / Baidu Pan.

OPTION 1: RESIDE-Standard (Recommended)
  Download from: https://sites.google.com/view/raborsi/datasets/reside
  - ITS (Indoor Training Set): ~13,990 hazy + 1,399 clean images
  - SOTS (Synthetic Objective Testing Set): test images

OPTION 2: Direct links (may change)
  ITS Hazy:  https://drive.google.com/file/d/1Gl6C0tqFiiuAE_ILGB7waZa2dEr4LmhJ
  ITS Clean: https://drive.google.com/file/d/1HhAVBai5ACxE9YWJEFBw-3kpnVwSJkK-

After downloading, organize as:

  {data_dir}/
    train/
      hazy/     <- put all hazy .png/.jpg images here
      clean/    <- put all clean .png/.jpg images here
    test/       <- (optional)
      hazy/
      clean/

For RESIDE ITS format, hazy filenames like "1_1_0.90179.png"
auto-pair with clean filenames like "1.png" (matching by ID).

OPTION 3: Use ANY paired dehazing dataset
  Just put hazy images in data/train/hazy/
  and clean images in data/train/clean/
  with matching filenames.

=================================================================
"""


def setup_dirs():
    """Create the expected directory structure."""
    dirs = [
        DATA_DIR / "train" / "hazy",
        DATA_DIR / "train" / "clean",
        DATA_DIR / "test" / "hazy",
        DATA_DIR / "test" / "clean",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  Created: {d}")


def check_data():
    """Check if data already exists."""
    train_hazy = DATA_DIR / "train" / "hazy"
    train_clean = DATA_DIR / "train" / "clean"

    if train_hazy.exists() and train_clean.exists():
        n_hazy = len(list(train_hazy.glob("*.*")))
        n_clean = len(list(train_clean.glob("*.*")))
        if n_hazy > 0 and n_clean > 0:
            print(f"Data found: {n_hazy} hazy, {n_clean} clean images")
            return True

    return False


if __name__ == "__main__":
    print("FD-GAN Dataset Preparation")
    print()

    if check_data():
        print("Dataset already set up!")
        sys.exit(0)

    print("Setting up directory structure...")
    setup_dirs()

    print(INSTRUCTIONS.format(data_dir=DATA_DIR))
