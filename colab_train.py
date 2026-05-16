"""
FD-GAN Image Dehazing — Google Colab Training Script

Run this in Google Colab with GPU runtime.
Upload your project files to Colab or clone from GitHub.

Instructions:
  1. Open Google Colab
  2. Runtime -> Change runtime type -> GPU (T4 is fine)
  3. Upload this entire FD-GAN2 folder to Colab
  4. Run this script

This script handles:
  - Installing dependencies
  - Downloading RESIDE-6K dataset from Kaggle
  - Training the model
  - Saving checkpoints (downloadable)
"""

# ============================================================
# CELL 1: Setup & Install
# ============================================================

import subprocess
import sys
import os

# Install dependencies
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "torch", "torchvision", "pillow", "numpy", "gdown", "kaggle"])

print("Dependencies installed!")

# Check GPU
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
else:
    print("WARNING: No GPU detected! Training will be very slow.")
    print("Go to Runtime -> Change runtime type -> GPU")

# ============================================================
# CELL 2: Download RESIDE-6K Dataset
# ============================================================

DATA_DIR = "data"
TRAIN_HAZY = os.path.join(DATA_DIR, "train", "hazy")
TRAIN_CLEAN = os.path.join(DATA_DIR, "train", "clean")
TEST_HAZY = os.path.join(DATA_DIR, "test", "hazy")
TEST_CLEAN = os.path.join(DATA_DIR, "test", "clean")


def download_reside_6k():
    """Download RESIDE-6K from Kaggle (3K indoor + 3K outdoor pairs, 400x400)."""
    import zipfile

    os.makedirs(DATA_DIR, exist_ok=True)

    # Method 1: Try gdown (Google Drive)
    # RESIDE-6K mirrors
    try:
        import gdown

        # RESIDE ITS + OTS subset commonly used
        # Try Kaggle dataset download via direct URL
        print("Attempting RESIDE-6K download...")

        # Alternative: download directly from known mirrors
        url_its = "https://www.kaggle.com/api/v1/datasets/download/thedevastator/reside-6k"

        print("Downloading from Kaggle (this may take a few minutes)...")
        print("If this fails, download manually from:")
        print("  https://www.kaggle.com/datasets/thedevastator/reside-6k")
        print()

    except Exception as e:
        print(f"gdown not available: {e}")

    # Method 2: Kaggle CLI
    try:
        print("Trying Kaggle CLI download...")
        subprocess.run([
            "kaggle", "datasets", "download", "-d", "thedevastator/reside-6k",
            "-p", DATA_DIR, "--unzip"
        ], check=True)
        print("Downloaded via Kaggle CLI!")
        organize_reside_6k()
        return True
    except Exception as e:
        print(f"Kaggle CLI failed: {e}")
        print()

    # Method 3: Manual instructions
    print("=" * 60)
    print("MANUAL DOWNLOAD REQUIRED")
    print("=" * 60)
    print()
    print("1. Go to: https://www.kaggle.com/datasets/thedevastator/reside-6k")
    print("2. Click 'Download' (you need a Kaggle account)")
    print("3. Upload the zip to this Colab session")
    print("4. Run: !unzip reside-6k.zip -d data/")
    print()
    print("OR use Kaggle API:")
    print("  1. Go to kaggle.com -> Account -> Create API Token")
    print("  2. Upload kaggle.json to Colab")
    print("  3. Run:")
    print("     !mkdir -p ~/.kaggle")
    print("     !mv kaggle.json ~/.kaggle/")
    print("     !chmod 600 ~/.kaggle/kaggle.json")
    print("     !kaggle datasets download -d thedevastator/reside-6k -p data/ --unzip")
    print()
    return False


def organize_reside_6k():
    """Organize downloaded RESIDE-6K into train/test splits."""
    import shutil
    from pathlib import Path

    # Find the extracted data
    data_path = Path(DATA_DIR)

    # Common RESIDE-6K folder structure after unzipping:
    # - ITS/hazy, ITS/clear  OR  train/hazy, train/GT  etc.
    possible_structures = [
        # (hazy_pattern, clean_pattern)
        ("ITS/hazy", "ITS/clear"),
        ("ITS/hazy", "ITS/GT"),
        ("train/hazy", "train/clear"),
        ("train/hazy", "train/GT"),
        ("indoor/hazy", "indoor/GT"),
        ("hazy", "clear"),
        ("hazy", "GT"),
    ]

    for hazy_sub, clean_sub in possible_structures:
        hazy_p = data_path / hazy_sub
        clean_p = data_path / clean_sub
        if hazy_p.exists() and clean_p.exists():
            print(f"Found data at: {hazy_sub} / {clean_sub}")

            os.makedirs(TRAIN_HAZY, exist_ok=True)
            os.makedirs(TRAIN_CLEAN, exist_ok=True)

            # Copy/link images
            for f in hazy_p.glob("*.*"):
                shutil.copy2(f, os.path.join(TRAIN_HAZY, f.name))
            for f in clean_p.glob("*.*"):
                shutil.copy2(f, os.path.join(TRAIN_CLEAN, f.name))

            n_h = len(list(Path(TRAIN_HAZY).glob("*.*")))
            n_c = len(list(Path(TRAIN_CLEAN).glob("*.*")))
            print(f"Organized: {n_h} hazy, {n_c} clean images in train/")
            return

    # If none matched, list what we have
    print("Could not auto-detect folder structure. Contents:")
    for item in sorted(data_path.rglob("*")):
        if item.is_dir():
            n = len(list(item.glob("*.*")))
            if n > 0:
                print(f"  {item.relative_to(data_path)}/  ({n} files)")

    print("\nPlease manually move images to:")
    print(f"  {TRAIN_HAZY}/  and  {TRAIN_CLEAN}/")


# Run dataset setup
if not (os.path.exists(TRAIN_HAZY) and len(os.listdir(TRAIN_HAZY)) > 0):
    download_reside_6k()
else:
    n_h = len(os.listdir(TRAIN_HAZY))
    n_c = len(os.listdir(TRAIN_CLEAN))
    print(f"Dataset already present: {n_h} hazy, {n_c} clean images")


# ============================================================
# CELL 3: Verify Dataset
# ============================================================

from dataset import DehazingDataset
from torch.utils.data import DataLoader

try:
    ds = DehazingDataset(TRAIN_HAZY, TRAIN_CLEAN, crop_size=256, augment=True)
    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=2, drop_last=True)
    batch = next(iter(loader))
    print(f"Dataset OK! {len(ds)} pairs")
    print(f"  Batch: hazy={batch['hazy'].shape}, clean={batch['clean'].shape}")
except Exception as e:
    print(f"Dataset error: {e}")
    print("Make sure your images are in data/train/hazy/ and data/train/clean/")
    sys.exit(1)


# ============================================================
# CELL 4: Train!
# ============================================================

from model import ModernFDGAN
from discriminator import NLayerDiscriminator
from losses import VGGPerceptualLoss, GANLoss

import torch.nn as nn
import torch.optim as optim
import time

# Hyperparameters
EPOCHS = 100
BATCH_SIZE = 8       # Increase if you have more VRAM (T4=16GB -> try 8-16)
CROP_SIZE = 256
LR = 2e-4
LAMBDA_L1 = 10.0
LAMBDA_PERC = 1.0
LAMBDA_GAN = 1.0
SAVE_EVERY = 10

# DataLoader
train_ds = DehazingDataset(TRAIN_HAZY, TRAIN_CLEAN, crop_size=CROP_SIZE, augment=True)
train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=4, pin_memory=True, drop_last=True,
)

# Models
gen = ModernFDGAN().to(device)
disc = NLayerDiscriminator(in_channels=6, ndf=64, n_layers=3).to(device)

print(f"Generator:     {sum(p.numel() for p in gen.parameters()):,} params")
print(f"Discriminator: {sum(p.numel() for p in disc.parameters()):,} params")

# Optimizers
opt_g = optim.Adam(gen.parameters(), lr=LR, betas=(0.5, 0.999))
opt_d = optim.Adam(disc.parameters(), lr=LR, betas=(0.5, 0.999))

# Losses
criterion_l1 = nn.L1Loss()
criterion_perc = VGGPerceptualLoss().to(device)
criterion_gan = GANLoss().to(device)

# Checkpoint dir
os.makedirs("checkpoints", exist_ok=True)

print(f"\nTraining: {EPOCHS} epochs, {len(train_ds)} images, batch={BATCH_SIZE}")
print(f"Steps per epoch: {len(train_loader)}")
print("=" * 70)

best_g_loss = float("inf")

for epoch in range(EPOCHS):
    t0 = time.perf_counter()
    gen.train()
    disc.train()

    # LR decay after 50% of epochs
    if epoch > EPOCHS // 2:
        decay = 1.0 - (epoch - EPOCHS // 2) / (EPOCHS // 2)
        for pg in opt_g.param_groups:
            pg["lr"] = LR * decay
        for pg in opt_d.param_groups:
            pg["lr"] = LR * decay

    epoch_g_loss = 0.0
    epoch_d_loss = 0.0

    for i, batch in enumerate(train_loader):
        hazy = batch["hazy"].to(device)
        clean = batch["clean"].to(device)

        # ── Discriminator ──
        opt_d.zero_grad()
        with torch.no_grad():
            fake = gen(hazy)
        loss_d = 0.5 * (
            criterion_gan(disc(hazy, clean), is_real=True)
            + criterion_gan(disc(hazy, fake.detach()), is_real=False)
        )
        loss_d.backward()
        opt_d.step()

        # ── Generator ──
        opt_g.zero_grad()
        fake = gen(hazy)
        loss_g_gan = criterion_gan(disc(hazy, fake), is_real=True)
        loss_g_l1 = criterion_l1(fake, clean)
        loss_g_perc = criterion_perc(fake, clean)
        loss_g = LAMBDA_GAN * loss_g_gan + LAMBDA_L1 * loss_g_l1 + LAMBDA_PERC * loss_g_perc
        loss_g.backward()
        opt_g.step()

        epoch_g_loss += loss_g.item()
        epoch_d_loss += loss_d.item()

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(train_loader)}] G={loss_g.item():.4f} D={loss_d.item():.4f}")

    # Epoch stats
    avg_g = epoch_g_loss / len(train_loader)
    avg_d = epoch_d_loss / len(train_loader)
    elapsed = time.perf_counter() - t0

    print(f"Epoch {epoch+1}/{EPOCHS} | G={avg_g:.4f} D={avg_d:.4f} | {elapsed:.1f}s")

    # Save checkpoints
    if (epoch + 1) % SAVE_EVERY == 0:
        torch.save({
            "epoch": epoch + 1,
            "generator": gen.state_dict(),
            "discriminator": disc.state_dict(),
            "optimizer_g": opt_g.state_dict(),
            "optimizer_d": opt_d.state_dict(),
        }, f"checkpoints/epoch_{epoch+1:03d}.pth")
        print(f"  Saved checkpoint: epoch_{epoch+1:03d}.pth")

    # Save latest + best
    torch.save(gen.state_dict(), "checkpoints/latest_gen.pth")

    if avg_g < best_g_loss:
        best_g_loss = avg_g
        torch.save(gen.state_dict(), "checkpoints/best_gen.pth")
        print(f"  New best! G_loss={avg_g:.4f}")


# Final export
torch.save(gen.state_dict(), "checkpoints/generator_final.pth")
print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print(f"Best generator: checkpoints/best_gen.pth")
print(f"Final generator: checkpoints/generator_final.pth")
print()
print("Download checkpoints and run locally:")
print("  python infer.py --input your_image.jpg --checkpoint checkpoints/best_gen.pth")
