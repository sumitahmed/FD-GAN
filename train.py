"""
FD-GAN Training Script.

GAN training with:
    - Generator:     ModernFDGAN (DenseNet-121 encoder + U-Net decoder)
    - Discriminator: PatchGAN (conditional on hazy input)
    - Losses:        L1 pixel + VGG perceptual + LSGAN adversarial

Usage:
    python train.py --hazy_dir data/train/hazy --clean_dir data/train/clean
    python train.py --hazy_dir data/train/hazy --clean_dir data/train/clean --epochs 100 --batch_size 4
    python train.py --resume checkpoints/latest.pth  (resume training)

Dataset layout:
    data/
      train/
        hazy/    (hazy images)
        clean/   (corresponding clean images, same filenames)
"""

import os
import sys
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import ModernFDGAN
from discriminator import NLayerDiscriminator
from losses import VGGPerceptualLoss, GANLoss
from dataset import DehazingDataset


def parse_args():
    p = argparse.ArgumentParser(description="FD-GAN Training")

    # Data
    p.add_argument("--hazy_dir", type=str, required=True,
                   help="Path to hazy training images")
    p.add_argument("--clean_dir", type=str, required=True,
                   help="Path to clean ground-truth images")
    p.add_argument("--val_hazy_dir", type=str, default=None,
                   help="Path to hazy validation images (optional)")
    p.add_argument("--val_clean_dir", type=str, default=None,
                   help="Path to clean validation images (optional)")

    # Training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--crop_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--num_workers", type=int, default=4)

    # Loss weights
    p.add_argument("--lambda_l1", type=float, default=10.0,
                   help="Weight for L1 pixel loss")
    p.add_argument("--lambda_perceptual", type=float, default=1.0,
                   help="Weight for VGG perceptual loss")
    p.add_argument("--lambda_gan", type=float, default=1.0,
                   help="Weight for adversarial loss")

    # Checkpoints
    p.add_argument("--save_dir", type=str, default="checkpoints")
    p.add_argument("--save_every", type=int, default=5,
                   help="Save checkpoint every N epochs")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from")

    return p.parse_args()


def save_checkpoint(path, epoch, gen, disc, opt_g, opt_d, best_loss):
    """Save training state."""
    torch.save({
        "epoch": epoch,
        "generator": gen.state_dict(),
        "discriminator": disc.state_dict(),
        "optimizer_g": opt_g.state_dict(),
        "optimizer_d": opt_d.state_dict(),
        "best_loss": best_loss,
    }, path)


def load_checkpoint(path, gen, disc, opt_g, opt_d, device):
    """Load training state from checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    gen.load_state_dict(ckpt["generator"])
    disc.load_state_dict(ckpt["discriminator"])
    opt_g.load_state_dict(ckpt["optimizer_g"])
    opt_d.load_state_dict(ckpt["optimizer_d"])
    return ckpt["epoch"], ckpt.get("best_loss", float("inf"))


def train_one_epoch(gen, disc, loader, opt_g, opt_d,
                    criterion_l1, criterion_perceptual, criterion_gan,
                    args, device, epoch):
    """Train for one epoch. Returns average generator loss."""
    gen.train()
    disc.train()

    total_g_loss = 0.0
    total_d_loss = 0.0

    for i, batch in enumerate(loader):
        hazy = batch["hazy"].to(device)
        clean = batch["clean"].to(device)

        # ────────────────────────────────────────────────────────
        # (1) Update Discriminator
        # ────────────────────────────────────────────────────────
        opt_d.zero_grad()

        with torch.no_grad():
            fake = gen(hazy)

        # Real pair
        pred_real = disc(hazy, clean)
        loss_d_real = criterion_gan(pred_real, is_real=True)

        # Fake pair
        pred_fake = disc(hazy, fake.detach())
        loss_d_fake = criterion_gan(pred_fake, is_real=False)

        loss_d = (loss_d_real + loss_d_fake) * 0.5
        loss_d.backward()
        opt_d.step()

        # ────────────────────────────────────────────────────────
        # (2) Update Generator
        # ────────────────────────────────────────────────────────
        opt_g.zero_grad()

        fake = gen(hazy)

        # Adversarial loss — fool discriminator
        pred_fake = disc(hazy, fake)
        loss_g_gan = criterion_gan(pred_fake, is_real=True)

        # L1 pixel loss
        loss_g_l1 = criterion_l1(fake, clean)

        # Perceptual loss
        loss_g_perc = criterion_perceptual(fake, clean)

        # Total generator loss
        loss_g = (
            args.lambda_gan * loss_g_gan
            + args.lambda_l1 * loss_g_l1
            + args.lambda_perceptual * loss_g_perc
        )

        loss_g.backward()
        opt_g.step()

        total_g_loss += loss_g.item()
        total_d_loss += loss_d.item()

        # Log every 50 batches
        if (i + 1) % 50 == 0 or (i + 1) == len(loader):
            print(
                f"  [{i+1}/{len(loader)}]  "
                f"G: {loss_g.item():.4f} "
                f"(L1={loss_g_l1.item():.4f}, "
                f"Perc={loss_g_perc.item():.4f}, "
                f"GAN={loss_g_gan.item():.4f})  "
                f"D: {loss_d.item():.4f}"
            )

    avg_g = total_g_loss / len(loader)
    avg_d = total_d_loss / len(loader)
    return avg_g, avg_d


@torch.no_grad()
def validate(gen, loader, criterion_l1, device):
    """Run validation and return average L1 loss."""
    gen.eval()
    total_loss = 0.0

    for batch in loader:
        hazy = batch["hazy"].to(device)
        clean = batch["clean"].to(device)
        fake = gen(hazy)
        total_loss += criterion_l1(fake, clean).item()

    return total_loss / len(loader)


def main():
    args = parse_args()

    # ── Device ──────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if device.type == "cpu":
        print("WARNING: Training on CPU will be very slow. GPU recommended.")
        print()

    # ── Dataset ─────────────────────────────────────────────────
    train_ds = DehazingDataset(
        args.hazy_dir, args.clean_dir,
        crop_size=args.crop_size, augment=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    val_loader = None
    if args.val_hazy_dir and args.val_clean_dir:
        val_ds = DehazingDataset(
            args.val_hazy_dir, args.val_clean_dir,
            crop_size=None, augment=False,
        )
        val_loader = DataLoader(
            val_ds, batch_size=1, shuffle=False,
            num_workers=2, pin_memory=(device.type == "cuda"),
        )

    # ── Models ──────────────────────────────────────────────────
    gen = ModernFDGAN().to(device)
    disc = NLayerDiscriminator(in_channels=6, ndf=64, n_layers=3).to(device)

    # Count parameters
    g_params = sum(p.numel() for p in gen.parameters())
    d_params = sum(p.numel() for p in disc.parameters())
    print(f"Generator:     {g_params:,} params")
    print(f"Discriminator: {d_params:,} params")
    print()

    # ── Optimizers ──────────────────────────────────────────────
    opt_g = optim.Adam(gen.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = optim.Adam(disc.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # ── Losses ──────────────────────────────────────────────────
    criterion_l1 = nn.L1Loss()
    criterion_perceptual = VGGPerceptualLoss().to(device)
    criterion_gan = GANLoss().to(device)

    # ── Resume ──────────────────────────────────────────────────
    start_epoch = 0
    best_loss = float("inf")

    if args.resume and os.path.isfile(args.resume):
        start_epoch, best_loss = load_checkpoint(
            args.resume, gen, disc, opt_g, opt_d, device
        )
        print(f"Resumed from epoch {start_epoch}, best_loss={best_loss:.4f}")

    # ── Training Loop ───────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Training for {args.epochs} epochs ({len(train_ds)} images, batch={args.batch_size})")
    print("=" * 70)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.perf_counter()

        # Linear LR decay after 50% of epochs
        if epoch > args.epochs // 2:
            decay = 1.0 - (epoch - args.epochs // 2) / (args.epochs // 2)
            for pg in opt_g.param_groups:
                pg["lr"] = args.lr * decay
            for pg in opt_d.param_groups:
                pg["lr"] = args.lr * decay

        print(f"\nEpoch {epoch+1}/{args.epochs}  (lr={opt_g.param_groups[0]['lr']:.6f})")
        print("-" * 70)

        avg_g, avg_d = train_one_epoch(
            gen, disc, train_loader, opt_g, opt_d,
            criterion_l1, criterion_perceptual, criterion_gan,
            args, device, epoch,
        )

        elapsed = time.perf_counter() - t0

        # Validation
        val_str = ""
        if val_loader:
            val_loss = validate(gen, val_loader, criterion_l1, device)
            val_str = f"  Val L1: {val_loss:.4f}"

            if val_loss < best_loss:
                best_loss = val_loss
                save_checkpoint(
                    os.path.join(args.save_dir, "best.pth"),
                    epoch + 1, gen, disc, opt_g, opt_d, best_loss,
                )
                val_str += " (best!)"

        print(
            f"  Avg G: {avg_g:.4f}  Avg D: {avg_d:.4f}  "
            f"Time: {elapsed:.1f}s{val_str}"
        )

        # Save periodic + latest
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                os.path.join(args.save_dir, f"epoch_{epoch+1:03d}.pth"),
                epoch + 1, gen, disc, opt_g, opt_d, best_loss,
            )

        save_checkpoint(
            os.path.join(args.save_dir, "latest.pth"),
            epoch + 1, gen, disc, opt_g, opt_d, best_loss,
        )

    # ── Export generator-only weights for inference ──────────────
    torch.save(gen.state_dict(), os.path.join(args.save_dir, "generator_final.pth"))
    print(f"\nDone! Generator saved to {args.save_dir}/generator_final.pth")
    print(f"Run inference: python infer.py --input IMAGE --checkpoint {args.save_dir}/generator_final.pth")


if __name__ == "__main__":
    main()
