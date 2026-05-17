"""
Training script for the modern FD-GAN-inspired dehazing model.

This trainer is built for measurable image restoration quality:
L1 reconstruction, VGG perceptual loss, optional SSIM loss, optional delayed
GAN training, PSNR/SSIM validation, sample-image export, AMP, warm-start
from generator-only weights, and full checkpoint resume.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dataset import DehazingDataset
from discriminator import NLayerDiscriminator
from losses import GANLoss, VGGPerceptualLoss
from model import ModernFDGAN


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ModernFDGAN for image dehazing")

    p.add_argument("--hazy_dir", required=True, help="Path to hazy training images")
    p.add_argument("--clean_dir", required=True, help="Path to clean training images")
    p.add_argument("--val_hazy_dir", default=None, help="Path to hazy validation images")
    p.add_argument("--val_clean_dir", default=None, help="Path to clean validation images")

    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--crop_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4, help="Fallback LR for G and D")
    p.add_argument("--lr_g", type=float, default=None, help="Generator LR override")
    p.add_argument("--lr_d", type=float, default=None, help="Discriminator LR override")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--amp", action="store_true", help="Use CUDA mixed precision")

    p.add_argument("--lambda_l1", type=float, default=20.0)
    p.add_argument("--lambda_perceptual", type=float, default=1.0)
    p.add_argument("--lambda_ssim", type=float, default=2.0)
    p.add_argument("--lambda_gan", type=float, default=0.05)
    p.add_argument("--gan_start_epoch", type=int, default=20)

    p.add_argument("--save_dir", default="checkpoints")
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--sample_every", type=int, default=5)
    p.add_argument("--resume", default=None, help="Full checkpoint to resume")
    p.add_argument("--pretrained_gen", default=None, help="Generator-only warm start")

    return p.parse_args()


def save_checkpoint(path, epoch, gen, disc, opt_g, opt_d, best_psnr, scaler=None):
    payload = {
        "epoch": epoch,
        "generator": gen.state_dict(),
        "discriminator": disc.state_dict(),
        "optimizer_g": opt_g.state_dict(),
        "optimizer_d": opt_d.state_dict(),
        "best_psnr": best_psnr,
    }
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, path)


def load_checkpoint(path, gen, disc, opt_g, opt_d, scaler, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    gen.load_state_dict(ckpt["generator"])
    disc.load_state_dict(ckpt["discriminator"])
    opt_g.load_state_dict(ckpt["optimizer_g"])
    opt_d.load_state_dict(ckpt["optimizer_d"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt["epoch"], ckpt.get("best_psnr", -float("inf"))


def load_generator_weights(path, gen, device):
    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "generator" in state:
        state = state["generator"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model" in state:
        state = state["model"]
    gen.load_state_dict(state, strict=True)


def denormalize(x):
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


def psnr(pred, target, eps=1e-8):
    pred = denormalize(pred)
    target = denormalize(target)
    mse = F.mse_loss(pred, target)
    return 10.0 * torch.log10(1.0 / (mse + eps))


def ssim(pred, target, window_size=11, eps=1e-8):
    pred = denormalize(pred)
    target = denormalize(target)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    padding = window_size // 2

    mu_x = F.avg_pool2d(pred, window_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(target, window_size, stride=1, padding=padding)
    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x = F.avg_pool2d(pred * pred, window_size, stride=1, padding=padding) - mu_x2
    sigma_y = F.avg_pool2d(target * target, window_size, stride=1, padding=padding) - mu_y2
    sigma_xy = F.avg_pool2d(pred * target, window_size, stride=1, padding=padding) - mu_xy

    score = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x + sigma_y + c2) + eps
    )
    return score.mean()


def train_one_epoch(gen, disc, loader, opt_g, opt_d, losses, args, device, epoch, scaler):
    gen.train()
    disc.train()
    use_amp = scaler is not None
    use_gan = epoch >= args.gan_start_epoch and args.lambda_gan > 0.0
    totals = {"g": 0.0, "d": 0.0, "l1": 0.0, "perc": 0.0, "ssim_loss": 0.0, "gan": 0.0}

    for i, batch in enumerate(loader):
        hazy = batch["hazy"].to(device, non_blocking=True)
        clean = batch["clean"].to(device, non_blocking=True)

        opt_d.zero_grad(set_to_none=True)
        if use_gan:
            with torch.no_grad():
                fake_detached = gen(hazy)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss_d = 0.5 * (
                    losses["gan"](disc(hazy, clean), True)
                    + losses["gan"](disc(hazy, fake_detached.detach()), False)
                )
            if use_amp:
                scaler.scale(loss_d).backward()
                scaler.step(opt_d)
            else:
                loss_d.backward()
                opt_d.step()
        else:
            loss_d = torch.zeros((), device=device)

        opt_g.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            fake = gen(hazy)
            loss_l1 = losses["l1"](fake, clean)
            loss_perc = losses["perc"](fake, clean)
            loss_ssim = 1.0 - ssim(fake, clean)
            if use_gan:
                loss_gan = losses["gan"](disc(hazy, fake), True)
            else:
                loss_gan = torch.zeros((), device=device)

            loss_g = (
                args.lambda_l1 * loss_l1
                + args.lambda_perceptual * loss_perc
                + args.lambda_ssim * loss_ssim
                + args.lambda_gan * loss_gan
            )

        if use_amp:
            scaler.scale(loss_g).backward()
            scaler.step(opt_g)
            scaler.update()
        else:
            loss_g.backward()
            opt_g.step()

        totals["g"] += loss_g.item()
        totals["d"] += loss_d.item()
        totals["l1"] += loss_l1.item()
        totals["perc"] += loss_perc.item()
        totals["ssim_loss"] += loss_ssim.item()
        totals["gan"] += loss_gan.item()

        if (i + 1) % 50 == 0 or (i + 1) == len(loader):
            print(
                f"  [{i+1}/{len(loader)}] "
                f"G={loss_g.item():.4f} D={loss_d.item():.4f} "
                f"L1={loss_l1.item():.4f} Perc={loss_perc.item():.4f} "
                f"SSIMLoss={loss_ssim.item():.4f} GAN={loss_gan.item():.4f}"
            )

    n = len(loader)
    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def validate(gen, loader, criterion_l1, device):
    gen.eval()
    totals = {"l1": 0.0, "psnr": 0.0, "ssim": 0.0}
    for batch in loader:
        hazy = batch["hazy"].to(device, non_blocking=True)
        clean = batch["clean"].to(device, non_blocking=True)
        fake = gen(hazy)
        totals["l1"] += criterion_l1(fake, clean).item()
        totals["psnr"] += psnr(fake, clean).item()
        totals["ssim"] += ssim(fake, clean).item()
    n = len(loader)
    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def save_samples(gen, loader, device, save_dir, epoch, max_items=4):
    gen.eval()
    batch = next(iter(loader))
    hazy = batch["hazy"][:max_items].to(device)
    clean = batch["clean"][:max_items].to(device)
    fake = gen(hazy)

    rows = []
    for i in range(hazy.shape[0]):
        rows.extend([denormalize(hazy[i]), denormalize(fake[i]), denormalize(clean[i])])
    os.makedirs(save_dir, exist_ok=True)
    save_image(rows, os.path.join(save_dir, f"epoch_{epoch:03d}.png"), nrow=3)


def append_metrics(path, row):
    new_file = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def set_lr(opt, lr):
    for group in opt.param_groups:
        group["lr"] = lr


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if device.type == "cpu":
        print("WARNING: CPU training is not practical for research-grade results.")

    train_ds = DehazingDataset(args.hazy_dir, args.clean_dir, crop_size=args.crop_size, augment=True)
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
        val_ds = DehazingDataset(args.val_hazy_dir, args.val_clean_dir, crop_size=None, augment=False)
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=max(1, min(2, args.num_workers)),
            pin_memory=(device.type == "cuda"),
        )

    gen = ModernFDGAN().to(device)
    disc = NLayerDiscriminator(in_channels=6, ndf=64, n_layers=3).to(device)

    lr_g = args.lr_g if args.lr_g is not None else args.lr
    lr_d = args.lr_d if args.lr_d is not None else args.lr
    opt_g = optim.Adam(gen.parameters(), lr=lr_g, betas=(0.5, 0.999))
    opt_d = optim.Adam(disc.parameters(), lr=lr_d, betas=(0.5, 0.999))
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))
    scaler = scaler if scaler.is_enabled() else None

    losses = {
        "l1": nn.L1Loss(),
        "perc": VGGPerceptualLoss().to(device),
        "gan": GANLoss().to(device),
    }

    os.makedirs(args.save_dir, exist_ok=True)
    start_epoch = 0
    best_psnr = -float("inf")

    if args.resume:
        start_epoch, best_psnr = load_checkpoint(args.resume, gen, disc, opt_g, opt_d, scaler, device)
        print(f"Resumed from epoch {start_epoch}, best_psnr={best_psnr:.2f}")
    elif args.pretrained_gen:
        load_generator_weights(args.pretrained_gen, gen, device)
        print(f"Warm-started generator from: {args.pretrained_gen}")
        print("Discriminator and optimizers are initialized from scratch.")

    print(f"Generator params: {sum(p.numel() for p in gen.parameters()):,}")
    print(f"Discriminator params: {sum(p.numel() for p in disc.parameters()):,}")
    print(f"Training: epochs={args.epochs}, batch={args.batch_size}, crop={args.crop_size}")
    print(
        f"Loss weights: L1={args.lambda_l1}, Perc={args.lambda_perceptual}, "
        f"SSIM={args.lambda_ssim}, GAN={args.lambda_gan}, GAN start={args.gan_start_epoch}"
    )

    metrics_path = os.path.join(args.save_dir, "metrics.csv")
    sample_loader = val_loader if val_loader is not None else train_loader

    for epoch in range(start_epoch, args.epochs):
        t0 = time.perf_counter()

        if epoch > args.epochs // 2:
            decay = max(0.0, 1.0 - (epoch - args.epochs // 2) / max(1, args.epochs // 2))
            set_lr(opt_g, lr_g * decay)
            set_lr(opt_d, lr_d * decay)

        print(f"\nEpoch {epoch + 1}/{args.epochs} lr_g={opt_g.param_groups[0]['lr']:.6g} lr_d={opt_d.param_groups[0]['lr']:.6g}")
        train_metrics = train_one_epoch(
            gen, disc, train_loader, opt_g, opt_d, losses, args, device, epoch, scaler
        )

        val_metrics = {"l1": None, "psnr": None, "ssim": None}
        if val_loader is not None:
            val_metrics = validate(gen, val_loader, losses["l1"], device)

        elapsed = time.perf_counter() - t0
        row = {
            "epoch": epoch + 1,
            "train_g": train_metrics["g"],
            "train_d": train_metrics["d"],
            "train_l1": train_metrics["l1"],
            "train_perc": train_metrics["perc"],
            "train_ssim_loss": train_metrics["ssim_loss"],
            "train_gan": train_metrics["gan"],
            "val_l1": val_metrics["l1"],
            "val_psnr": val_metrics["psnr"],
            "val_ssim": val_metrics["ssim"],
            "seconds": elapsed,
        }
        append_metrics(metrics_path, row)

        val_text = ""
        if val_metrics["psnr"] is not None:
            val_text = (
                f" ValL1={val_metrics['l1']:.4f}"
                f" PSNR={val_metrics['psnr']:.2f}dB"
                f" SSIM={val_metrics['ssim']:.4f}"
            )

        print(
            f"AvgG={train_metrics['g']:.4f} AvgD={train_metrics['d']:.4f}"
            f"{val_text} time={elapsed:.1f}s"
        )

        improved = val_metrics["psnr"] is not None and val_metrics["psnr"] > best_psnr
        if improved:
            best_psnr = val_metrics["psnr"]
            save_checkpoint(os.path.join(args.save_dir, "best.pth"), epoch + 1, gen, disc, opt_g, opt_d, best_psnr, scaler)
            save_generator(os.path.join(args.save_dir, "best_gen.pth"), gen)
            print(f"Saved new best by PSNR: {best_psnr:.2f}dB")

        if (epoch + 1) % args.sample_every == 0:
            save_samples(gen, sample_loader, device, os.path.join(args.save_dir, "samples"), epoch + 1)

        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(os.path.join(args.save_dir, f"epoch_{epoch + 1:03d}.pth"), epoch + 1, gen, disc, opt_g, opt_d, best_psnr, scaler)

        save_checkpoint(os.path.join(args.save_dir, "latest.pth"), epoch + 1, gen, disc, opt_g, opt_d, best_psnr, scaler)
        save_generator(os.path.join(args.save_dir, "latest_gen.pth"), gen)

    save_generator(os.path.join(args.save_dir, "generator_final.pth"), gen)
    print(f"\nDone. Best PSNR: {best_psnr:.2f}dB")
    print(f"Inference: python infer.py --checkpoint {args.save_dir}/best_gen.pth --input IMAGE --output OUT.png")


if __name__ == "__main__":
    main()
