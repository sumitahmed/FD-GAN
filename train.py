"""Train a paper-aligned FD-GAN dehazing model.

This is a modern PyTorch training script for the FD-GAN principles:
densely connected generator, Fusion-discriminator over RGB/LF/HF samples,
and the paper losses L1 + SSIM + VGG relu1_2 perceptual + adversarial.
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dataset import DehazingDataset
from discriminator import FusionDiscriminator
from losses import GANLoss, SSIMLoss, VGGRelu12Loss, denormalize, ssim_index
from model import FDGANGenerator


def parse_crop_size(value: str) -> tuple[int, int]:
    if "x" in value.lower():
        h, w = value.lower().split("x", 1)
        return int(h), int(w)
    size = int(value)
    return size, size


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train FD-GAN with Fusion-discriminator")
    p.add_argument("--hazy_dir", required=True)
    p.add_argument("--clean_dir", required=True)
    p.add_argument("--val_hazy_dir", default=None)
    p.add_argument("--val_clean_dir", default=None)

    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--crop_size", type=parse_crop_size, default=(256, 320), help="H or HxW; paper uses 256x320")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--amp", action="store_true")

    p.add_argument("--lr_g", type=float, default=2e-3)
    p.add_argument("--lr_d", type=float, default=2e-3)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--beta2", type=float, default=0.999)

    # Paper weights alpha1, alpha2, alpha3, alpha4 = 2, 1, 2, 0.1.
    p.add_argument("--lambda_l1", type=float, default=2.0)
    p.add_argument("--lambda_ssim", type=float, default=1.0)
    p.add_argument("--lambda_perceptual", type=float, default=2.0)
    p.add_argument("--lambda_gan", type=float, default=0.1)
    p.add_argument("--gan_start_epoch", type=int, default=0)
    p.add_argument("--fusion_mode", choices=["full", "lf", "hf", "rgb"], default="full")

    p.add_argument("--save_dir", default="checkpoints_fdgan")
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--sample_every", type=int, default=5)
    p.add_argument("--resume", default=None, help="Full checkpoint containing generator/discriminator/optimizers")
    p.add_argument("--pretrained_gen", default=None, help="Generator state_dict warm-start")
    p.add_argument("--no_pretrained_encoder", action="store_true")
    return p.parse_args()


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def load_generator_state(path: str, gen: nn.Module, device: torch.device) -> None:
    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "generator" in state:
        state = state["generator"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model" in state:
        state = state["model"]
    if any(k.startswith("module.") for k in state):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    unwrap(gen).load_state_dict(state, strict=True)


def save_generator(path: str, gen: nn.Module) -> None:
    torch.save(unwrap(gen).state_dict(), path)


def save_checkpoint(path, epoch, gen, disc, opt_g, opt_d, best_psnr, scaler):
    payload = {
        "epoch": epoch,
        "generator": unwrap(gen).state_dict(),
        "discriminator": unwrap(disc).state_dict(),
        "optimizer_g": opt_g.state_dict(),
        "optimizer_d": opt_d.state_dict(),
        "best_psnr": best_psnr,
    }
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, path)


def load_checkpoint(path, gen, disc, opt_g, opt_d, scaler, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    unwrap(gen).load_state_dict(ckpt["generator"], strict=True)
    unwrap(disc).load_state_dict(ckpt["discriminator"], strict=True)
    opt_g.load_state_dict(ckpt["optimizer_g"])
    opt_d.load_state_dict(ckpt["optimizer_d"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt["epoch"]), float(ckpt.get("best_psnr", -1e9))


def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pred = denormalize(pred)
    target = denormalize(target)
    mse = torch.mean((pred - target) ** 2)
    return 10.0 * torch.log10(1.0 / (mse + eps))


def append_metrics(path: str, row: dict) -> None:
    new_file = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def save_samples(gen, loader, device, save_dir, epoch, max_items=4):
    gen.eval()
    batch = next(iter(loader))
    hazy = batch["hazy"][:max_items].to(device)
    clean = batch["clean"][:max_items].to(device)
    with torch.no_grad():
        fake = gen(hazy)
    rows = []
    for i in range(hazy.shape[0]):
        rows.extend([denormalize(hazy[i]), denormalize(fake[i]), denormalize(clean[i])])
    os.makedirs(save_dir, exist_ok=True)
    save_image(rows, os.path.join(save_dir, f"epoch_{epoch:03d}.png"), nrow=3)


def train_one_epoch(gen, disc, loader, opt_g, opt_d, losses, args, device, epoch, scaler):
    gen.train()
    disc.train()
    use_amp = scaler is not None
    use_gan = epoch >= args.gan_start_epoch and args.lambda_gan > 0.0
    totals = {"g": 0.0, "d": 0.0, "l1": 0.0, "ssim": 0.0, "perc": 0.0, "gan": 0.0}

    for i, batch in enumerate(loader):
        hazy = batch["hazy"].to(device, non_blocking=True)
        clean = batch["clean"].to(device, non_blocking=True)

        opt_d.zero_grad(set_to_none=True)
        if use_gan:
            with torch.no_grad():
                fake_detached = gen(hazy)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred_real = disc(clean)
                pred_fake = disc(fake_detached.detach())
                loss_d = 0.5 * (losses["gan"](pred_real, True) + losses["gan"](pred_fake, False))
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
            loss_ssim = losses["ssim"](fake, clean)
            loss_perc = losses["perc"](fake, clean)
            loss_gan = losses["gan"](disc(fake), True) if use_gan else torch.zeros((), device=device)
            loss_g = (
                args.lambda_l1 * loss_l1
                + args.lambda_ssim * loss_ssim
                + args.lambda_perceptual * loss_perc
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
        totals["ssim"] += loss_ssim.item()
        totals["perc"] += loss_perc.item()
        totals["gan"] += loss_gan.item()

        if (i + 1) % 50 == 0 or (i + 1) == len(loader):
            print(
                f"  [{i+1}/{len(loader)}] G={loss_g.item():.4f} D={loss_d.item():.4f} "
                f"L1={loss_l1.item():.4f} SSIMLoss={loss_ssim.item():.4f} "
                f"Perc={loss_perc.item():.4f} GAN={loss_gan.item():.4f}",
                flush=True,
            )

    return {k: v / len(loader) for k, v in totals.items()}


@torch.no_grad()
def validate(gen, loader, losses, device):
    gen.eval()
    totals = {"l1": 0.0, "ssim_loss": 0.0, "psnr": 0.0, "ssim": 0.0}
    for batch in loader:
        hazy = batch["hazy"].to(device, non_blocking=True)
        clean = batch["clean"].to(device, non_blocking=True)
        fake = gen(hazy)
        totals["l1"] += losses["l1"](fake, clean).item()
        totals["ssim_loss"] += losses["ssim"](fake, clean).item()
        totals["psnr"] += psnr(fake, clean).item()
        totals["ssim"] += ssim_index(fake, clean).item()
    return {k: v / len(loader) for k, v in totals.items()}


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"CUDA devices: {torch.cuda.device_count()}")

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

    gen = FDGANGenerator(pretrained_encoder=not args.no_pretrained_encoder).to(device)
    disc = FusionDiscriminator(fusion_mode=args.fusion_mode).to(device)

    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        gen = nn.DataParallel(gen)
        disc = nn.DataParallel(disc)

    opt_g = optim.Adam(gen.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    opt_d = optim.Adam(disc.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))
    scaler = scaler if scaler.is_enabled() else None

    losses = {
        "l1": nn.L1Loss(),
        "ssim": SSIMLoss(),
        "perc": VGGRelu12Loss().to(device),
        "gan": GANLoss().to(device),
    }

    os.makedirs(args.save_dir, exist_ok=True)
    start_epoch = 0
    best_psnr = -1e9
    if args.resume:
        start_epoch, best_psnr = load_checkpoint(args.resume, gen, disc, opt_g, opt_d, scaler, device)
        print(f"Resumed from epoch {start_epoch}, best_psnr={best_psnr:.2f}")
    elif args.pretrained_gen:
        load_generator_state(args.pretrained_gen, gen, device)
        print(f"Warm-started generator from {args.pretrained_gen}")

    print(f"Generator params: {sum(p.numel() for p in gen.parameters()):,}")
    print(f"Discriminator params: {sum(p.numel() for p in disc.parameters()):,}")
    print(f"Crop={args.crop_size}, batch={args.batch_size}, fusion={args.fusion_mode}")
    print(
        f"Loss weights: L1={args.lambda_l1}, SSIM={args.lambda_ssim}, "
        f"Perceptual={args.lambda_perceptual}, GAN={args.lambda_gan}"
    )

    metrics_path = os.path.join(args.save_dir, "metrics.csv")
    sample_loader = val_loader if val_loader is not None else train_loader

    for epoch in range(start_epoch, args.epochs):
        t0 = time.perf_counter()
        print(f"\nEpoch {epoch + 1}/{args.epochs}", flush=True)
        train_metrics = train_one_epoch(gen, disc, train_loader, opt_g, opt_d, losses, args, device, epoch, scaler)
        val_metrics = {"l1": None, "ssim_loss": None, "psnr": None, "ssim": None}
        if val_loader is not None:
            val_metrics = validate(gen, val_loader, losses, device)

        elapsed = time.perf_counter() - t0
        row = {
            "epoch": epoch + 1,
            "train_g": train_metrics["g"],
            "train_d": train_metrics["d"],
            "train_l1": train_metrics["l1"],
            "train_ssim_loss": train_metrics["ssim"],
            "train_perc": train_metrics["perc"],
            "train_gan": train_metrics["gan"],
            "val_l1": val_metrics["l1"],
            "val_ssim_loss": val_metrics["ssim_loss"],
            "val_psnr": val_metrics["psnr"],
            "val_ssim": val_metrics["ssim"],
            "seconds": elapsed,
        }
        append_metrics(metrics_path, row)

        val_text = ""
        if val_metrics["psnr"] is not None:
            val_text = f" ValL1={val_metrics['l1']:.4f} PSNR={val_metrics['psnr']:.2f}dB SSIM={val_metrics['ssim']:.4f}"
        print(f"AvgG={train_metrics['g']:.4f} AvgD={train_metrics['d']:.4f}{val_text} time={elapsed:.1f}s")

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
