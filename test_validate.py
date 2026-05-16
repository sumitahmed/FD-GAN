"""
Full end-to-end validation: runs 3 actual training steps with dummy data.
Tests dataset loading, G/D forward+backward, loss computation, checkpointing, 
and inference from saved checkpoint — the EXACT flow you'll run on Colab.
"""
import os
import shutil
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import ModernFDGAN
from discriminator import NLayerDiscriminator
from losses import VGGPerceptualLoss, GANLoss
from dataset import DehazingDataset


def main():
    print("=" * 60)
    print("  FD-GAN FULL PIPELINE VALIDATION")
    print("=" * 60)

    device = torch.device("cpu")
    tmpdir = os.path.join(os.path.dirname(__file__), "_validation_tmp")
    ckpt_dir = os.path.join(tmpdir, "checkpoints")

    try:
        # ── 1. Create dummy dataset ────────────────────────────────
        print("\n[1/7] Creating dummy dataset...")
        hazy_dir = os.path.join(tmpdir, "hazy")
        clean_dir = os.path.join(tmpdir, "clean")
        os.makedirs(hazy_dir, exist_ok=True)
        os.makedirs(clean_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)

        for i in range(8):
            # Clean = random, Hazy = clean + fog effect (lighter)
            clean = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
            fog = np.random.randint(150, 220, (256, 256, 3), dtype=np.uint8)
            hazy = ((clean.astype(float) * 0.5 + fog.astype(float) * 0.5)).astype(np.uint8)
            Image.fromarray(hazy).save(os.path.join(hazy_dir, f"{i:03d}.png"))
            Image.fromarray(clean).save(os.path.join(clean_dir, f"{i:03d}.png"))
        print("  OK - 8 dummy pairs created")

        # ── 2. Dataset + DataLoader ────────────────────────────────
        print("\n[2/7] Testing dataset loading...")
        ds = DehazingDataset(hazy_dir, clean_dir, crop_size=128, augment=True)
        loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0, drop_last=True)
        batch = next(iter(loader))
        assert batch["hazy"].shape == (2, 3, 128, 128)
        assert batch["clean"].shape == (2, 3, 128, 128)
        print(f"  OK - batch shapes: hazy={batch['hazy'].shape}, clean={batch['clean'].shape}")

        # ── 3. Models ──────────────────────────────────────────────
        print("\n[3/7] Initializing models...")
        gen = ModernFDGAN().to(device)
        disc = NLayerDiscriminator(in_channels=6, ndf=64, n_layers=3).to(device)
        g_params = sum(p.numel() for p in gen.parameters())
        d_params = sum(p.numel() for p in disc.parameters())
        print(f"  OK - Generator: {g_params:,} params, Discriminator: {d_params:,} params")

        # ── 4. Losses ──────────────────────────────────────────────
        print("\n[4/7] Initializing losses...")
        criterion_l1 = nn.L1Loss()
        criterion_perc = VGGPerceptualLoss().to(device)
        criterion_gan = GANLoss().to(device)
        print("  OK - L1 + VGG Perceptual + LSGAN")

        # ── 5. Training steps ──────────────────────────────────────
        print("\n[5/7] Running 3 training steps...")
        opt_g = optim.Adam(gen.parameters(), lr=2e-4, betas=(0.5, 0.999))
        opt_d = optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.999))

        gen.train()
        disc.train()

        for step, batch in enumerate(loader):
            if step >= 3:
                break

            hazy = batch["hazy"].to(device)
            clean = batch["clean"].to(device)

            # ── D step ──
            opt_d.zero_grad()
            with torch.no_grad():
                fake = gen(hazy)
            loss_d_real = criterion_gan(disc(hazy, clean), is_real=True)
            loss_d_fake = criterion_gan(disc(hazy, fake.detach()), is_real=False)
            loss_d = (loss_d_real + loss_d_fake) * 0.5
            loss_d.backward()
            opt_d.step()

            # ── G step ──
            opt_g.zero_grad()
            fake = gen(hazy)
            loss_g_gan = criterion_gan(disc(hazy, fake), is_real=True)
            loss_g_l1 = criterion_l1(fake, clean)
            loss_g_perc = criterion_perc(fake, clean)
            loss_g = loss_g_gan + 10.0 * loss_g_l1 + loss_g_perc
            loss_g.backward()
            opt_g.step()

            print(
                f"  Step {step+1}: G={loss_g.item():.4f} "
                f"(L1={loss_g_l1.item():.4f}, Perc={loss_g_perc.item():.4f}, "
                f"GAN={loss_g_gan.item():.4f})  D={loss_d.item():.4f}"
            )

        print("  OK - all training steps completed")

        # ── 6. Checkpoint save/load ────────────────────────────────
        print("\n[6/7] Testing checkpoint save/load...")
        ckpt_path = os.path.join(ckpt_dir, "test.pth")
        gen_path = os.path.join(ckpt_dir, "generator.pth")

        # Save full checkpoint
        torch.save({
            "epoch": 1,
            "generator": gen.state_dict(),
            "discriminator": disc.state_dict(),
            "optimizer_g": opt_g.state_dict(),
            "optimizer_d": opt_d.state_dict(),
            "best_loss": 99.0,
        }, ckpt_path)

        # Save generator-only
        torch.save(gen.state_dict(), gen_path)

        # Load into fresh model
        gen2 = ModernFDGAN().to(device)
        gen2.load_state_dict(torch.load(gen_path, map_location=device, weights_only=True))
        print(f"  OK - checkpoint saved ({os.path.getsize(ckpt_path)/1e6:.1f} MB) and reloaded")

        # ── 7. Inference from checkpoint ───────────────────────────
        print("\n[7/7] Testing inference from saved checkpoint...")
        gen2.eval()
        test_input = torch.randn(1, 3, 200, 300).to(device)
        with torch.no_grad():
            test_output = gen2(test_input)
        assert test_output.shape == (1, 3, 200, 300), f"Bad shape: {test_output.shape}"

        # Convert to image
        img = test_output.squeeze(0).cpu()
        img = ((img + 1.0) / 2.0).clamp(0, 1)
        img = (img.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        result = Image.fromarray(img, mode="RGB")
        result_path = os.path.join(tmpdir, "test_result.png")
        result.save(result_path)
        print(f"  OK - inference output: {test_output.shape}, saved as {result.size} RGB image")

        # ── Done ───────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("  ALL VALIDATIONS PASSED - READY FOR TRAINING!")
        print("=" * 60)

    finally:
        # Cleanup
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)


if __name__ == "__main__":
    main()
