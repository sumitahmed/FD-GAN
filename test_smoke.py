"""
Smoke test: verify the full GAN training loop runs for 1 step
using synthetic dummy data. No real dataset needed.
"""
import os
import shutil
import tempfile

import torch
import numpy as np
from PIL import Image

# Create tiny dummy dataset in a temp dir
tmpdir = os.path.join(os.path.dirname(__file__), "_smoke_test_data")
hazy_dir = os.path.join(tmpdir, "hazy")
clean_dir = os.path.join(tmpdir, "clean")
os.makedirs(hazy_dir, exist_ok=True)
os.makedirs(clean_dir, exist_ok=True)

# Generate 4 random 256x256 images
for i in range(4):
    hazy_img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    clean_img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    hazy_img.save(os.path.join(hazy_dir, f"{i:03d}.png"))
    clean_img.save(os.path.join(clean_dir, f"{i:03d}.png"))

print("Created dummy dataset")

# Test imports
from model import FDGANGenerator
from discriminator import FusionDiscriminator
from losses import VGGRelu12Loss, GANLoss
from dataset import DehazingDataset
print("Imports OK")

# Test dataset
ds = DehazingDataset(hazy_dir, clean_dir, crop_size=128, augment=True)
print(f"Dataset: {len(ds)} pairs")

batch = ds[0]
print(f"  hazy:  {batch['hazy'].shape}")
print(f"  clean: {batch['clean'].shape}")

# Test models
device = torch.device("cpu")
gen = FDGANGenerator(pretrained_encoder=False).to(device)
disc = FusionDiscriminator(fusion_mode="full", ndf=32).to(device)
print(f"Generator:     {sum(p.numel() for p in gen.parameters()):,} params")
print(f"Discriminator: {sum(p.numel() for p in disc.parameters()):,} params")

# Test forward pass
hazy_t = batch["hazy"].unsqueeze(0).to(device)
clean_t = batch["clean"].unsqueeze(0).to(device)

gen.train()
disc.train()

fake = gen(hazy_t)
print(f"Generator output: {fake.shape}")

pred_real = disc(clean_t)
pred_fake = disc(fake.detach())
print(f"Discriminator output: {pred_real.shape}")

# Test losses
criterion_l1 = torch.nn.L1Loss()
criterion_perc = VGGRelu12Loss().to(device)
criterion_gan = GANLoss().to(device)

l1 = criterion_l1(fake, clean_t)
perc = criterion_perc(fake, clean_t)
gan_g = criterion_gan(pred_fake, is_real=True)
gan_d = criterion_gan(pred_real, is_real=True)

print(f"Losses - L1: {l1.item():.4f}, Perceptual: {perc.item():.4f}, GAN: {gan_g.item():.4f}")

# Test backward
total_loss = l1 + perc + gan_g
total_loss.backward()
print("Backward pass OK")

# Cleanup
shutil.rmtree(tmpdir)
print("\n=== ALL SMOKE TESTS PASSED ===")
