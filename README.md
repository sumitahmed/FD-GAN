# FD-GAN — Single Image Dehazing

A modern PyTorch reimplementation of the FD-GAN architecture for single image dehazing, trained on the RESIDE-6K benchmark.

> **FD-GAN: Generative Adversarial Networks with Fusion-Discriminator for Single Image Dehazing**
> Yu Dong, et al. — AAAI 2020

This is not a direct port of the original repository. The original codebase relies on PyTorch 0.3 with broken checkpoints and architecture mismatches. This project is a ground-up reimplementation using PyTorch 2.0+.

---

## Architecture

The generator uses a DenseNet-121 backbone (pretrained on ImageNet) as the encoder, paired with a U-Net style decoder that uses skip connections at multiple resolutions. The output is produced via residual learning — the network predicts a residual that is added to the input.

The discriminator is a PatchGAN operating on 70×70 receptive fields, conditioned on the hazy input concatenated with the output.

| Component | Details |
|-----------|---------|
| Generator | DenseNet-121 encoder, U-Net decoder, 5 skip connections |
| Discriminator | PatchGAN, conditional, 6-channel input |
| Losses | L1 pixel (×10) + VGG-16 perceptual + LSGAN adversarial |
| Upsampling | Bilinear interpolation |
| Output mode | Residual learning (`output = input + residual`) |

```
Encoder (DenseNet-121)              Decoder
  stem       → 64ch,  H/4   ──→    up4 → 64ch,  H/2
  denseblock1 → 128ch, H/8  ──→    up3 → 128ch, H/4
  denseblock2 → 256ch, H/16 ──→    up2 → 256ch, H/8
  denseblock3 → 512ch, H/32 ──→    up1 → 512ch, H/16
  bottleneck  → 1024ch, H/32        up5 → 32ch,  H/1 → head → 3ch residual
```

---

## Usage

### Installation

```bash
pip install -r requirements.txt
```

### Inference

```bash
python infer.py --checkpoint best_gen.pth --input path/to/hazy.jpg --output dehazed.png
```

The checkpoint file (`best_gen.pth`, ~75MB) is not included in the repository due to size. Train your own or download from the Releases page if available.

### Training

```bash
# Local training
python train.py --hazy_dir data/train/hazy --clean_dir data/train/clean --epochs 300

# Resume from checkpoint
python train.py --hazy_dir data/train/hazy --clean_dir data/train/clean \
    --resume checkpoints/latest.pth
```

A Kaggle notebook (`FD_GAN_Training.ipynb`) is included for cloud GPU training.

### Dataset

This model was trained on [RESIDE-6K](https://www.kaggle.com/datasets/kmljts/reside-6k), a 6,000-image subset of the RESIDE benchmark containing paired hazy and ground-truth images.

Any paired dataset will work with the following structure:

```
data/
  train/
    hazy/
    clean/
  test/
    hazy/
    clean/
```

---

## Training Details

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch size | 8 |
| Learning rate | 2e-4 |
| Optimizer | Adam (β1=0.5, β2=0.999) |
| LR schedule | Linear decay after 50% of training |
| Crop size | 256×256 |
| L1 weight | 10.0 |
| Perceptual weight | 1.0 |
| GAN weight | 1.0 |

### Current Training Run

Trained for 88 epochs on RESIDE-6K using Kaggle (GPU T4 ×2, ~12 hours total).

| Metric | Value |
|--------|-------|
| Final G loss | 2.71 |
| Final D loss | 0.009 |
| Best epoch | 88 |

### Known Limitations

The discriminator collapsed early in training (D loss dropped to near-zero), which means the adversarial component stopped contributing meaningful gradients. The generator was effectively trained with L1 + VGG perceptual loss only.

The current model produces visible haze reduction but does not reach paper-quality metrics. For reference:

| | Current (est.) | Paper-grade |
|---|---|---|
| PSNR | ~25–28 dB | 30–35 dB |
| SSIM | ~0.90–0.93 | 0.95–0.98 |

To improve results:

- Train for 300–500 epochs on a larger dataset (RESIDE OTS, 72K images)
- Fix discriminator collapse: lower G learning rate, increase D update frequency, add spectral normalization
- Add SSIM loss and multi-scale discrimination
- Use larger batch sizes (16–32)

---

## Project Structure

```
FD-GAN2/
├── model.py              Generator (DenseNet-121 encoder + U-Net decoder)
├── discriminator.py      PatchGAN discriminator
├── losses.py             L1, VGG-16 perceptual, LSGAN losses
├── dataset.py            Paired dataset loader with augmentation
├── train.py              Training loop with checkpointing
├── infer.py              Inference CLI
├── FD_GAN_Training.ipynb Kaggle training notebook
├── prepare_data.py       Dataset directory setup
├── setup_dataset.py      RESIDE dataset downloader
├── colab_train.py        Google Colab training script
├── test_shape.py         Architecture shape tests
├── test_smoke.py         Pipeline smoke tests
├── test_data.py          Dataset loading tests
├── test_validate.py      Validation tests
├── requirements.txt
└── sample_images/
```

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- torchvision
- Pillow
- NumPy

GPU strongly recommended for training. Inference runs on CPU.

---

## Verification

```bash
python test_shape.py      # Architecture shape checks
python test_smoke.py      # Full pipeline smoke test
python test_data.py       # Dataset loading verification
```

---

## References

- Yu Dong et al., "FD-GAN: Generative Adversarial Networks with Fusion-Discriminator for Single Image Dehazing," AAAI 2020. [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/6701)
- Huang et al., "Densely Connected Convolutional Networks," CVPR 2017. [arXiv:1608.06993](https://arxiv.org/abs/1608.06993)
- Isola et al., "Image-to-Image Translation with Conditional Adversarial Networks," CVPR 2017. [arXiv:1611.07004](https://arxiv.org/abs/1611.07004)
- Simonyan & Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition," ICLR 2015. [arXiv:1409.1556](https://arxiv.org/abs/1409.1556)
- Li et al., "Benchmarking Single-Image Dehazing and Beyond," IEEE TIP 2019. [IEEE](https://ieeexplore.ieee.org/document/8451944)

---

## License

MIT