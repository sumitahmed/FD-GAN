# FD-GAN2

Modern PyTorch implementation of the core method from:

**FD-GAN: Generative Adversarial Networks with Fusion-discriminator for Single Image Dehazing**  
Yu Dong, Yihao Liu, He Zhang, Shifeng Chen, Yu Qiao, AAAI 2020 / arXiv 2021.

This code is not a direct PyTorch 0.3 port. It keeps the research method intact while making it practical on current PyTorch/Kaggle:

- end-to-end hazy image -> haze-free image generation;
- densely connected encoder-decoder generator inspired by the official `WeilanAnnn/FD-GAN` implementation;
- Fusion-discriminator over `[image, Gaussian low-frequency image, Laplacian high-frequency image]`;
- paper losses: L1, SSIM, VGG16 relu1_2 perceptual loss, and adversarial loss;
- paper default loss weights: `2, 1, 2, 0.1`;
- paper default crop size: `256x320`;
- AMP, multi-GPU `DataParallel`, validation metrics, sample grids, full resume checkpoints.

## Core Files

```text
model.py              FDGANGenerator
discriminator.py      FusionDiscriminator with LF/HF extraction
losses.py             L1/SSIM/VGG/GAN loss helpers
dataset.py            paired PNG/JPG RESIDE loader
train.py              training, validation, checkpointing
infer.py              single-image inference
test_shape.py         generator shape check
test_data.py          dataset pairing check
test_smoke.py         forward/loss/backward smoke test
```

## Install

```bash
pip install -r requirements.txt
```

## Verify

```bash
python test_shape.py
python test_data.py
python test_smoke.py
```

## Train

Short sanity run:

```bash
python train.py \
  --hazy_dir data/train/hazy \
  --clean_dir data/train/clean \
  --val_hazy_dir data/test/hazy \
  --val_clean_dir data/test/clean \
  --epochs 2 \
  --batch_size 8 \
  --crop_size 256x320 \
  --lr_g 0.002 \
  --lr_d 0.002 \
  --lambda_l1 2 \
  --lambda_ssim 1 \
  --lambda_perceptual 2 \
  --lambda_gan 0.1 \
  --fusion_mode full \
  --save_dir checkpoints_sanity \
  --sample_every 1 \
  --save_every 1 \
  --amp
```

Real run:

```bash
python train.py \
  --hazy_dir data/train/hazy \
  --clean_dir data/train/clean \
  --val_hazy_dir data/test/hazy \
  --val_clean_dir data/test/clean \
  --epochs 150 \
  --batch_size 8 \
  --crop_size 256x320 \
  --lr_g 0.002 \
  --lr_d 0.002 \
  --lambda_l1 2 \
  --lambda_ssim 1 \
  --lambda_perceptual 2 \
  --lambda_gan 0.1 \
  --fusion_mode full \
  --save_dir checkpoints_fdgan \
  --sample_every 5 \
  --save_every 5 \
  --amp
```

Resume:

```bash
python train.py \
  --hazy_dir data/train/hazy \
  --clean_dir data/train/clean \
  --val_hazy_dir data/test/hazy \
  --val_clean_dir data/test/clean \
  --resume checkpoints_fdgan/latest.pth \
  --epochs 150 \
  --batch_size 8 \
  --crop_size 256x320 \
  --save_dir checkpoints_fdgan \
  --amp
```

## Inference

```bash
python infer.py \
  --checkpoint checkpoints_fdgan/best_gen.pth \
  --input sample_images/hazy.jpg \
  --output outputs/fdgan_result.png
```

## Notes

The paper reports best results with a COCO-derived synthetic dataset using estimated depth. For a 24-hour Kaggle limit, RESIDE-6K is the practical training set. The algorithm here still follows FD-GAN; the dataset is the compromise.

Official references:

- Paper: https://arxiv.org/abs/2001.06968
- Official code: https://github.com/WeilanAnnn/FD-GAN
