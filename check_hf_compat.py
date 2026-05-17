"""
Quick compatibility check: Does the HuggingFace FDGAN-generator.pth
match our ModernFDGAN architecture?

Downloads the checkpoint and compares state_dict keys/shapes.
"""

import torch
import sys
import os

# First, check if huggingface_hub is available
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Installing huggingface_hub...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
    from huggingface_hub import hf_hub_download

from model import ModernFDGAN


def main():
    # ── Step 1: Download checkpoint from HuggingFace ─────────────
    print("Downloading FDGAN-generator.pth from Ramssesdlsm/FDGAN ...")
    ckpt_path = hf_hub_download(
        repo_id="Ramssesdlsm/FDGAN",
        filename="FDGAN-generator.pth",
    )
    print(f"Downloaded to: {ckpt_path}")

    # ── Step 2: Load the HF state dict ───────────────────────────
    print("\nLoading HuggingFace checkpoint...")
    hf_state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Handle the case where it's a full training checkpoint dict
    if isinstance(hf_state, dict) and "generator" in hf_state:
        print("  → Found 'generator' key inside checkpoint (full training checkpoint)")
        hf_state = hf_state["generator"]
    elif isinstance(hf_state, dict) and "state_dict" in hf_state:
        print("  → Found 'state_dict' key inside checkpoint")
        hf_state = hf_state["state_dict"]

    hf_keys = set(hf_state.keys())
    print(f"  HF checkpoint has {len(hf_keys)} keys")
    hf_params = sum(v.numel() for v in hf_state.values())
    hf_size_mb = sum(v.numel() * v.element_size() for v in hf_state.values()) / 1e6
    print(f"  HF total params: {hf_params:,}  ({hf_size_mb:.1f} MB)")

    # ── Step 3: Build our model and get its state dict ───────────
    print("\nBuilding our ModernFDGAN model...")
    model = ModernFDGAN()
    our_state = model.state_dict()
    our_keys = set(our_state.keys())
    our_params = sum(v.numel() for v in our_state.values())
    our_size_mb = sum(v.numel() * v.element_size() for v in our_state.values()) / 1e6
    print(f"  Our model has {len(our_keys)} keys")
    print(f"  Our total params: {our_params:,}  ({our_size_mb:.1f} MB)")

    # ── Step 4: Compare keys ─────────────────────────────────────
    print("\n" + "=" * 60)
    matching = hf_keys & our_keys
    only_hf = hf_keys - our_keys
    only_ours = our_keys - hf_keys

    print(f"Matching keys:      {len(matching)}")
    print(f"Only in HF ckpt:    {len(only_hf)}")
    print(f"Only in our model:  {len(only_ours)}")

    if only_hf:
        print("\n── Keys ONLY in HuggingFace checkpoint ──")
        for k in sorted(only_hf)[:20]:
            print(f"  {k}: {hf_state[k].shape}")
        if len(only_hf) > 20:
            print(f"  ... and {len(only_hf) - 20} more")

    if only_ours:
        print("\n── Keys ONLY in our model ──")
        for k in sorted(only_ours)[:20]:
            print(f"  {k}: {our_state[k].shape}")
        if len(only_ours) > 20:
            print(f"  ... and {len(only_ours) - 20} more")

    # ── Step 5: Check shape mismatches ───────────────────────────
    shape_mismatches = []
    for k in matching:
        if hf_state[k].shape != our_state[k].shape:
            shape_mismatches.append((k, hf_state[k].shape, our_state[k].shape))

    if shape_mismatches:
        print(f"\n── Shape mismatches ({len(shape_mismatches)}) ──")
        for k, hf_shape, our_shape in shape_mismatches:
            print(f"  {k}: HF={hf_shape}  ours={our_shape}")

    # ── Step 6: Verdict ──────────────────────────────────────────
    print("\n" + "=" * 60)
    if not only_hf and not only_ours and not shape_mismatches:
        print("✅ PERFECT MATCH! The HF checkpoint is fully compatible.")
        print("   You can load it directly with model.load_state_dict()")
        
        # Actually try loading it
        print("\nAttempting load...")
        model.load_state_dict(hf_state)
        print("✅ load_state_dict succeeded!")
        
        # Copy to project
        local_path = os.path.join("checkpoints", "hf_FDGAN_generator.pth")
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(hf_state, local_path)
        print(f"\nSaved to: {local_path}")
        print(f"\nRun inference with:")
        print(f"  python infer.py --input test_hazy.png --checkpoint {local_path}")
        
    elif not shape_mismatches and len(only_hf) == 0:
        print("⚠️  PARTIAL MATCH: HF checkpoint is a subset of our model.")
        print("   Use strict=False to load partial weights.")
    else:
        print("❌ INCOMPATIBLE: Different architecture.")
        print("   Cannot directly load this checkpoint.")
        
        # Show first few HF keys to understand architecture
        print("\n── Sample HF keys (first 15) ──")
        for k in sorted(hf_keys)[:15]:
            print(f"  {k}: {hf_state[k].shape}")


if __name__ == "__main__":
    main()
