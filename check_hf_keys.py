"""Inspect HuggingFace FDGAN checkpoint keys to understand its architecture."""
import torch
import os

ckpt_path = os.path.join(
    os.path.expanduser("~"),
    ".cache", "huggingface", "hub",
    "models--Ramssesdlsm--FDGAN",
    "snapshots",
    "ab6d68914c43b86699366cbf085470dae714aff8",
    "FDGAN-generator.pth"
)

print(f"Loading: {ckpt_path}")
state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

if isinstance(state, dict) and "generator" in state:
    state = state["generator"]
elif isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]

print(f"\nTotal keys: {len(state)}")
print(f"Total params: {sum(v.numel() for v in state.values()):,}")

# Show first 30 keys
print("\n-- First 30 keys --")
for i, (k, v) in enumerate(sorted(state.items())):
    if i >= 30:
        print(f"  ... and {len(state) - 30} more")
        break
    print(f"  {k}: {list(v.shape)}")

# Show top-level prefixes
prefixes = set()
for k in state.keys():
    parts = k.split(".")
    if len(parts) >= 2:
        prefixes.add(parts[0] + "." + parts[1])
    else:
        prefixes.add(parts[0])

print(f"\n-- Top-level prefixes ({len(prefixes)}) --")
for p in sorted(prefixes)[:30]:
    count = sum(1 for k in state if k.startswith(p))
    print(f"  {p} ({count} keys)")

# Also show our model keys for comparison
from model import ModernFDGAN
model = ModernFDGAN()
our_state = model.state_dict()

print(f"\n-- Our model first 20 keys --")
for i, (k, v) in enumerate(sorted(our_state.items())):
    if i >= 20:
        break
    print(f"  {k}: {list(v.shape)}")
