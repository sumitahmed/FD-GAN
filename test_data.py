"""Quick verification that dataset loads correctly with real data."""
from dataset import DehazingDataset

# Train
ds = DehazingDataset('data/train/hazy', 'data/train/clean', crop_size=256)
print(f"Train: {len(ds)} pairs")
b = ds[0]
print(f"  hazy:  {b['hazy'].shape}  range=[{b['hazy'].min():.2f}, {b['hazy'].max():.2f}]")
print(f"  clean: {b['clean'].shape}  range=[{b['clean'].min():.2f}, {b['clean'].max():.2f}]")
print(f"  name:  {b['name']}")

# Test
ds2 = DehazingDataset('data/test/hazy', 'data/test/clean', crop_size=None, augment=False)
print(f"\nTest: {len(ds2)} pairs")
b2 = ds2[0]
print(f"  hazy:  {b2['hazy'].shape}")
print(f"  clean: {b2['clean'].shape}")

print("\nDataset OK!")
