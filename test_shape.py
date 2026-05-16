"""Verify model output shapes for various input sizes."""
import torch
from model import ModernFDGAN

model = ModernFDGAN()
model.eval()

# Test multiple resolutions including odd sizes
test_sizes = [
    (256, 256),
    (512, 512),
    (480, 640),
    (333, 501),   # Odd dimensions
    (720, 1280),  # HD
    (100, 100),   # Small
]

print("=" * 65)
print(f"  {'Input':>14s}  ->  {'Output':>20s}  {'Status'}")
print("=" * 65)

all_passed = True
for h, w in test_sizes:
    x = torch.randn(1, 3, h, w)
    with torch.no_grad():
        y = model(x)

    expected = (1, 3, h, w)
    status = "PASS" if y.shape == torch.Size(expected) else "FAIL"
    if y.shape != torch.Size(expected):
        all_passed = False

    print(f"  ({h:>4d}, {w:>4d})       ->  {str(tuple(y.shape)):>20s}  {status}")

print("=" * 65)
print(f"\n{'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
