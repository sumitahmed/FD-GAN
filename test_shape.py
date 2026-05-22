"""Verify model output shapes for various input sizes."""
import torch
from model import FDGANGenerator

model = FDGANGenerator(pretrained_encoder=False)
model.eval()

# Test multiple resolutions including odd sizes
test_sizes = [
    (256, 320),   # Paper training size
    (256, 256),
    (480, 640),
    (333, 501),   # Odd dimensions guarded by final resize
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
