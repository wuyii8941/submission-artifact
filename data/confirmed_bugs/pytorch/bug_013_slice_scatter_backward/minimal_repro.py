"""
Bug 013: torch.slice_scatter backward produces wrong gradients under torch.compile

Inductor incorrectly computes x.grad for slice_scatter — positions that should
retain gradient from x are either zeroed or set to all-ones.

Reproduces on: PyTorch 2.11.0+cu126, 2.12.0.dev20260410+cu126 (nightly)
Devices: both CUDA and CPU
"""
import torch

def test(desc, x, y, dim, start, end):
    def f(x, y):
        return torch.slice_scatter(x, y, dim=dim, start=start, end=end).sum()

    x1 = x.clone().requires_grad_(True)
    y1 = y.clone().requires_grad_(True)
    f(x1, y1).backward()

    x2 = x.clone().requires_grad_(True)
    y2 = y.clone().requires_grad_(True)
    torch._dynamo.reset()
    fn = torch.compile(f, backend='inductor')
    fn(x2, y2).backward()
    torch._dynamo.reset()

    diff = (x1.grad - x2.grad).abs().max().item()
    status = "✅" if diff == 0 else "❌"
    print(f"{status} {desc}")
    print(f"   eager   x.grad = {x1.grad.flatten().tolist()}")
    print(f"   inductor x.grad = {x2.grad.flatten().tolist()}")
    print(f"   diff = {diff}")
    print()

print("=== CUDA ===")
test("start=0,end=2", torch.randn(5, device='cuda'), torch.randn(2, device='cuda'), 0, 0, 2)
test("start=2,end=4", torch.randn(5, device='cuda'), torch.randn(2, device='cuda'), 0, 2, 4)
test("start=3,end=5", torch.randn(5, device='cuda'), torch.randn(2, device='cuda'), 0, 3, 5)
test("2D,dim=1",      torch.randn(3,5, device='cuda'), torch.randn(3,2, device='cuda'), 1, 0, 2)

print("=== CPU ===")
test("start=0,end=2", torch.randn(5), torch.randn(2), 0, 0, 2)
