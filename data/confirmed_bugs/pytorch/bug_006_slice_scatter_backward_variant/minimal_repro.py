"""
Bug 013 variant (#<anon>): slice_scatter backward wrong gradient when slice excludes index 0.

This is a distinct trigger condition from #<anon> (which fires when the slice
includes index 0). Here the overwritten region starts after index 0, triggering
a different stride-0 broadcasting path in Inductor's backward codegen.

Related: #<anon> (bug_013), #<anon> (bug_007)
"""
import torch

def test_variant():
    # Slice that EXCLUDES index 0 (start=1)
    x = torch.randn(4, 4, device='cuda', requires_grad=True)
    y = torch.randn(2, 4, device='cuda', requires_grad=True)

    def f(x, y):
        return torch.slice_scatter(x, y, dim=0, start=1, end=3).sum()

    # Eager
    x1, y1 = x.clone().requires_grad_(True), y.clone().requires_grad_(True)
    f(x1, y1).backward()
    eager_grad = x1.grad.clone()

    # Compiled
    x2, y2 = x.clone().requires_grad_(True), y.clone().requires_grad_(True)
    torch._dynamo.reset()
    compiled_f = torch.compile(f, backend='inductor')
    compiled_f(x2, y2).backward()
    compiled_grad = x2.grad.clone()

    diff = (eager_grad - compiled_grad).abs().max().item()
    print(f"Max gradient diff: {diff}")
    if diff > 0:
        print(f"  Eager x.grad:\n{eager_grad}")
        print(f"  Compiled x.grad:\n{compiled_grad}")
        print("BUG CONFIRMED: gradient mismatch when slice excludes index 0")
    else:
        print("OK (may be fixed in this PyTorch version)")

if __name__ == '__main__':
    test_variant()
