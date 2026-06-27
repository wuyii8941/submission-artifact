#!/usr/bin/env python3
"""Systematic check: which in-place ternary ops crash in torch.compile backward?"""
import torch

DEVICE = 'cuda'

def test_op(name, fn, *tensors):
    """Test if fn compiles + backward works."""
    # Eager baseline
    try:
        eager_args = [t.detach().clone().requires_grad_(True) for t in tensors]
        fn(*eager_args).sum().backward()
    except Exception as e:
        print(f"  {name}: EAGER FAILS ({str(e)[:60]})")
        return

    # Compiled
    torch._dynamo.reset()
    try:
        comp_args = [t.detach().clone().requires_grad_(True) for t in tensors]
        compiled = torch.compile(fn, backend='inductor')
        compiled(*comp_args).sum().backward()
        print(f"  {name}: PASS")
    except Exception as e:
        err = str(e).split('\n')[0][:100]
        print(f"  {name}: CRASH ({err})")

x = torch.randn(4, 4, device=DEVICE, requires_grad=True)
t1 = torch.randn(4, 4, device=DEVICE, requires_grad=True)
t2 = torch.randn(4, 4, device=DEVICE, requires_grad=True).abs() + 0.1

print("=== In-place ternary ops ===")

# addcmul_ variants
test_op("addcmul_(value=0.5)",
        lambda x, t1, t2: x.clone().addcmul_(t1, t2, value=0.5), x, t1, t2)
test_op("addcmul_(value=1.0)",
        lambda x, t1, t2: x.clone().addcmul_(t1, t2, value=1.0), x, t1, t2)
test_op("addcmul_(value=0.0)",
        lambda x, t1, t2: x.clone().addcmul_(t1, t2, value=0.0), x, t1, t2)

# addcdiv_ variants
test_op("addcdiv_(value=0.1)",
        lambda x, t1, t2: x.clone().addcdiv_(t1, t2, value=0.1), x, t1, t2)
test_op("addcdiv_(value=1.0)",
        lambda x, t1, t2: x.clone().addcdiv_(t1, t2, value=1.0), x, t1, t2)

# Out-of-place equivalents (should work)
print("\n=== Out-of-place equivalents ===")
test_op("torch.addcmul(value=0.5)",
        lambda x, t1, t2: torch.addcmul(x, t1, t2, value=0.5), x, t1, t2)
test_op("torch.addcdiv(value=0.1)",
        lambda x, t1, t2: torch.addcdiv(x, t1, t2, value=0.1), x, t1, t2)

# Manual equivalents (should work)
print("\n=== Manual equivalents ===")
test_op("x + 0.5 * t1 * t2",
        lambda x, t1, t2: x + 0.5 * t1 * t2, x, t1, t2)
test_op("x + 0.1 * t1 / t2",
        lambda x, t1, t2: x + 0.1 * t1 / t2, x, t1, t2)

# Other in-place ops for comparison
print("\n=== Other in-place ops (for comparison) ===")
test_op("lerp_(end, 0.3)",
        lambda x, end: x.clone().lerp_(end, 0.3), x, t1)
test_op("add_(t1, alpha=0.5)",
        lambda x, t1: x.clone().add_(t1, alpha=0.5), x, t1)
test_op("sub_(t1, alpha=0.5)",
        lambda x, t1: x.clone().sub_(t1, alpha=0.5), x, t1)
test_op("addmm_(m1, m2)",
        lambda x, m1, m2: x.clone().addmm_(m1, m2), x, t1, t2)

# Different dtypes
print("\n=== Different dtypes ===")
for dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
    try:
        x_d = torch.randn(4, 4, device=DEVICE, dtype=dtype, requires_grad=True)
        t1_d = torch.randn(4, 4, device=DEVICE, dtype=dtype, requires_grad=True)
        t2_d = torch.randn(4, 4, device=DEVICE, dtype=dtype, requires_grad=True)
        test_op(f"addcmul_ {dtype}", lambda x, t1, t2: x.clone().addcmul_(t1, t2, value=0.5), x_d, t1_d, t2_d)
    except Exception as e:
        print(f"  addcmul_ {dtype}: SKIP ({e})")

# CPU
print("\n=== CPU ===")
x_cpu = torch.randn(4, 4, requires_grad=True)
t1_cpu = torch.randn(4, 4, requires_grad=True)
t2_cpu = torch.randn(4, 4, requires_grad=True)
test_op("addcmul_ CPU",
        lambda x, t1, t2: x.clone().addcmul_(t1, t2, value=0.5), x_cpu, t1_cpu, t2_cpu)
