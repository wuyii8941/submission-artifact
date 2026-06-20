#!/usr/bin/env python3
"""Bug: addcmul_/addcdiv_ backward crashes under torch.compile.

Error: RuntimeError: Trying to backward through prims.fma.default
       but no autograd formula was registered.

Works in eager, fails under torch.compile (both inductor and aot_eager).
Out-of-place torch.addcmul() works fine.
"""
import torch

def fn(x, t1, t2):
    y = x.clone()
    y.addcmul_(t1, t2, value=0.5)
    return y

x = torch.randn(4, 4, device='cuda', requires_grad=True)
t1 = torch.randn(4, 4, device='cuda', requires_grad=True)
t2 = torch.randn(4, 4, device='cuda', requires_grad=True)

# Eager: OK
x_e, t1_e, t2_e = [t.detach().clone().requires_grad_(True) for t in [x, t1, t2]]
fn(x_e, t1_e, t2_e).sum().backward()
print(f"Eager: OK")

# Compiled: CRASH
torch._dynamo.reset()
x_c, t1_c, t2_c = [t.detach().clone().requires_grad_(True) for t in [x, t1, t2]]
compiled = torch.compile(fn, backend='inductor')
out = compiled(x_c, t1_c, t2_c)
out.sum().backward()  # RuntimeError: Trying to backward through prims.fma.default
