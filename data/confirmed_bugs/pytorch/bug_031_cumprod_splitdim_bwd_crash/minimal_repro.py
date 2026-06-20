"""
Bug 031: torch.compile crashes on cumprod backward when scan dim >= 8193

torch.compile(backend='inductor') crashes with
  NotImplementedError: NYI TritonSplitDimKernel reductions
when compiling the backward pass of torch.cumprod with scan dimension >= 8193.

Root cause: For dim >= 8193, Inductor uses TritonSplitDimKernel (split-scan
codegen) for the forward pass. The backward pass decomposes cumprod's gradient
into a reduction (argmax over the product prefix), which must also be scheduled
by TritonSplitDimKernel — but this kernel does not yet support reductions.

The forward pass works correctly. Only backward crashes.
cumsum backward does NOT crash (its gradient decomposition avoids reductions).

Trigger conditions:
  - torch.cumprod with scan dim >= 8193
  - requires_grad=True (backward pass needed)
  - Any batch size, any float dtype

Boundary: dim=8192 OK, dim=8193 CRASH

Tested on: PyTorch 2.13.0.dev20260501+cu126, Tesla T4
"""

import torch
import torch._inductor
torch._inductor.config.force_disable_caches = True

# Setup
x = torch.randn(2, 8193, device='cuda', requires_grad=True).abs() + 0.01

# Eager: works fine
x_e = x.clone().detach().requires_grad_(True)
y_e = torch.cumprod(x_e, dim=1).sum()
y_e.backward()
print(f"Eager OK: grad shape={x_e.grad.shape}, grad norm={x_e.grad.norm():.4f}")

# Compiled: crashes on backward
torch._dynamo.reset()
x_c = x.clone().detach().requires_grad_(True)
compiled = torch.compile(lambda x: torch.cumprod(x, dim=1).sum(), backend='inductor')
y_c = compiled(x_c)
try:
    y_c.backward()
    print("Compiled OK (BUG NOT REPRODUCED)")
except NotImplementedError as e:
    print(f"Compiled CRASH: {e}")
except Exception as e:
    print(f"Compiled CRASH (unexpected): {type(e).__name__}: {e}")
