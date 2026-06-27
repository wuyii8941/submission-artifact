"""
Minimal reproducer for bug_011:
torch.compile crashes with TypeError in SplitScan codegen when compiling
cumsum over a fused matmul+bias+pointwise graph.

Trigger: cumsum(sigmoid(x @ w + b), dim=1)  where b is a broadcast bias vector
         and the scan dimension (dim=1) >= 129.

Without bias: OK. With scalar add: OK. With broadcast bias: CRASH.

Error: InductorError: TypeError: list indices must be integers or slices, not NoneType
  (in torch/_inductor/scheduler.py, SplitScan codegen)

Tested on: PyTorch 2.11.0+cu126, 2.12.0.dev20260410+cu126
GPU: NVIDIA Tesla T4
"""

import torch

def fn(x, w, b):
    return torch.cumsum(torch.sigmoid(x @ w + b), dim=1)

H = 129  # scan dim size; 128 is OK, 129 crashes
x = torch.randn(1, H, H, device='cuda')
w = torch.randn(H, H, device='cuda')
b = torch.zeros(H, device='cuda')  # broadcast bias — this is the trigger

# Eager: works
print("Eager:", fn(x, w, b).shape)

# Compiled: crashes
torch._dynamo.reset()
compiled = torch.compile(fn, backend='inductor')
print("Compiled:", compiled(x, w, b).shape)
# -> InductorError: TypeError: list indices must be integers or slices, not NoneType
