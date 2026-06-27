"""
Bug 036: torch.compile crashes when torch.randperm result is used for fancy indexing.

Root cause: Pattern matcher in joint_graph_passes fails because a 'slice_shape'
argument is expected but not found in match.kwargs.

Error: RuntimeError: Not all inputs to pattern found in match.kwargs.
       Perhaps one of the inputs is unused?
       argnames=['x', 'slice_shape'], match.kwargs={'x': arg0_1}

Location: torch/_inductor/pattern_matcher.py:1530, during joint_graph_passes

Properties:
- Inductor-specific (aot_eager works fine)
- Crashes on both CUDA and CPU
- Not hardware-dependent
- PyTorch 2.13.0.dev20260513+cu126
"""
import torch
import torch._dynamo

# Minimal reproducer
def fn(x):
    idx = torch.randperm(x.shape[0], device=x.device)
    return x[idx]

x = torch.randn(8, 64, device='cuda')

# Eager works
print("Eager:", fn(x).shape)

# aot_eager works
torch._dynamo.reset()
print("aot_eager:", torch.compile(fn, backend='aot_eager')(x).shape)

# Inductor crashes
torch._dynamo.reset()
print("Inductor:", torch.compile(fn, backend='inductor')(x).shape)
