# `torch.compile` crashes on `cumprod` backward when scan dimension >= 8193

## 🐛 Describe the bug

`torch.compile(backend='inductor')` crashes with `NotImplementedError: NYI TritonSplitDimKernel reductions` when backpropagating through `torch.cumprod` with a scan dimension >= 8193.

The forward pass compiles and executes correctly at all sizes. `cumsum` backward also works fine at the same dimension. Only `cumprod` backward is affected, because its gradient decomposition introduces a reduction (argmax) that the TritonSplitDimKernel codegen path does not support — and no fallback to a non-split-dim kernel is attempted.

## Reproducer

**Minimal:**

```python
import torch
torch._inductor.config.force_disable_caches = True

x = torch.randn(2, 8193, device='cuda', requires_grad=True)

# Eager: OK
y = torch.cumprod(x, dim=1).sum()
y.backward()
print("Eager OK:", x.grad.shape)

# Compiled: crashes on backward
torch._dynamo.reset()
x2 = x.clone().detach().requires_grad_(True)
compiled = torch.compile(lambda x: torch.cumprod(x, dim=1).sum(), backend='inductor')
compiled(x2).backward()
# NotImplementedError: NYI TritonSplitDimKernel reductions
```

**Realistic (RetNet-style learnable retention):**

```python
import torch
torch._inductor.config.force_disable_caches = True

seq_len = 8193
x = torch.randn(2, seq_len, 64, device='cuda', requires_grad=True)
gamma = torch.nn.Parameter(torch.sigmoid(torch.randn(64, device='cuda')))

def retention_forward(x, gamma):
    decay = gamma.unsqueeze(0).unsqueeze(0).expand(2, seq_len, 64)
    retention = torch.cumprod(decay, dim=1)
    return (x * retention).sum()

# Eager: OK
retention_forward(x, gamma).backward()
print(f"Eager OK: gamma.grad norm = {gamma.grad.norm():.4f}")

# Compiled: crashes
torch._dynamo.reset()
x2 = x.clone().detach().requires_grad_(True)
gamma2 = gamma.clone().detach().requires_grad_(True)
torch.compile(retention_forward, backend='inductor')(x2, gamma2).backward()
# NotImplementedError: NYI TritonSplitDimKernel reductions
```

## Boundary and scope

The crash boundary is at dim = 8193, where Inductor switches from a single-block scan to the TritonSplitDimKernel (multi-block split-scan) codegen path.

| seq_len | `cumprod` forward | `cumprod` backward | `cumsum` backward |
|---------|-------------------|--------------------|-------------------|
| 4096    | OK                | OK                 | OK                |
| 8192    | OK                | OK                 | OK                |
| 8193    | OK                | **CRASH**          | OK                |
| 16384   | OK                | **CRASH**          | OK                |
| 32768   | OK                | **CRASH**          | OK                |

- Forward pass: works at all sizes (TritonSplitDimKernel handles the scan correctly).
- `cumsum` backward: works at all sizes (its gradient does not require a reduction).
- `cumprod` backward: crashes at dim >= 8193 because its gradient decomposition includes an argmax reduction, which TritonSplitDimKernel does not support. The scheduler does not fall back to a non-split-dim reduction kernel.

All batch sizes and dtypes (float32, float64) are affected.

## Error traceback

```
  File ".../torch/_inductor/codegen/triton_split_scan.py", line 85, in reduction
    raise NotImplementedError("NYI TritonSplitDimKernel reductions")
torch._inductor.exc.InductorError: NotImplementedError: NYI TritonSplitDimKernel reductions
```

## Expected behavior

`torch.compile` should either:
1. Generate correct backward code for `cumprod` at all dimensions (by implementing reduction support in TritonSplitDimKernel), or
2. Fall back to a non-split-dim codegen path for the reduction in the backward graph (as it already does for `cumsum` backward).

## Environment

- PyTorch: 2.13.0.dev20260501+cu126
- GPU: Tesla T4
- CUDA: 12.6
- OS: Linux

cc <upstream maintainer> <upstream maintainer> <upstream maintainer>
