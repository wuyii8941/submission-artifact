# `torch.compile` + `create_graph=True` crashes with donated buffer error on variance normalization

## 🐛 Describe the bug

`torch.compile(fn, backend='inductor')` followed by `torch.autograd.grad(..., create_graph=True)` (or `backward(retain_graph=True)`) crashes with a `RuntimeError` when the compiled function computes variance normalization (`(x - x.mean()) / (x.var() + eps).sqrt()`) at shape `(4, 64)`.

The same code works correctly in eager mode. The compiled function also works with `create_graph=False`.

**Error message:**
```
RuntimeError: This backward function was compiled with non-empty donated buffers
which requires create_graph=False and retain_graph=False.
Please keep backward(create_graph=False, retain_graph=False) across all backward calls,
or set torch._functorch.config.donated_buffer = False to disable donated buffer.
```

## Minimal reproducer

```python
import torch

def var_norm(x):
    return (x - x.mean(-1, keepdim=True)) / (x.var(-1, keepdim=True) + 1e-5).sqrt()

# Eager: works fine
x = torch.randn(4, 64, device='cuda', requires_grad=True)
out = var_norm(x).sum()
g = torch.autograd.grad(out, x, create_graph=True)[0]
print(f"Eager: OK, grad.requires_grad={g.requires_grad}")

# Compiled: crashes
torch._dynamo.reset()
x2 = x.detach().clone().requires_grad_(True)
out2 = torch.compile(var_norm, backend='inductor')(x2).sum()
g2 = torch.autograd.grad(out2, x2, create_graph=True)[0]  # RuntimeError!
```

## Analysis

The issue is specific to the combination of `x.mean()` and `x.var()` used together in the variance normalization pattern. Decomposing the function shows:

| Sub-expression | create_graph=True |
|---|---|
| `x.var(-1, keepdim=True)` alone | ✅ OK |
| `x.mean(-1, keepdim=True)` alone | ✅ OK |
| `x - x.mean(...)` alone | ✅ OK |
| `x / (x.var(...) + eps).sqrt()` alone | ✅ OK |
| `(x - x.mean(...)) / (x.var(...) + eps).sqrt()` | ❌ **CRASH** |
| `(x - x.mean(...)) / (x.std(...) + eps)` (using `std` instead of `var`) | ✅ OK |
| Manual var: `(x - x.mean(...)).pow(2).mean(...)` instead of `x.var(...)` | ✅ OK |

The crash is **shape-dependent** — only `(4, 64)` triggers it among tested shapes:
```
(2, 64): OK    (3, 64): OK    (4, 64): CRASH    (5, 64): OK
(4, 32): OK    (4, 48): OK    (4, 63): OK        (4, 65): OK
(4, 128): OK   (8, 64): OK
```

This suggests the Inductor reduction heuristic donates buffers in the backward pass under a specific code path that only activates at this shape, and that buffer is still needed when `create_graph=True`.

## Workaround

```python
import torch._functorch.config
torch._functorch.config.donated_buffer = False
```

With donated buffers disabled, the compiled function produces correct results matching eager mode.

## Versions

```
PyTorch: 2.13.0.dev20260521+cu126
CUDA: 12.6
Python: 3.11
GPU: Tesla T4
```

cc <upstream maintainer> <upstream maintainer>
