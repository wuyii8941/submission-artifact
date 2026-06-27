# `torch.compile` produces wrong gradients for `torch.slice_scatter` backward

## 🐛 Describe the bug

`torch.compile(backend='inductor')` computes incorrect gradients for `torch.slice_scatter` with respect to the input tensor `x`. The gradient of positions **not** overwritten by the `src` slice is silently zeroed out (or set to all-ones, depending on the slice position), instead of correctly reflecting that those positions pass through unchanged.

This is a **silent correctness bug** — no error is raised, but any training loop using `torch.slice_scatter` under `torch.compile` will receive wrong gradients, potentially causing training to diverge without warning.

## Minimal repro

```python
import torch

x = torch.tensor([1., 2., 3., 4., 5.], device='cuda', requires_grad=True)
y = torch.tensor([10., 20.], device='cuda', requires_grad=True)

def f(x, y):
    return torch.slice_scatter(x, y, dim=0, start=0, end=2).sum()

# Eager — correct
f(x, y).backward()
print(f"Eager x.grad = {x.grad}")
# tensor([0., 0., 1., 1., 1.])  ✅  positions 0:2 overwritten by y → grad=0

# Inductor — wrong
x2 = x.detach().clone().requires_grad_(True)
y2 = y.detach().clone().requires_grad_(True)
torch._dynamo.reset()
fn = torch.compile(f, backend='inductor')
fn(x2, y2).backward()
print(f"Inductor x.grad = {x2.grad}")
# tensor([0., 0., 0., 0., 0.])  ❌  positions 2:5 should be 1, not 0
```

## Expected behavior

`slice_scatter(x, y, dim=0, start=0, end=2)` copies `y` into positions `[0:2]` of `x`, while positions `[2:5]` pass through from `x` unchanged. Therefore:

- `x.grad[0:2]` should be `0` (these positions are replaced by `y`)
- `x.grad[2:5]` should be `1` (these positions flow through to the output)

Eager mode computes this correctly. Inductor zeros out the entire `x.grad`.

## Additional observations

The bug is **not limited to `start=0`**. Different slice positions produce different (but still wrong) gradient patterns:

```python
# start=0, end=2: eager=[0,0,1,1,1]  inductor=[0,0,0,0,0]  ← zeros everywhere
# start=2, end=4: eager=[1,1,0,0,1]  inductor=[1,1,1,1,1]  ← ones everywhere
# start=3, end=5: eager=[1,1,1,0,0]  inductor=[1,1,1,1,1]  ← ones everywhere
```

The pattern suggests the inductor backward either ignores the slice indices entirely, or applies an incorrect mask.

The bug also reproduces on **CPU backend** (not just CUDA), and on **multi-dimensional tensors** (e.g., 2D with `dim=1`).

## Versions

- **PyTorch 2.11.0+cu126**: bug present
- **PyTorch 2.12.0.dev20260410+cu126 (nightly)**: bug present
- CUDA 12.6, Tesla T4
- OS: Ubuntu 20.04, Python 3.11

cc <upstream maintainer> <upstream maintainer>
