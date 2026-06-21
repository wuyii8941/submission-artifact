# `torch.compile` silently produces all-zero `x.grad` for `diagonal_scatter(...).sum().backward()`

## 🐛 Describe the bug

`torch.compile(backend='inductor')` silently produces `x.grad = 0` for
`torch.diagonal_scatter(x, src, offset).sum().backward()`, while eager mode
correctly produces `x.grad = 1` at all non-diagonal positions (and `0` on the
diagonal, since those positions were overwritten by `src`).

The forward output is correct; only the backward gradient for `x` is wrong.
`src.grad` is computed correctly. `aot_eager` also produces the correct gradient,
confirming the bug is in Inductor's code generation, not in AOTAutograd or the
decomposition.

## Minimal reproducer

```python
import torch

x = torch.randn(8, 8, device='cuda', requires_grad=True)
src = torch.randn(8, device='cuda', requires_grad=True)

def fn(x, src):
    return torch.diagonal_scatter(x, src, 0).sum()

# Eager: correct
x1 = x.clone().detach().requires_grad_(True)
s1 = src.clone().detach().requires_grad_(True)
fn(x1, s1).backward()
print(f"eager x.grad:\n{x1.grad}")
# tensor([[0., 1., 1., 1., 1., 1., 1., 1.],
#         [1., 0., 1., 1., 1., 1., 1., 1.],
#         ...])  # 1 everywhere except diagonal

# Compiled: wrong
torch._dynamo.reset()
x2 = x.clone().detach().requires_grad_(True)
s2 = src.clone().detach().requires_grad_(True)
torch.compile(fn)(x2, s2).backward()
print(f"compiled x.grad:\n{x2.grad}")
# tensor([[0., 0., 0., 0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0., 0., 0., 0.],
#         ...])  # ALL ZEROS — incorrect
```

## Root cause analysis

The backward FX graph for `diagonal_scatter(x, src).sum()` is:

```
expand(tangent_scalar, [N, N])  →  clone  →  diagonal_scatter(clone, zeros)  →  return
```

Inductor lowers `clone(expand(scalar, [N, N]))` into a Triton kernel that
writes a single element to `buf0`, then returns
`reinterpret_tensor(buf0, (N, N), (0, 0), 0)` — a stride-(0, 0) view where
all N² positions alias one memory cell. The subsequent `diagonal_scatter`
kernel writes `0.0` to that single cell (to zero the diagonal), so the entire
gradient reads back as 0.

Correct behavior: `clone` must materialize the expanded tensor into a
contiguous buffer so that `diagonal_scatter` can selectively zero only the
diagonal positions.

This is **not** the same root cause as #<anon> / #<anon> (slice_scatter
backward), which was in AOTAutograd's `meta_slice_scatter` preserving zero
strides. Here `aot_eager` is correct; only Inductor's codegen is wrong.

## Affected configurations

- **Shapes**: all tested sizes (4×4 through 256×256), square and non-square, all diagonal offsets
- **Dtypes**: float32 and float64
- **Devices**: CUDA (tested on Tesla T4)
- **Trigger pattern**: `diagonal_scatter(x, src, offset)` followed directly by `.sum()` (any dim variant)
- Adding a pointwise op between `diagonal_scatter` and `sum` (e.g. `.relu()`, `.mul(2)`) avoids the bug by preventing the problematic fusion

## Characterization

**BUG (x.grad = all zeros)** — ops that preserve the stride-0 expand in backward:

| Pattern | x.grad |
|---------|--------|
| `.sum()` | **all zeros** |
| `.sum(dim=0)`, `.sum(dim=1)`, `.sum(dim=(0,1))` | **all zeros** |
| `.add(1).sum()` | **all zeros** |
| `.clone().sum()` | **all zeros** |
| `.contiguous().sum()` | **all zeros** |
| `.view(64).sum()`, `.reshape(4,16).sum()`, `.flatten().sum()` | **all zeros** |
| `.t().sum()`, `.transpose(0,1).sum()` | **all zeros** |

**Correct** — ops that force materialization of the expanded tensor:

| Pattern | x.grad |
|---------|--------|
| `.mean()` | correct |
| `.relu().sum()` | correct |
| `.mul(2).sum()`, `.neg().sum()` | correct |
| `.abs().sum()`, `.square().sum()` | correct |
| `.exp().sum()`, `.sigmoid().sum()`, `.tanh().sum()` | correct |
| `@ w .sum()` (matmul) | correct |
| `.amax()`, `.norm()`, `.logsumexp(dim=-1).sum()` | correct |

The dividing line: ops whose backward produces a **uniform** gradient (all elements equal) get fused into a stride-0 buffer. Non-trivial pointwise ops or non-uniform reductions produce element-dependent gradients that force Inductor to materialize the full buffer.

## Versions

- **PyTorch**: 2.13.0.dev20260429+cu126
- **Triton**: 3.2.0+git4b3bb1e8
- **CUDA**: 12.6
- **GPU**: Tesla T4 (sm_75)
- **Python**: 3.11

## cc

<upstream maintainer> <upstream maintainer> <upstream maintainer>
