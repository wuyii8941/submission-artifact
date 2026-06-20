# `torch.compile` silently produces wrong `x.grad` for `select_scatter(x, src, dim, idx).sum()` — Inductor backward is incorrect

## Summary

For the pattern
```python
def fn(x, src):
    return torch.select_scatter(x, src, dim=1, index=3).sum()
```
`torch.compile(fn, backend="inductor")` produces **correct forward** but **silently wrong `x.grad`** — the gradient w.r.t. the `self` argument is all-ones instead of ones-with-zero-at-idx.

- `src.grad` is correct.
- `aot_eager` backend produces the correct `x.grad` — the bug is **Inductor-specific**.
- Affects **both CPU and CUDA** Inductor paths.
- Reproduces across ALL shapes (2D, 3D+), ALL dims, ALL dtypes (fp16/fp32/fp64).
- Any pointwise op between `select_scatter` and `sum` (relu, sigmoid, clamp, etc.) breaks the fusion and prevents the bug.

## Reproducer

```python
import torch
import torch._dynamo

torch._dynamo.reset()
x = torch.randn(4, 8, device='cuda', requires_grad=True)
src = torch.randn(4, device='cuda', requires_grad=True)

# Eager
x_e, s_e = x.detach().clone().requires_grad_(True), src.detach().clone().requires_grad_(True)
torch.select_scatter(x_e, s_e, dim=1, index=3).sum().backward()

# Compiled (Inductor)
torch._dynamo.reset()
x_c, s_c = x.detach().clone().requires_grad_(True), src.detach().clone().requires_grad_(True)
torch.compile(lambda x, s: torch.select_scatter(x, s, dim=1, index=3).sum())(x_c, s_c).backward()

print("x.grad[0]:")
print(f"  eager:    {x_e.grad[0].tolist()}")   # [1, 1, 1, 0, 1, 1, 1, 1]
print(f"  compiled: {x_c.grad[0].tolist()}")   # [1, 1, 1, 1, 1, 1, 1, 1]  <-- WRONG
print(f"  expected: ones with ZERO at index=3")
print(f"x.grad max_diff = {(x_e.grad - x_c.grad).abs().max().item()}")
```

Output:
```
x.grad[0]:
  eager:    [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]
  compiled: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  <-- WRONG
  expected: ones with ZERO at index=3
x.grad max_diff = 1.0
```

## Expected vs. Actual

Mathematical expectation:
```
out = select_scatter(x, src, dim=1, index=3)
    = x  with x[:, 3] replaced by src
L   = out.sum()

dL/dx[i, j] = 1.0   for j ≠ 3
            = 0.0   for j == 3  (position overwritten by src)
dL/dsrc     = 1.0   (always correct)
```

Eager produces the correct gradient. Inductor produces `x.grad = all-ones`, ignoring the zero at the scattered index.

## Root cause

Same mechanism as #179561 (slice_scatter) and #182012 (diagonal_scatter):

The backward formula for `select_scatter` w.r.t. `self` is:
```
grad_x = select_scatter(tangent, zeros_like(src), dim, idx)
```
When `tangent` comes from `sum().backward()`, it is an `expand` with stride-0:
```
tangent: f32[4, 8][0, 0]  (scalar expanded to [4, 8])
```
Inductor trusts the stride-0 annotation on the `select_scatter` output, collapses the iteration space, and the index check that should place a zero at position `idx` is evaluated at a constant coordinate (0), causing the zero to be misplaced or the entire tangent to pass through unmodified.

## Characterization

### Inductor-specific (aot_eager is correct)
```
aot_eager: x.grad diff = 0.000000e+00  OK
inductor:  x.grad diff = 1.000000e+00  BUG
```

### CPU + CUDA both affected
```
cpu:   x.grad diff = 1.000000e+00  BUG
cuda:  x.grad diff = 1.000000e+00  BUG
```

### Intermediate op prevents bug (breaks fusion)
```
select_scatter → sum                       diff=1.000000e+00  BUG
select_scatter → relu → sum                diff=0.000000e+00  OK
select_scatter → *2+1 → sum               diff=0.000000e+00  OK
select_scatter → sigmoid → sum            diff=0.000000e+00  OK
select_scatter → clamp → sum              diff=0.000000e+00  OK
```

### Shape/dim/dtype sweep
```
shape=(4, 8)         dim=1 idx=3 dtype=float32   diff=1.0e+00  BUG
shape=(8, 16)        dim=0 idx=4 dtype=float32   diff=1.0e+00  BUG
shape=(3, 7, 13)     dim=1 idx=5 dtype=float32   diff=1.0e+00  BUG
shape=(3, 7, 13)     dim=2 idx=6 dtype=float32   diff=1.0e+00  BUG
shape=(2, 16, 32)    dim=1 idx=5 dtype=float32   diff=1.0e+00  BUG
shape=(4, 8)         dim=1 idx=3 dtype=float64    diff=1.0e+00  BUG
shape=(4, 8)         dim=1 idx=3 dtype=float16    diff=1.0e+00  BUG
```

## Relationship to other issues

This is the same vulnerability family as:
- **#179561** (`slice_scatter` backward stride-0) — same root cause, different scatter variant
- **#182012** (`diagonal_scatter` backward stride-0) — same root cause, different scatter variant
- **#178995** (scatter ops on expanded stride-0 tensor in backward graph) — broader framing

All share the root cause: `*_scatter` ops receiving an expanded (stride-0) tensor as `self` in the backward graph, with Inductor trusting the stride annotation to collapse the iteration space.

## Environment

```
PyTorch: 2.7.0.dev20250310+cu124
GPU:     Tesla T4, sm_75
CUDA:    12.2
OS:      Linux 5.4.0-42-generic
```

cc <upstream maintainer> <upstream maintainer> <upstream maintainer> <upstream maintainer>
