# `torch.compile` crashes on `scaled_dot_product_attention(...).sum().backward()` when head_dim is not a multiple of 8

**Filed:** <upstream issue, anonymized for review> (Closed — repro bot could not reproduce on H100)

## Summary

`torch.compile(F.scaled_dot_product_attention(q, k, v).sum(), backend="inductor").backward()` crashes with `IndexError: list index out of range` in `sdpa_constraint` whenever `head_dim % 8 != 0`.

The crash occurs because the stride-0 tangent from `sum().backward()` (a scalar expanded to 4D with all-zero strides) gets `unwrap_view()`'d to a 0-d scalar in the alignment check, and `get_size()[-1]` fails on an empty size list.

- `aot_eager` backend works correctly — the bug is **Inductor-specific**.
- Eager mode works correctly for all head_dim values.
- Any intermediate op between SDPA and `sum()` prevents the crash (non-stride-0 tangent).
- head_dim=1 does not crash (likely different SDPA backend dispatch).

## Reproducer

```python
import torch
import torch.nn.functional as F

q = torch.randn(2, 4, 8, 33, device='cuda', requires_grad=True)
k = torch.randn(2, 4, 8, 33, device='cuda', requires_grad=True)
v = torch.randn(2, 4, 8, 33, device='cuda', requires_grad=True)

# Eager — works
F.scaled_dot_product_attention(q, k, v).sum().backward()

# Compiled — crashes
torch._dynamo.reset()
q2 = q.detach().clone().requires_grad_(True)
k2 = k.detach().clone().requires_grad_(True)
v2 = v.detach().clone().requires_grad_(True)
torch.compile(
    lambda q, k, v: F.scaled_dot_product_attention(q, k, v).sum()
)(q2, k2, v2).backward()
# InductorError: LoweringException: IndexError: list index out of range
#   target: aten._scaled_dot_product_efficient_attention_backward.default
```

## Error traceback

```
File ".../torch/_inductor/lowering.py", line 2580, in is_aligned
    return (V.graph.sizevars.size_hint(x.get_size()[-1]) % ALIGNMENT) == 0
                                       ~~~~~~~~~~~~^^^^
torch._inductor.exc.InductorError: LoweringException: IndexError: list index out of range
  target: aten._scaled_dot_product_efficient_attention_backward.default
```

## Root cause

In `lowering.py`, `sdpa_constraint` checks alignment of SDPA backward arguments:

```python
def is_aligned(x):
    return (V.graph.sizevars.size_hint(x.get_size()[-1]) % ALIGNMENT) == 0

def apply_constraint(idx, arg, fx_arg):
    if is_aligned(arg.unwrap_view()):
        ...
```

When the tangent is a stride-0 expanded scalar (from `sum().backward()`):
```
args[0]: ReinterpretView(
    StorageBox(InputBuffer(size=[], stride=[])),  # 0-d scalar
    FixedLayout(size=[2, 4, 8, 33], stride=[0, 0, 0, 0])
)
```

`unwrap_view()` strips the ReinterpretView, returning the scalar StorageBox with `size=[]`. Then `get_size()[-1]` on an empty list raises `IndexError`.

For head_dim values that ARE multiples of 8, SDPA likely dispatches to a different backend (flash attention path) that doesn't go through this constraint check with the stride-0 tensor.

## Affected head_dim values

| head_dim | Status |
|----------|--------|
| 1 | OK (different backend) |
| 2, 3, 4, 5, 6, 7 | CRASH |
| 8 | OK |
| 9-15 | CRASH |
| 16 | OK |
| 17-23 | CRASH |
| 24 | OK |
| ... pattern continues ... |
| Any multiple of 8 | OK |
| Any non-multiple of 8 (except 1) | CRASH |

## Intermediate ops prevent the crash

```
sdpa.sum()              → CRASH  (stride-0 tangent)
sdpa.mean()             → OK     (non-stride-0 tangent)
sdpa.norm()             → OK
sdpa.sum(dim=-1).sum()  → OK     (partial reduction first)
(sdpa @ W).sum()        → OK
```

## Environment

```
PyTorch: 2.7.0.dev20250310+cu124
GPU:     Tesla T4, sm_75
CUDA:    12.2
OS:      Linux 5.4.0-42-generic
```

cc <upstream maintainer> <upstream maintainer> <upstream maintainer> <upstream maintainer>
