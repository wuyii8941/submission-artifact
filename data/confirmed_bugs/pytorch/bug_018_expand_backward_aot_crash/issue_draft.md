# `torch.compile` backward crashes when compiled function output is an expanded tensor

## 🐛 Describe the bug

`torch.compile` crashes during backward when the compiled function's output is an expanded (stride-0) tensor created from a **computed intermediate** (not a direct view of the input). The error is:

```
RuntimeError: unsupported operation: more than one element of the written-to
tensor refers to a single memory location. Please clone() the tensor before
performing the operation.
```

The bug is in **AOT Autograd's functionalization layer** — it reproduces with `backend="aot_eager"` (no Inductor involved). Functionalization fails to detect that the function output has aliased memory (stride-0 from expand) and does not insert the necessary clone before the backward in-place write.

Eager mode handles this correctly.

## Minimal reproducer

```python
import torch

def fn(x):
    return x.sum(dim=0, keepdim=True).expand_as(x)

x = torch.randn(4, 8, device="cuda", requires_grad=True)

# Eager: works
fn(x).sum().backward()
print(f"Eager grad: {x.grad.shape}")  # torch.Size([4, 8])

# Compiled: crashes
x2 = x.detach().clone().requires_grad_(True)
torch.compile(fn, backend="inductor")(x2).sum().backward()
```

## Error traceback (abbreviated)

```
RuntimeError: unsupported operation: more than one element of the written-to
tensor refers to a single memory location. Please clone() the tensor before
performing the operation.
```

## Backend isolation

| Backend | Result |
|---------|--------|
| `eager` | ✅ works |
| `aot_eager` | ❌ **crashes** |
| `inductor` | ❌ **crashes** |

Since `aot_eager` crashes, the bug is in **AOT Autograd / functionalization**, not in Inductor codegen.

## Affected patterns

The trigger is: **any computation (not a pure view of the input) followed by `expand()` / `broadcast_to()` as the final output of the compiled function**.

```python
# ALL reductions + expand → CRASH
x.sum(dim=0, keepdim=True).expand_as(x)
x.mean(dim=0, keepdim=True).expand_as(x)
x.amax(dim=0, keepdim=True).expand_as(x)
x.amin(dim=0, keepdim=True).expand_as(x)
x.prod(dim=0, keepdim=True).expand_as(x)
x.norm(dim=0, keepdim=True).expand_as(x)
x.std(dim=0, keepdim=True).expand_as(x)
x.var(dim=0, keepdim=True).expand_as(x)
x.logsumexp(dim=0, keepdim=True).expand_as(x)

# Non-reduction computation + expand → ALSO CRASH
x[:1].sin().expand_as(x)
x[:1].softmax(-1).expand_as(x)
(x[:1] * 2 + 1).expand_as(x)

# broadcast_to (equivalent to expand) → CRASH
x.sum(0, keepdim=True).broadcast_to(4, 8)

# Any dim, 2D/3D/ND:
x.sum(dim=1, keepdim=True).expand_as(x)        # 2D, CRASH
x.sum(dim=0, keepdim=True).expand_as(x)        # 3D shape (4,8,16), CRASH
```

## Non-triggering patterns

```python
# Pure view of input + expand: OK (AOT Autograd correctly handles input aliasing)
x[:1].expand_as(x)                     # OK
x.narrow(0, 0, 1).expand_as(x)         # OK

# Expand followed by any materializing op: OK
x.sum(0, keepdim=True).expand_as(x).sin()         # OK
x.sum(0, keepdim=True).expand_as(x) * x           # OK
x.sum(0, keepdim=True).expand_as(x) + 0           # OK
x.sum(0, keepdim=True).expand_as(x).contiguous()  # OK
x.sum(0, keepdim=True).expand_as(x).clone()       # OK

# repeat (copies memory, no aliasing): OK
x.sum(0, keepdim=True).repeat(4, 1)    # OK

# Trivial expand (size 1 → size 1, no actual aliasing): OK
x = torch.randn(1, 8, requires_grad=True)
x.sum(0, keepdim=True).expand_as(x)    # OK (expand is no-op)
```

## Root cause analysis

The key distinction is between **expand of an input view** vs **expand of a computed intermediate**:

- `x[:1].expand_as(x)`: The expand is on a view that traces back to the function input. AOT Autograd's alias analysis recognizes this relationship and handles the stride-0 output correctly.
- `x.sum(0, keepdim=True).expand_as(x)`: The expand is on a **new tensor** produced by the reduction. This intermediate has no alias relationship with the input. When functionalization processes the backward graph, it encounters a stride-0 gradient tensor (inherited from the expanded output) and attempts an in-place write without first cloning it.

## Practical impact

This pattern appears in real code:
- Manual broadcasting in custom normalization layers: `(x - x.mean(dim, keepdim=True).expand_as(x))`
- Attention score broadcasting
- Any pattern where a per-batch/per-feature statistic is broadcast back to full tensor shape

The workaround (adding `+ 0`, `.contiguous()`, or `.clone()` after expand) is non-obvious and adds unnecessary computation.

## Versions

- PyTorch: 2.13.0.dev20260513+cu126
- Python: 3.11
- CUDA: 12.6
- GPU: Tesla T4

cc <upstream maintainer> <upstream maintainer>
