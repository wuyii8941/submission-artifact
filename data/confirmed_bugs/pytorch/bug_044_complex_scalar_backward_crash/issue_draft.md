# `torch.compile` backward crashes when complex tensor is multiplied by Python complex literal

## 🐛 Describe the bug

`torch.compile(backend="inductor")` crashes during backward when a complex-dtype tensor is multiplied or divided by a **Python complex literal** (e.g., `x * (2.0 + 1.0j)`). The forward pass succeeds, but the backward graph contains a raw Python `complex` object that Inductor codegen attempts to call `.get_name()` on, expecting an IR node.

The bug is **Inductor-specific** — it reproduces with `backend="inductor"` but NOT with `aot_eager`, indicating the issue is in Inductor's lowering/codegen for the backward graph, not in AOT Autograd.

## Minimal reproducer

```python
import torch

x = torch.randn(4, dtype=torch.complex64, device="cuda", requires_grad=True)

def fn(x):
    return x * (2.0 + 1.0j)

# Eager: works
fn(x).abs().sum().backward()
print(x.grad)  # OK

# Compiled: crashes on backward
x2 = x.detach().clone().requires_grad_(True)
torch.compile(fn, backend="inductor")(x2).abs().sum().backward()
```

## Error message

```
torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
InductorError: AttributeError: 'complex' object has no attribute 'get_name'
```

## Backend isolation

| Backend | Forward | Backward |
|---------|---------|----------|
| `eager` | ✅ | ✅ |
| `aot_eager` | ✅ | ✅ |
| `inductor` | ✅ | ❌ **crashes** |

The forward pass compiles fine — the crash is specifically in **Inductor's compilation of the backward graph**. Since `aot_eager` works, the bug is in Inductor's handling of complex scalar constants in the joint graph, not in AOT Autograd's tracing.

## Affected patterns

Any operation that produces a complex scalar constant in the backward graph:

```python
# Multiplication by complex literal — CRASH
x * (2.0 + 1.0j)
x * (1.0 + 0.5j)
x * 1j

# Division by complex literal — CRASH
x / (1.0 + 1.0j)
x / 2j

# Multiple literals — CRASH
x * (2.0 + 1.0j) + x * (0.5 - 0.5j)
```

All crash with the same `'complex' object has no attribute 'get_name'` error.

## Non-triggering patterns

```python
# Real scalar literal: OK (Inductor handles float constants correctly)
x * 2.0           # OK
x * 3             # OK

# Complex tensor (not literal): OK
x * torch.tensor(2.0 + 1.0j, device=x.device)   # OK

# Forward-only (no backward): OK
with torch.no_grad():
    torch.compile(fn)(x)  # OK

# complex128: ALSO CRASH (same bug)
x = torch.randn(4, dtype=torch.complex128, device="cuda", requires_grad=True)
torch.compile(lambda x: x * (1+1j))(x).abs().sum().backward()  # CRASH
```

## Root cause

During backward graph compilation, the complex scalar constant `(2.0+1.0j)` appears as a raw Python `complex` object in the FX graph. Inductor's codegen assumes all constant nodes are either:
1. Real Python scalars (`int`/`float`) — handled by literal embedding
2. `TensorBox` IR nodes — handled via `.get_name()`

A Python `complex` falls into neither category. The codegen path attempts `.get_name()` and hits `AttributeError`. The fix would be to handle `complex` literals similarly to `float` literals in the constant-lowering path.

## Workaround

Wrap complex constants in `torch.tensor()`:

```python
def fn(x):
    # Instead of: x * (2.0 + 1.0j)
    return x * torch.tensor(2.0 + 1.0j, device=x.device)
```

## Practical impact

Complex literals in arithmetic are idiomatic Python and common in signal processing, physics simulations, and any code working with complex-valued neural networks (e.g., complex-valued convolutions, Fourier-space operations). The pattern `x * (a + bj)` is the most natural way to write complex scaling.

## Versions

- PyTorch: 2.13.0.dev20260513+cu126
- Python: 3.11
- CUDA: 12.6
- GPU: Tesla T4
