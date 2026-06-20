# `torch.compile` crashes on `flip()` applied to 0-dimensional tensors

## 🐛 Describe the bug

`torch.compile(backend="inductor")` crashes with `IndexError: list index out of range` when `flip()` is applied to a 0-dimensional (scalar) tensor. In eager mode, `flip()` on a scalar is a valid no-op (there are no elements along any dimension to reverse).

The bug is in **Inductor's lowering/codegen for flip** — it indexes into the tensor's dimension list without handling the ndim=0 case.

## Minimal reproducer

```python
import torch

x = torch.tensor(3.14, device="cuda")

# Eager: works (flip on scalar is a no-op)
print(x.flip(0))  # tensor(3.14)

# Compiled: crashes
torch.compile(lambda x: x.flip(0), backend="inductor")(x)
```

## Error message

```
InductorError: IndexError: list index out of range
```

## Backend isolation

| Backend | Result |
|---------|--------|
| `eager` | ✅ works |
| `aot_eager` | ✅ works |
| `inductor` | ❌ **crashes** |

The bug is Inductor-specific (codegen/lowering), not in AOT Autograd.

## Practical scenario

This pattern can appear when a reduction produces a scalar and subsequent operations are applied generically:

```python
def fn(x):
    result = x.sum()       # → scalar (0-d tensor)
    return result.flip(0)  # valid in eager, crashes compiled

x = torch.randn(4, 8, device="cuda", requires_grad=True)
fn(x).backward()                                            # Eager: OK
torch.compile(fn, backend="inductor")(x).backward()         # CRASH
```

## Affected variants

```python
x.flip(0)           # CRASH (x is 0-d)
x.flip([0])         # CRASH
torch.flip(x, [0])  # CRASH
```

## Non-triggering

```python
x.flip([])          # OK (empty dims list, no flip needed)
x.flip(0)           # OK when x.ndim >= 1
```

## Workaround

```python
# Unsqueeze before flip, squeeze after:
x.sum().unsqueeze(0).flip(0).squeeze(0)  # OK
```

## Versions

- PyTorch: 2.13.0.dev20260513+cu126
- Python: 3.11
- CUDA: 12.6
- GPU: Tesla T4
