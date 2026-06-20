# `torch.compile` + `torch._higher_order_ops.map`: Inductor generates invalid C++ wrapper code

**Filed:** <upstream issue, anonymized for review>

## Summary

`torch.compile(fullgraph=True)` with the Inductor backend crashes when compiling a function that uses `torch._higher_order_ops.map`. The generated C++ wrapper code is syntactically invalid and fails compilation with g++.

- Eager mode: works fine
- `aot_eager` backend: works fine
- Inductor backend: **CRASHES** (CppCompileError)

Crash occurs even in pure inference mode (no grad, no backward).

## Minimal reproducer

```python
import torch
import torch.nn.functional as F

xs = torch.randn(8, 64, device='cuda')

def body_fn(x):
    return F.gelu(x)

def fn(xs):
    return torch._higher_order_ops.map(body_fn, xs).sum()

torch._dynamo.reset()
torch.compile(fn, fullgraph=True)(xs)  # CppCompileError
```

## Root cause

Inductor codegen produces invalid C++ for the `map` higher-order op wrapper path.
