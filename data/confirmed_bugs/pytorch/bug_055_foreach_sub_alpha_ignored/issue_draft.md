# [Inductor] `torch._foreach_sub` silently ignores `alpha` under `torch.compile`

## 🐛 Describe the bug

`torch._foreach_sub(tensors, other_tensors, alpha=k)` silently ignores the `alpha` parameter when compiled with `torch.compile`. Instead of computing `a_i - alpha * b_i`, the compiled code computes `a_i - b_i` (as if `alpha=1`).

The analogous `torch._foreach_add` with `alpha` works correctly under `torch.compile`.

## Reproducer

```python
import torch

def fn(a, b):
    return torch._foreach_sub(a, b, alpha=2.0)

torch.manual_seed(42)
a = [torch.randn(4, device='cuda') for _ in range(3)]
b = [torch.randn(4, device='cuda') for _ in range(3)]

eager = fn(a, b)
compiled = torch.compile(fn)([t.clone() for t in a], [t.clone() for t in b])

for i in range(3):
    print(f"eager[{i}]:    {eager[i].tolist()}")
    print(f"compiled[{i}]: {compiled[i].tolist()}")
    print(f"  match: {torch.equal(eager[i], compiled[i])}")  # False

# compiled output exactly matches alpha=1 (no alpha):
no_alpha = torch._foreach_sub(a, b)
for i in range(3):
    assert torch.equal(compiled[i], no_alpha[i]), "compiled should match no-alpha"
print("\nalpha was silently ignored: compiled == no_alpha for all tensors")
```

Output:
```
eager[0]:    [1.2315161228179932, -0.292142391204834, -1.423011302947998, 2.6724295616149902]
compiled[0]: [0.7127674221992493, 0.9346156120300293, -0.7975307703018188, 1.7607448101043701]
  match: False
...
alpha was silently ignored: compiled == no_alpha for all tensors
```

## Root Cause

In `torch/_inductor/lowering.py`, `_foreach_sub.List` is registered without `allow_alpha=True`:

```python
# line 7953 — _foreach_add correctly passes allow_alpha:
foreach_add_list = register_foreach_pointwise(
    aten._foreach_add.List, add, allow_alpha=True
)

# line 7963 — _foreach_sub is missing allow_alpha:
register_foreach_pointwise(aten._foreach_sub.List, sub)  # ← alpha silently dropped
```

Inside `make_foreach_pointwise` (line 784), the `alpha` value is extracted from kwargs into `scalar_val`. But when `allow_alpha=False` (the default), the `apply_fn` on line 816 calls `pw_fn(*args)` without forwarding `scalar_val`, so the alpha is silently discarded.

## Suggested Fix

```python
register_foreach_pointwise(aten._foreach_sub.List, sub, allow_alpha=True)
```

## Versions

```
PyTorch: 2.13.0.dev20260513+cu126
GPU: NVIDIA Tesla T4
CUDA: 12.6
Triton: 3.3.1
```

cc <upstream maintainer> <upstream maintainer>
