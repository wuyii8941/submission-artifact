## `torch.compile` crashes on `addcmul_`/`addcdiv_` backward — missing autograd formula for `prims.fma`

### 🐛 Describe the bug

`torch.compile` crashes during backward when the forward pass uses `Tensor.addcmul_()` or `Tensor.addcdiv_()`. The error is:

```
RuntimeError: Trying to backward through prims.fma.default but no autograd formula was registered.
Please use register_autograd to add one.
```

The out-of-place equivalents (`torch.addcmul`, `torch.addcdiv`) work correctly. Eager mode works correctly.

### Reproducer

```python
import torch

def fn(x, t1, t2):
    y = x.clone()
    y.addcmul_(t1, t2, value=0.5)
    return y.sum()

x = torch.randn(4, 4, device="cuda", requires_grad=True)
t1 = torch.randn(4, 4, device="cuda", requires_grad=True)
t2 = torch.randn(4, 4, device="cuda", requires_grad=True)

# Eager: OK
fn(x, t1, t2).backward()
print("Eager OK")

# Compiled: CRASH
torch._dynamo.reset()
torch.compile(fn)(x, t1, t2).backward()
# RuntimeError: Trying to backward through prims.fma.default
# but no autograd formula was registered.
```

### Root cause

AOT autograd functionalizes the in-place `addcmul_()` and decomposes it into `prims.fma.default` (fused multiply-add). However, `prims.fma.default` does not have a registered autograd formula, so the backward pass fails. The out-of-place `torch.addcmul()` follows a different decomposition path that avoids `prims.fma` and works correctly.

### Affected ops

| Op | Forward | Backward | Status |
|----|---------|----------|--------|
| `Tensor.addcmul_(t1, t2)` | OK | CRASH | **BUG** |
| `Tensor.addcdiv_(t1, t2)` | OK | CRASH | **BUG** |
| `torch.addcmul(x, t1, t2)` | OK | OK | Fine |
| `torch.addcdiv(x, t1, t2)` | OK | OK | Fine |

### Trigger conditions

- In-place `addcmul_()` or `addcdiv_()` on a tensor that requires grad
- `torch.compile` with **any** backend (both `inductor` and `aot_eager` crash — this is an AOT autograd-level issue)
- Backward pass only (forward compiles and runs correctly)
- All dtypes (float16, bfloat16, float32, float64)
- All devices (CUDA and CPU)
- All shapes (scalar, 2D, 3D — tested from `(1,)` to `(2,4,8)`)
- All `value` arguments (including `value=0.0` and `value=1.0`)
- Other in-place ops (`lerp_`, `add_`, `sub_`, `mul_`, `addmm_`) are NOT affected

### Workaround

Use the out-of-place equivalent:
```python
# Instead of:
y.addcmul_(t1, t2, value=0.5)

# Use:
y = torch.addcmul(y, t1, t2, value=0.5)
```

### Versions

- PyTorch: 2.13.0.dev20260513+cu126
- GPU: Tesla T4 (SM 7.5), also reproducible on CPU
- OS: Linux

cc <upstream maintainer> <upstream maintainer>
