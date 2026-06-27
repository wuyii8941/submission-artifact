# Bug 034: addcmul_/addcdiv_ backward crashes under torch.compile

## Summary
`torch.compile(fn)` where `fn` uses `Tensor.addcmul_()` or `Tensor.addcdiv_()` crashes on backward with `RuntimeError: Trying to backward through prims.fma.default but no autograd formula was registered`.

## Category
Crash bug (missing prims autograd formula)

## Root cause
When AOT autograd functionalizes the in-place `addcmul_()` op, it decomposes it into `prims.fma.default` (fused multiply-add). However, `prims.fma.default` does not have a registered autograd formula, so backward fails. The out-of-place `torch.addcmul()` follows a different decomposition path that avoids `prims.fma` and works correctly.

## Pattern
| Op | Forward | Backward | Status |
|----|---------|----------|--------|
| `Tensor.addcmul_(t1, t2, value=v)` | OK | CRASH | **BUG** |
| `Tensor.addcdiv_(t1, t2, value=v)` | OK | CRASH | **BUG** |
| `torch.addcmul(x, t1, t2, value=v)` | OK | OK | Fine |
| `torch.addcdiv(x, t1, t2, value=v)` | OK | OK | Fine |
| `x + v * t1 * t2` (manual) | OK | OK | Fine |

## Trigger conditions
- In-place `addcmul_()` or `addcdiv_()` on a tensor that requires grad
- `torch.compile` with ANY backend (inductor, aot_eager)
- Backward pass (forward-only works)
- ALL dtypes (float16, bfloat16, float32, float64)
- ALL devices (CUDA and CPU)
- ALL shapes and `value` arguments (including value=0.0)
- Other in-place ops (`lerp_`, `add_`, `sub_`, `addmm_`) are NOT affected

## Minimal reproducer

```python
import torch

def fn(x, t1, t2):
    y = x.clone()
    y.addcmul_(t1, t2, value=0.5)
    return y

x = torch.randn(4, 4, device='cuda', requires_grad=True)
t1 = torch.randn(4, 4, device='cuda', requires_grad=True)
t2 = torch.randn(4, 4, device='cuda', requires_grad=True)

# Eager: OK
fn(x.clone(), t1.clone(), t2.clone()).sum().backward()

# Compiled: CRASH
torch._dynamo.reset()
compiled = torch.compile(fn, backend='inductor')
compiled(x, t1, t2).sum().backward()
# RuntimeError: Trying to backward through prims.fma.default
# but no autograd formula was registered.
```

## Impact
Any code using `addcmul_()` or `addcdiv_()` with gradient tracking under `torch.compile` will crash. These ops are commonly used in:
- Custom optimizer implementations (Adam uses both `addcmul_` and `addcdiv_`)
- Numerical algorithms that fuse multiply-accumulate patterns
- Performance-critical gradient update loops

## Workaround
Use the out-of-place equivalent:
```python
# Instead of:  y.addcmul_(t1, t2, value=0.5)
# Use:         y = torch.addcmul(y, t1, t2, value=0.5)
```

## Verified on
- PyTorch 2.13.0.dev20260501+cu126
- CUDA: Tesla T4 (SM 7.5)
- Also crashes on CPU
