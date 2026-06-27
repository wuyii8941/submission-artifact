# `torch.compile` crashes on cast to `float8_e4m3fn` on GPUs without native FP8 support

## 🐛 Describe the bug

Inductor generates Triton code using hardware-specific float8 types (`fp8e4nv`) without verifying GPU compute capability >= 8.9. On GPUs like T4, V100, or A100, the generated Triton kernel fails with "type fp8e4nv not supported in this architecture".

Eager mode handles these conversions correctly via CUDA kernels.

## Filed

- Issue: <upstream issue, anonymized for review>
- Developer confirmed: <upstream issue, anonymized for review>

## Affected formats

- `torch.float8_e4m3fn`
- `torch.float8_e4m3fnuz`
- `torch.float8_e5m2fnuz`

## Minimal reproducer

```python
import torch

<upstream maintainer>.compile(backend="inductor")
def f(x):
    return x.to(torch.float8_e4m3fn)

x = torch.randn(16, device="cuda")
f(x)  # crashes on GPUs with compute capability < 8.9
```
