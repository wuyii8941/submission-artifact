## [inductor] `x.to(torch.bfloat16).float()` precision cast silently elided during fusion

### Summary

When a computation graph contains an explicit `fp32 → bfloat16 → fp32` round-trip cast (e.g. `x.to(torch.bfloat16).float()`), Inductor treats the round-trip as a no-op and eliminates both casts during graph lowering. This produces silently incorrect results because the `bfloat16` cast intentionally truncates the mantissa from 23 bits to 7 bits.

The same issue also affects `fp16` round-trips (`x.half().float()`).

### Reproducer

```python
import torch
import torch._dynamo

torch.manual_seed(0)
x = torch.randn(4, 32, 32, device='cuda')
w = torch.randn(32, 32, device='cuda')

def fn(x, w):
    x = torch.matmul(x, w)
    x = x.to(torch.bfloat16).float()  # intentional precision truncation
    x = x * torch.sigmoid(x)
    return x.sum(dim=1)

# Eager (correct)
eager_out = fn(x, w)

# Inductor (incorrect — cast is elided)
torch._dynamo.reset()
compiled_fn = torch.compile(fn, backend='inductor')
inductor_out = compiled_fn(x.clone(), w.clone())

# aot_eager (correct — preserves cast)
torch._dynamo.reset()
aot_fn = torch.compile(fn, backend='aot_eager')
aot_out = aot_fn(x.clone(), w.clone())

print('eager vs inductor: ', (eager_out - inductor_out).abs().max().item())
# => 0.109 (WRONG — should be ~0)
print('eager vs aot_eager:', (eager_out - aot_out).abs().max().item())
# => 0.0 (correct)

# Proof that inductor output matches "no cast at all":
def fn_no_cast(x, w):
    x = torch.matmul(x, w)
    # bf16 cast removed
    x = x * torch.sigmoid(x)
    return x.sum(dim=1)

no_cast_out = fn_no_cast(x, w)
print('inductor vs fn_without_cast:', (inductor_out - no_cast_out).abs().max().item())
# => 1.5e-5 (inductor behaves as if cast never existed)
```

### Evidence from generated Triton code

Setting `TORCH_LOGS=output_code` confirms the cast is removed. The generated Triton kernel header is:

```
# Topologically Sorted Source Nodes: [sigmoid, x_2, sum_1]
# Original ATen: [aten.sigmoid, aten.mul, aten.sum]
```

The `aten._to_copy` nodes for `bfloat16` and `float32` are absent from the fused kernel. The Triton kernel directly loads the `fp32` matmul output and applies `sigmoid → mul → sum` without any intermediate dtype conversion.

### Impact

This affects any user code that uses explicit dtype casts for precision control, which is common in:
- **Mixed-precision training**: `x.to(torch.bfloat16).float()` is used to simulate low-precision forward passes
- **Quantization-aware training**: explicit cast round-trips model quantization noise
- **Numerical stability testing**: intentional precision reduction to test robustness

The diff scales with tensor values — larger matmul outputs (from larger hidden dimensions) produce larger errors:

| Shape | max_diff (bf16) | max_diff (fp16) |
|-------|----------------|----------------|
| (4, 32, 32) | 0.109 | 0.018 |
| (8, 69, 53) | 0.264 | 0.025 |
| (3, 257, 129) | 0.704 | — |

### Additional notes

- **100% reproducible** across all random seeds, all shapes, all config presets (default, aggressive_fusion, no_fusion, freezing)
- **Not affected by** `torch._inductor.config.emulate_precision_casts` — setting it to `True` does not fix the issue
- **`aot_eager` is correct** — the cast is preserved in the AOT Autograd graph, and only eliminated during Inductor lowering/codegen
- The `fp16` round-trip (`x.half().float()`) has the same problem but with smaller magnitude

### Environment

- PyTorch: 2.6.0+cu124
- Triton: 3.2.0
- CUDA: 12.4
- GPU: Tesla T4
- Python: 3.10.19
- OS: Linux 5.4.0-42-generic x86_64

cc <upstream maintainer>/compilers
