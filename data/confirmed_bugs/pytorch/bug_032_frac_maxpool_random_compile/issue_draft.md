# `torch.compile` produces wrong results for `fractional_max_pool2d` when `_random_samples` is not provided

## 🐛 Describe the bug

`torch.compile(backend='inductor')` produces incorrect output for `torch.nn.functional.fractional_max_pool2d` when the internal random sampling path is used (i.e., `_random_samples` is not explicitly passed). The compiled version selects **different pooling indices** than eager mode, returning wrong values.

This is **not** a numerical precision issue — the compiled kernel picks entirely different source elements. With `_random_samples` explicitly provided, the results match exactly.

## Reproducer

```python
import torch
import torch._inductor
torch._inductor.config.force_disable_caches = True

x = torch.randn(1, 3, 16, 16, device='cuda')

# Eager
torch.manual_seed(42); torch.cuda.manual_seed(42)
eager_out, eager_idx = torch.nn.functional.fractional_max_pool2d(
    x, 3, output_size=(4, 4), return_indices=True)

# Compiled
torch._dynamo.reset()
torch.manual_seed(42); torch.cuda.manual_seed(42)
comp_out, comp_idx = torch.compile(
    lambda x: torch.nn.functional.fractional_max_pool2d(
        x, 3, output_size=(4, 4), return_indices=True),
    backend='inductor'
)(x)

print(f"Index mismatches: {(eager_idx != comp_idx).sum().item()}/{eager_idx.numel()}")
print(f"Value max_diff: {(eager_out - comp_out).abs().max().item():.4f}")
# Index mismatches: 6/48
# Value max_diff: 1.2810
```

**Workaround**: passing `_random_samples` explicitly produces correct results:

```python
samples = torch.rand(1, 3, 2, device='cuda')
# Both eager and compiled produce identical output when _random_samples is given
eager = torch.nn.functional.fractional_max_pool2d(x, 3, output_size=(4, 4), _random_samples=samples)[0]
compiled = torch.compile(
    lambda x: torch.nn.functional.fractional_max_pool2d(
        x, 3, output_size=(4, 4), _random_samples=samples)[0],
    backend='inductor'
)(x)
print((eager - compiled).abs().max().item())  # 0.0
```

## Observations

| Condition | Result |
|-----------|--------|
| Without `_random_samples` (default) | **Wrong**: 6/48 indices differ, `max_diff=1.28` |
| With explicit `_random_samples` | Correct: `max_diff=0.0` |
| Eager same-seed deterministic | Yes |
| Compiled same-seed deterministic | Yes |
| Eager == Compiled (same seed) | **No** — different indices selected |
| Multiple calls to compiled fn | Each call uses random state (not baked-in) |

The compiled function does consume random state across calls (different seeds produce different outputs), but the random state is consumed differently than in eager mode, causing the pooling regions to diverge.

## Expected behavior

`torch.compile` should produce the same output as eager mode for the same random state. `fractional_max_pool2d` without `_random_samples` should be functionally equivalent between eager and compiled execution.

## Environment

- PyTorch: 2.13.0.dev20260501+cu126
- GPU: Tesla T4
- CUDA: 12.6
- OS: Linux

cc <upstream maintainer> <upstream maintainer>
