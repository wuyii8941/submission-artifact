# `torch.compile` crashes on `cumsum(x + broadcast_bias)` when scan dim >= 129

## 🐛 Describe the bug

`torch.compile(backend='inductor')` crashes with `TypeError: list indices must be integers or slices, not NoneType` when compiling `torch.cumsum` over a tensor that includes a broadcast addition with a bias vector, and the scan dimension size is >= 129.

The crash is in Inductor's `SplitScan` codegen path. When the scan dimension is <= 128, Inductor uses a single-block scan (works fine). When >= 129, it switches to `SplitScan` (multi-block), which crashes because `TritonSplitScanKernel.initialize_range_tree` sets `tensor_dim=None` for pointwise range trees, but the codegen later indexes a list with this `None` value when loading broadcast variables.

The same crash affects `torch.cumprod`, `torch.logcumsumexp`, and `torch._higher_order_ops.associative_scan`.

## Reproducer

```python
import torch

x = torch.randn(1, 129, 64, device='cuda')
b = torch.randn(64, device='cuda')

# Works
print(torch.cumsum(x + b, dim=1).shape)

# Crashes
print(torch.compile(lambda x, b: torch.cumsum(x + b, dim=1))(x, b).shape)
```

**Error:**
```
torch._inductor.exc.InductorError: TypeError: list indices must be integers or slices, not NoneType
```

## Trigger conditions

The crash requires **both**:
1. A scan op (`cumsum`/`cumprod`/`logcumsumexp`) with scan dim size >= 129
2. A **broadcast** variable in the fused input (e.g., adding a bias vector)

Without broadcast (standalone `cumsum(x)`): compiles fine at any size.
With broadcast but scan dim <= 128: compiles fine (uses single-block scan, not SplitScan).

| Scan dim | `cumsum(x)` | `cumsum(x + bias)` |
|---|---|---|
| 128 | OK | OK |
| **129** | OK | **CRASH** |
| 256 | OK | CRASH |
| 512 | OK | CRASH |

## Root cause

In `torch/_inductor/codegen/triton_split_scan.py`, line 68:

```python
tensor_dim = 0 if prefix_is_reduction(prefix) else None
```

All non-reduction (pointwise) range trees get `tensor_dim=None`. Later, when the scan's `inner_fn` needs to load a broadcast variable from a pointwise dimension, `TritonSymbols.get_block_shape` (triton.py:255) does:

```python
shape[tree_match[0].tensor_dim] = ...  # TypeError: None is not a valid index
```

## Affected ops

All scan-family ops that go through `SplitScan` codegen:
- `torch.cumsum`
- `torch.cumprod`
- `torch.logcumsumexp`
- `torch._higher_order_ops.associative_scan`

## Versions

- PyTorch 2.11.0+cu126: **CRASH**
- PyTorch 2.12.0.dev20260410+cu126 (nightly): **CRASH**
- GPU: NVIDIA Tesla T4 (sm_75), but not GPU-specific (codegen crash before kernel launch)

cc <upstream maintainer> <upstream maintainer> <upstream maintainer>
