# `torch.compile` produces wrong results when fusing `adaptive_avg_pool2d` with `flatten + sum`

## Describe the bug

`torch.compile(backend='inductor')` silently produces incorrect results when `adaptive_avg_pool2d` is fused with downstream `flatten` + `sum` operations. The bug triggers when the **channel dimension is not a power of 2** (e.g., C=33, 65, 129).

This is a **silent correctness bug** — no error is raised, but the fused kernel computes wrong output values. The `adaptive_avg_pool2d` kernel alone produces correct results; the error is introduced during fusion with the subsequent reduction.

## Minimal repro

```python
import torch

def f(x):
    y = torch.nn.functional.adaptive_avg_pool2d(x, 7)
    return y.flatten(1).sum(dim=-1)

torch.manual_seed(42)
x = torch.randn(2, 33, 8, 8, dtype=torch.float64, device='cuda')

# Eager — correct
print(f"Eager: {f(x).tolist()}")
# [-52.27041819362077, 30.75847231895647]

# Inductor — wrong
torch._dynamo.reset()
compiled = torch.compile(f, backend='inductor')(x)
print(f"Inductor: {compiled.tolist()}")
# [-52.27041819362077, 34.325837206874446]  ← second element is wrong

print(f"Max diff: {(f(x) - compiled).abs().max().item()}")
# 3.567365  ← catastrophic for fp64
```

## Key observations

**1. Pool alone is correct; fusion introduces the error:**

```python
# These are all correct (diff ≈ 0):
torch.compile(lambda x: F.adaptive_avg_pool2d(x, 7))(x)           # pool only
torch.compile(lambda x: F.adaptive_avg_pool2d(x, 7).sum())(x)     # pool + sum
torch.compile(lambda x: F.adaptive_avg_pool2d(x, 7).flatten(1))(x) # pool + flatten

# This is WRONG (diff ≈ 3.6):
torch.compile(lambda x: F.adaptive_avg_pool2d(x, 7).flatten(1).sum(-1))(x)
```

**2. Channel dimension determines whether the bug triggers:**

| C (channels) | Power of 2? | Diff |
|:---|:---|:---|
| 32 | yes | 0 |
| **33** | **no** | **7.37** |
| 48 | no (but 16×3) | 0 |
| 64 | yes | 0 |
| **65** | **no** | **4.71** |
| 128 | yes | 0 |
| **129** | **no** | **6.45** |

**3. Spatial dimensions also matter:**

With C=33, output_size=7:
- H=7 (no resize needed): diff=0 ✅
- H=8 (8→7, non-integer stride): diff=7.37 ❌
- H=14 (14→7, integer stride): diff=0 ✅ on pt211, diff≠0 on nightly
- H=15 (15→7, non-integer stride): diff=8.86 ❌

**4. CUDA only — CPU produces correct results.**

**5. Deterministic across random seeds** — not a race condition.

## Impact

`adaptive_avg_pool2d` followed by `flatten + sum` (or similar reductions) is a **standard pattern in classification heads** (e.g., ResNet's global average pooling). Any model with a non-power-of-2 channel count using this pattern under `torch.compile` will silently compute wrong values — affecting both inference accuracy and training gradients.

## Versions

- **PyTorch 2.11.0+cu126**: bug present
- **PyTorch 2.12.0.dev20260410+cu126 (nightly)**: bug present
- CUDA 12.6, Tesla T4
- OS: Ubuntu 20.04, Python 3.11

cc <upstream maintainer> <upstream maintainer>
