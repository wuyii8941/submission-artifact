# `segment_min`/`segment_max` produces wrong results when segment contains NaN

## Summary

`tf.math.unsorted_segment_min`, `unsorted_segment_max`, `segment_min`, and `segment_max` produce results that are **wrong under any NaN semantics** when a segment contains NaN at certain positions on CPU eager mode. The result depends on where NaN appears in the data, which is incorrect for an order-independent reduction.

For example, `min([1.0, NaN, 3.0])` returns `3.0`, which is wrong regardless of how NaN is handled:
- **IEEE 754 (NaN-propagating)**: should return `NaN`
- **fmin (NaN-suppressing)**: should return `1.0`
- `3.0` is not a valid result under either interpretation

## Reproduction

```python
import tensorflow as tf

ids = tf.constant([0, 0, 0], dtype=tf.int32)

# min([1.0, NaN, 3.0]) should be NaN or 1.0, NOT 3.0
print(tf.math.unsorted_segment_min(
    tf.constant([1.0, float('nan'), 3.0]), ids, 1).numpy())
# [3.0]  — WRONG

# max([3.0, NaN, 1.0]) should be NaN or 3.0, NOT 1.0
print(tf.math.unsorted_segment_max(
    tf.constant([3.0, float('nan'), 1.0]), ids, 1).numpy())
# [1.0]  — WRONG

# Sorted segment ops have the same bug:
print(tf.math.segment_min(
    tf.constant([1.0, float('nan'), 3.0]), ids).numpy())
# [3.0]  — WRONG
```

Reproduces on TF 2.22.0-dev20260503 (nightly), CPU eager mode.

## Scope

The result depends on NaN position — an order-independent reduction should not depend on element order:

| Data | `segment_min` result | Correct (fmin) | Correct (IEEE) |
|---|---|---|---|
| `[1.0, NaN, 3.0]` | **3.0** | 1.0 | NaN |
| `[NaN, 1.0, 3.0]` | 1.0 | 1.0 | NaN |
| `[1.0, 3.0, NaN]` | NaN | 1.0 | NaN |
| `[3.0, NaN, 1.0]` | 1.0 | 1.0 | NaN |
| `[1.0, NaN, 3.0, 2.0]` | **2.0** | 1.0 | NaN |

| Data | `segment_max` result | Correct (fmax) | Correct (IEEE) |
|---|---|---|---|
| `[3.0, NaN, 1.0]` | **1.0** | 3.0 | NaN |
| `[1.0, NaN, 3.0]` | 3.0 | 3.0 | NaN |

### Affected ops

- `tf.math.unsorted_segment_min` / `unsorted_segment_max`
- `tf.math.segment_min` / `segment_max`
- All float dtypes (float32, float64)
- CPU eager mode only — GPU eager and XLA backends produce consistent results

## Root cause

The CPU segment reduction kernel uses a raw comparison like `acc < x ? acc : x` to compute the running minimum. Because IEEE 754 comparisons involving NaN always return `false`, this creates asymmetric behavior:

- `acc=1.0, x=NaN`: `1.0 < NaN` → false → `acc = NaN` (accumulated min lost)
- `acc=NaN, x=3.0`: `NaN < 3.0` → false → `acc = 3.0` (NaN dropped, replaced by next value)

Step by step for `min([1.0, NaN, 3.0])`:
1. acc = 1.0, x = NaN → `1.0 < NaN` = false → acc = NaN (correct min 1.0 is lost)
2. acc = NaN, x = 3.0 → `NaN < 3.0` = false → acc = 3.0 (NaN discarded)
3. Final: 3.0 — wrong

Note: `tf.reduce_min` and `tf.minimum` correctly handle NaN (both propagate NaN symmetrically). Only the segment reduction kernels have this bug. GPU eager returns the correct fmin result (1.0) for the same inputs, indicating the GPU kernel uses a NaN-aware comparison.

## Impact

This is not a NaN-semantics debate — the result `3.0` for `min(1.0, NaN, 3.0)` is objectively wrong. The reduction is supposed to be order-independent, but the result depends on the position of NaN in the input.

## Environment

- TensorFlow 2.22.0-dev20260503
- CPU eager mode (no XLA)
