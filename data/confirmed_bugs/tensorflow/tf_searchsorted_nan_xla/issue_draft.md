# `tf.searchsorted(side='right')` returns wrong position for NaN under XLA

## Summary

`tf.searchsorted` with `side='right'` returns different positions for NaN search values depending on execution mode:
- **Eager**: returns `5` (NaN treated as greater than all values) — matches NumPy
- **XLA** (`jit_compile=True`): returns `0` (NaN treated as less than all values)

## Reproduction

```python
import tensorflow as tf
import numpy as np

sorted_seq = tf.constant([[1.0, 3.0, 5.0, 7.0, 9.0]], dtype=tf.float32)
nan_val = tf.constant([[float('nan')]], dtype=tf.float32)

def ss_right(s, v):
    return tf.searchsorted(s, v, side='right')

print(ss_right(sorted_seq, nan_val).numpy())
# [[5]]  — eager: NaN > everything (matches NumPy)

print(tf.function(ss_right, jit_compile=True)(sorted_seq, nan_val).numpy())
# [[0]]  — XLA: NaN < everything

print(np.searchsorted([1.0, 3.0, 5.0, 7.0, 9.0], float('nan'), side='right'))
# 5  — NumPy reference
```

Reproduces on TF 2.22.0-dev20260503 (nightly), CPU.

## Scope

| Side | Eager | XLA | NumPy |
|------|-------|-----|-------|
| `'left'` | `0` | `0` | `0` |
| `'right'` | `5` | `0` | `5` |

For `side='left'`, all three agree. The inconsistency is only in `side='right'`.

## Root cause

The TF-XLA bridge (`tensorflow/compiler/tf2xla/kernels/lower_upper_bound_ops.cc`) implements searchsorted as a brute-force O(MN) sum-of-comparisons algorithm using `xla::Compare` with `ComparisonDirection::kGe`. Under IEEE 754 partial ordering, `NaN >= x` returns false for all `x`, causing the sum to be 0. The eager CPU kernel uses `std::upper_bound` with `<`, where different iteration patterns lead to NaN being placed at the end.

## Expected behavior

XLA-compiled `searchsorted(side='right')` should match eager and NumPy: NaN should return the end-of-array position.

## Environment

- TensorFlow 2.22.0-dev20260503
- CPU only
