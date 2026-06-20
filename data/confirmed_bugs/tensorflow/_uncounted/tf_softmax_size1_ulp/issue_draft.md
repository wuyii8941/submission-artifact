# `tf.nn.softmax` over size-1 axis returns `1.0 - 1 ULP` when batch ≥ 8

## Summary

`tf.nn.softmax(x, axis=-1)` where the last axis has size 1 should return exactly `1.0` for every element — this requires zero floating-point approximation:

```
softmax([v]) = exp(v - v) / sum(exp(v - v)) = exp(0) / exp(0) = 1.0 / 1.0 = 1.0
```

Every intermediate value is exactly representable in IEEE 754. Yet TF returns `0.99999994` (`1.0 - 2^{-24}`, i.e. 1 ULP below 1.0) for elements processed by the vectorized (8-wide) code path, while the scalar remainder is correct.

## Reproduction

```python
import tensorflow as tf

x = tf.ones((16, 1), dtype=tf.float32)

sm = tf.nn.softmax(x, axis=-1)
print(sm.numpy().flatten())
# [0.99999994 0.99999994 ... 0.99999994]  (all 16 wrong)
# Expected: [1. 1. ... 1.]

xla_sm = tf.function(lambda x: tf.nn.softmax(x, axis=-1), jit_compile=True)(x)
print(xla_sm.numpy().flatten())
# [1. 1. ... 1.]  (all correct)
```

## Batch-size boundary pattern

The bug depends on `N` (the outer dimension):

| N | Wrong elements | Pattern |
|---|---|---|
| 1 | 0/1 | scalar path: correct |
| 7 | 0/7 | scalar path: correct |
| **8** | **8/8** | 8-wide SIMD: all wrong |
| 9 | 8/9 | 8 SIMD + 1 scalar tail correct |
| 15 | 8/15 | 8 SIMD + 7 scalar tail correct |
| 16 | 16/16 | 2×8 SIMD: all wrong |
| 17 | 16/17 | 2×8 SIMD + 1 scalar tail correct |
| 128 | 128/128 | 16×8 SIMD: all wrong |
| 129 | 128/129 | 16×8 SIMD + 1 scalar tail correct |

The wrong count is always `⌊N/8⌋ × 8` — exactly the vectorized portion. The `N%8` scalar tail is correct.

## Why this is a code bug, not a precision issue

1. **The math is exact**: `exp(0) = 1.0` and `1.0 / 1.0 = 1.0` are IEEE 754 exact operations. There is no rounding involved.
2. **Scalar tail is correct**: The same softmax implementation processes the `N%8` remainder correctly, returning exactly `1.0`. The vectorized path produces a different (wrong) result for the same computation.
3. **XLA is correct**: `tf.function(jit_compile=True)` returns exactly `1.0` for all elements — proving the correct result is achievable.
4. **Affects both CPU and GPU**: Both devices use the Eigen softmax kernel and both produce the same 8-wide boundary pattern.

## Impact

- `tf.math.floor(tf.nn.softmax(x, axis=-1))` returns `0` instead of `1` for the SIMD rows
- `softmax(x) == 1.0` returns `False` for the SIMD rows
- `sum(softmax(x, axis=-1))` is not exactly `1.0` for the SIMD rows (violates the softmax invariant)
- `tf.keras.ops.softmax` is also affected (delegates to the same kernel)

## Environment

- TensorFlow 2.22.0-dev20260421 (nightly)
- Also confirmed on TensorFlow 2.21.0 (stable)
- CPU: x86_64 with AVX2/AVX-512
- GPU: NVIDIA T4 (CUDA 12)
- Both CPU and GPU affected; only XLA (`jit_compile=True`) is correct
