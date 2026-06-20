# `tf.math.sign` returns 0 for small nonzero complex numbers (eager mode)

> **Status: CONFIRMED** — Filed as [upstream issue #116945 (anonymized)], labeled `type:bug`, and assigned to a TensorFlow maintainer.

## Summary

`tf.math.sign` in eager mode incorrectly returns `0` for complex64 inputs with magnitude below ~1.08e-19, even though these are perfectly valid nonzero floating-point numbers well within the normal float32 range. The correct result for any nonzero `z` is `z / |z|` (a unit-magnitude complex number), not `0`.

XLA (`jit_compile=True`), NumPy, and JAX all return the correct result.

Related: #60656 reported the same function returning `0` for *very large* complex inputs. That issue appears to have been fixed (tested with `-3e307+0j`, now returns correct `(-1+0j)`). This issue covers *small* inputs where the root cause is different.

## Reproduction

```python
import tensorflow as tf
import numpy as np

z = tf.constant(np.complex64(1e-20 + 0j).reshape(1, 1))

print(tf.math.sign(z).numpy())
# [[0j]]  ← wrong (eager), should be [[(1+0j)]]

print(tf.function(tf.math.sign, jit_compile=True)(z).numpy())
# [[(1+0j)]]  ← correct (XLA)

print(np.sign(np.complex64(1e-20 + 0j)))
# (1+0j)  ← correct (numpy)
```

## Scale of the problem

The threshold is precisely at `|z| ≈ sqrt(float32_tiny) ≈ 1.084e-19`:

| z | \|z\|² | Eager | XLA | Correct |
|---|---|---|---|---|
| `1.10e-19+0j` | `1.21e-38` (normal) | `(1+0j)` | `(1+0j)` | `(1+0j)` |
| **`1.08e-19+0j`** | **`1.17e-38` (≈ tiny)** | **`0j`** | `(1+0j)` | `(1+0j)` |
| `1e-19+0j` | `1e-38` (subnormal) | **`0j`** | `(1+0j)` | `(1+0j)` |
| `1e-20+0j` | `1e-40` (subnormal) | **`0j`** | `(1+0j)` | `(1+0j)` |
| `1e-30+0j` | `1e-60` (zero) | **`0j`** | `(1+0j)` | `(1+0j)` |
| `0+1e-20j` | `1e-40` | **`0j`** | `1j` | `1j` |
| `-1e-20+0j` | `1e-40` | **`0j`** | `(-1+0j)` | `(-1+0j)` |
| `1e-20+1e-20j` | `2e-40` | **`0j`** | `(0.707+0.707j)` | `(0.707+0.707j)` |

All these values are well within the normal float32 range (tiny ≈ 1.175e-38), so there is no numerical reason to treat them as zero.

## Root cause

The eager kernel computes `|z|² = re² + im²` as a float32 intermediate. When `|z| < sqrt(float32_tiny) ≈ 1.084e-19`, the squared magnitude underflows to subnormal or zero, and the kernel treats this as `z == 0`, returning `sign = 0`.

The fix is to either:
1. Compute `|z|` via `hypot(re, im)` which avoids intermediate underflow/overflow, or
2. Check `re != 0 || im != 0` directly before computing the magnitude

## Environment

- TensorFlow 2.22.0-dev20260424 (nightly)
- Eager mode affected, XLA is correct (except for subnormals < 1.18e-38)
- NumPy and JAX both return correct results
