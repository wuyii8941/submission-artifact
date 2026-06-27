# `tf.pow` for complex64: `pow(0+0j, 0+0j)` returns `NaN+NaNj` in eager mode

## Summary

`tf.pow(0+0j, 0+0j)` returns `NaN+NaNj` in eager mode, while NumPy returns `(1+0j)`. The eager implementation computes complex power via `exp(b * log(a))`, which hits `log(0) = -inf` and produces NaN.

Under XLA (`jit_compile=True`), `tf.pow(0+0j, 0+0j)` correctly returns `1+0j`, matching NumPy.

## Reproduction

```python
import tensorflow as tf
import numpy as np

z = tf.constant([0+0j], dtype=tf.complex64)

print(tf.pow(z, z).numpy())
# [(nan+nanj)]  — eager: wrong

print(tf.function(lambda a, b: tf.pow(a, b), jit_compile=True)(z, z).numpy())
# [(1+0j)]  — XLA: correct

print(np.power(0+0j, 0+0j))
# (1+0j)  — NumPy reference
```

Reproduces on TF 2.22.0-dev20260503 (nightly), CPU.

## Scope

All complex zero variants are affected:

| Base | Exp | Eager | XLA | NumPy |
|------|-----|-------|-----|-------|
| `0+0j` | `0+0j` | `NaN+NaNj` | `1+0j` | `1+0j` |
| `-0+0j` | `0+0j` | `NaN+NaNj` | `1+0j` | `1+0j` |
| `0-0j` | `0+0j` | `NaN+NaNj` | `1+0j` | `1+0j` |

## Root cause

TensorFlow's eager complex `pow` kernel computes `pow(a, b) = exp(b * log(a))`. When `a = 0+0j`, `log(0+0j) = -inf + ...`, and subsequent multiplication produces NaN. The special case `pow(0, 0) = 1` (from IEEE 754 for real numbers and C99 for complex) is not handled.

## Expected behavior

Eager should return `(1+0j)` for `pow(0+0j, 0+0j)`, matching NumPy and XLA.

## Environment

- TensorFlow 2.22.0-dev20260503
- CPU only
