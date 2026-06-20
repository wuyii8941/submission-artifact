# `tf.nn.batch_normalization` returns wrong result under `jit_compile=True` for large-magnitude inputs

## Summary

`tf.nn.batch_normalization` computes incorrect normalized output under XLA compilation when input values have large magnitude, even when the provided mean and variance are exact. Three identical `float32` values `[1e15, 1e15, 1e15]` with mean `1e15` and variance `0` should normalize to `[0, 0, 0]`, but `jit_compile=True` returns `[-1.14e10, -1.14e10, -1.14e10]`.

## Reproduction

```python
import tensorflow as tf
import numpy as np

t = tf.constant([[1e15, 1e15, 1e15]], dtype=tf.float32)
mean = tf.constant([1e15], dtype=tf.float32)
var = tf.constant([0.0], dtype=tf.float32)

# Eager: correct
print(tf.nn.batch_normalization(t, mean, var, None, None, 1e-5).numpy())
# [[0. 0. 0.]]

# XLA: wrong
print(tf.function(lambda x: tf.nn.batch_normalization(x, mean, var, None, None, 1e-5),
                  jit_compile=True)(t).numpy())
# [[-1.1396235e+10 -1.1396235e+10 -1.1396235e+10]]
```

The error scales with input magnitude:

| Input (3 identical values) | Expected | Eager | `jit_compile=True` |
|---|---|---|---|
| `1e5` | 0 | 0 | **0.32** |
| `1e10` | 0 | 0 | **-105,121** |
| `1e15` | 0 | 0 | **-1.14e10** |
| `1e20` | 0 | 0 | **-1.82e14** |

## Root cause

`tf.nn.batch_normalization` computes `x * inv + (-mean * inv)` instead of `(x - mean) * inv`, where `inv = rsqrt(var + eps)`. These are mathematically equivalent but numerically different when XLA uses fused multiply-add (FMA):

```python
inv = tf.math.rsqrt(var + 1e-5)

# Correct: subtract first, then multiply — no cancellation
print(tf.function(lambda x, m: (x - m) * inv, jit_compile=True)(t, mean).numpy())
# [[0. 0. 0.]]

# What BN actually does: multiply first, then subtract — catastrophic cancellation
print(tf.function(lambda x, m: x * inv + (-m * inv), jit_compile=True)(t, mean).numpy())
# [[-1.1396235e+10 -1.1396235e+10 -1.1396235e+10]]
```

XLA's FMA instruction computes `x * inv` at higher intermediate precision, then subtracts the separately-rounded `mean * inv`. When `x ≈ mean` and `inv` is large, this amplifies the rounding difference.

## Impact

This affects any use of `tf.nn.batch_normalization` under `jit_compile=True` where input values are close to the mean at large magnitude. The threshold for visible error is around `|x| > 1e5` with `var ≈ 0`.

In typical neural network training, feature values rarely exceed 1e4, so this primarily affects edge cases. However, it can silently produce incorrect results when toggling `jit_compile` for inference optimization.

## Environment

- TensorFlow 2.22.0-dev20260429 (nightly)
- Tested on CPU
- float32 affected (threshold ~1e5)
