# `tf.math.cumsum`/`tf.math.cumprod` crash with `jit_compile=True` on complex dtypes

> **Status: CONFIRMED** — Filed as [upstream issue #117546 (anonymized)]. Labeled `type:bug`, assigned.

## Summary

`tf.math.cumsum` and `tf.math.cumprod` crash during XLA compilation when the input tensor has a complex dtype (`complex64` or `complex128`). The eager path works correctly.

The crash occurs in the HLO lowering layer: XLA does not have a native scan implementation for complex types, and the fallback decomposition is not wired up for `cumsum`/`cumprod`.

## Minimal Reproducer

```python
import tensorflow as tf

x = tf.constant([1.0 + 2.0j, 3.0 + 4.0j, 5.0 + 6.0j], dtype=tf.complex64)

# Eager: works
print(tf.math.cumsum(x).numpy())

# XLA: crashes
<upstream maintainer>.function(jit_compile=True)
def cumsum_xla(x):
    return tf.math.cumsum(x)

cumsum_xla(x)  # XLA compilation failure
```

## Trigger Conditions

| Condition | Required |
|-----------|----------|
| `jit_compile=True` | Yes |
| dtype | complex64 or complex128 |
| op | `tf.math.cumsum` or `tf.math.cumprod` |

Real-valued dtypes (float32, float64, int32, etc.) work correctly under XLA.

## Root Cause

XLA's HLO lowering for cumulative scan ops assumes real-valued types. The scan reduction function (addition for cumsum, multiplication for cumprod) is not instantiated for complex element types, causing the XLA compiler to fail during lowering.

## Environment

- TensorFlow 2.21.0 (stable) + tf-nightly 2.22.0.dev
- Python 3.11, CUDA 12, Ubuntu Linux
