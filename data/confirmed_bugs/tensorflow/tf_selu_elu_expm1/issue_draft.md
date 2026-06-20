# `tf.nn.selu` and `tf.nn.elu` return 0 for small negative inputs (catastrophic cancellation in `exp(x) - 1`)

> **Status: CONFIRMED** — Filed as [upstream issue #116943 (anonymized)]. A fix was submitted in [upstream issue #116947 (anonymized)], replacing `exp(x) - 1` with `expm1(x)` in Eigen ELU/SELU kernels.

## Summary

`tf.nn.selu` and `tf.nn.elu` compute `exp(x) - 1` via separate `exp` and subtraction. For small negative x, `exp(x)` rounds to `1.0` in float32, producing `exp(x) - 1 = 0` (catastrophic cancellation). The result is 100% wrong for |x| ≤ 1e-8, and already 1.3% wrong at x = -1e-6.

XLA (`jit_compile=True`) is not affected — it uses `expm1` internally and returns correct results.

## Reproduction

```python
import tensorflow as tf

x = tf.constant([[-1e-10]], dtype=tf.float32)

print(tf.nn.selu(x).numpy())   # [[0.]]  — should be [[-1.758e-10]]
print(tf.nn.elu(x).numpy())    # [[0.]]  — should be [[-1e-10]]

# XLA gives the correct result:
print(tf.function(tf.nn.selu, jit_compile=True)(x).numpy())  # [[-1.758e-10]]
print(tf.function(tf.nn.elu, jit_compile=True)(x).numpy())   # [[-1e-10]]
```

## Scale of the problem

SELU(x) for x < 0 is `scale × alpha × (exp(x) - 1)`. For small negative x, the correct value is approximately `1.7581 × x`.

| x | SELU expected | SELU eager | Relative error |
|---|----------|---------------|----------------|
| -1e-04 | -1.758e-4 | -1.758e-4 | 0.02% |
| -1e-06 | -1.758e-6 | -1.781e-6 | **1.33%** |
| -1e-07 | -1.758e-7 | -2.096e-7 | **19.21%** |
| -1e-08 | -1.758e-8 | **0.0** | **100%** |
| -1e-10 | -1.758e-10 | **0.0** | **100%** |

ELU has the same pattern (with alpha = 1):

| x | ELU expected | ELU eager | Relative error |
|---|----------|---------------|----------------|
| -1e-06 | -1.000e-6 | -1.013e-6 | **1.33%** |
| -1e-07 | -1.000e-7 | -1.192e-7 | **19.21%** |
| -1e-08 | -1.000e-8 | **0.0** | **100%** |

### Gradient is also affected (SELU)

Because the forward pass returns 0 for small negative x, the backward pass takes the wrong branch (`scale` instead of `scale × alpha × exp(x)`), producing ~40% gradient error:

```
x = -1e-07: selu_grad = 1.7581 (correct)
x = -1e-08: selu_grad = 1.0507 (wrong — should be 1.7581, ~40% error)
```

## Root cause

```
exp(-1e-8)     = 1.0       (in float32 — rounds to 1)
exp(-1e-8) - 1 = 0.0       (catastrophic cancellation)
expm1(-1e-8)   = -1e-8     (correct, no cancellation)
```

The fix is to replace `exp(x) - 1` with `expm1(x)` in the Eigen SELU/ELU kernels. XLA already does this, which is why `jit_compile=True` produces correct results.

## Environment

- TensorFlow 2.22.0-dev20260421 (nightly)
- CPU and GPU eager kernels both affected
- XLA is NOT affected
