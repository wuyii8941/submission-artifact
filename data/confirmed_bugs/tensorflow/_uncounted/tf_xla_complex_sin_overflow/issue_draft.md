# XLA `tf.math.sin` / `tf.math.cos` returns inf/nan for complex inputs with large imaginary part, even when the correct result is representable

> **Status: SUPPORTING / PENDING** — This remains a separate XLA complex special-function candidate. Do not cite TensorFlow issue #116943 for this finding; #116943 is the SELU/ELU `exp(x)-1` cancellation issue.

## Summary

`tf.math.sin` and `tf.math.cos` with `jit_compile=True` return `inf`/`nan` for complex64 inputs with imaginary parts around ±88, even though the mathematically correct result is well within float32 range. Eager mode returns the correct result.

## Reproduction

```python
import tensorflow as tf
import numpy as np

z = tf.constant(np.complex64(0-88j).reshape(1, 1))

print(tf.math.sin(z).numpy())
# [[-8.2581813e+37j]]  ← correct (eager)

print(tf.function(tf.math.sin, jit_compile=True)(z).numpy())
# [[(nan-infj)]]  ← wrong (XLA)
```

Both CPU and GPU XLA produce the wrong result.

## Scale of the problem

`sin` is affected starting at |Im(z)| = 88 for negative imaginary part, and |Im(z)| = 89 for both signs:

| z | Eager (correct) | XLA (wrong) | Correct result representable? |
|---|---|---|---|
| `0+88j` | `8.258e37j` | `8.258e37j` | Yes — XLA OK here |
| `0-88j` | `-8.258e37j` | **`nan-infj`** | Yes (< 3.4e38) |
| `1-88j` | `6.949e37-4.462e37j` | **`inf-infj`** | Yes |
| `0+89j` | `2.245e38j` | **`nan+infj`** | Yes |
| `0-89j` | `-2.245e38j` | **`nan-infj`** | Yes |
| `1+89j` | `1.889e38+1.213e38j` | **`inf+infj`** | Yes |

`cos` is also affected:

| z | Eager (correct) | XLA (wrong) |
|---|---|---|
| `0-88j` | `8.258e37+0j` | **`inf+nanj`** |
| `0+89j` | `2.245e38-0j` | **`inf+nanj`** |
| `0-89j` | `2.245e38+0j` | **`inf+nanj`** |
| `1+89j` | `1.213e38-1.889e38j` | **`inf-infj`** |

Values at |Im(z)| ≤ 87 work correctly in both eager and XLA.

## Root cause

`sin(x + iy) = sin(x)·cosh(y) + i·cos(x)·sinh(y)`

For y = ±88: `cosh(88) ≈ sinh(88) ≈ 8.26e37`. These are representable in float32, and the final products `sin(x)·cosh(y)` are also representable since |sin(x)| ≤ 1.

However, XLA's complex sin/cos implementation likely computes intermediate values via `exp`: `sinh(y) = (exp(y) - exp(-y)) / 2`. For |y| = 88, `exp(88) ≈ 1.65e38` — still representable but close to float32 max. The asymmetry (`sin(0+88j)` works but `sin(0-88j)` doesn't) suggests that one code path computes `exp(88)` successfully while the other triggers an overflow in the intermediate decomposition, possibly due to sign handling in the `exp(iz)` expansion.

Eager mode avoids this, likely by computing `sinh`/`cosh` directly rather than via `exp(iz)`.

## Environment

- TensorFlow 2.22.0-dev20260424 (nightly)
- Both CPU and GPU XLA affected
- Eager mode is NOT affected
