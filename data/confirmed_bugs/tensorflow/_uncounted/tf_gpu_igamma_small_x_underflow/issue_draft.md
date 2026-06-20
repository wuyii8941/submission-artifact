# tf.math.igamma underflows for small positive x on GPU eager

## Summary

For valid positive float32 inputs such as `a = 0.1` and `x = 1e-38`,
TensorFlow GPU eager/Grappler returns:

```text
igamma(a, x)  = 0.0
igammac(a, x) = 1.0
```

TensorFlow XLA, JAX, and SciPy return nontrivial finite values:

```text
igamma(a, x)  ~= 0.00016659
igammac(a, x) ~= 0.9998334
```

This looks like underflow or loss of precision in the TensorFlow GPU eager
kernel for small `x` and small `a`.

## Minimal Reproduction

```python
import tensorflow as tf

a = tf.constant([0.1], dtype=tf.float32)
x = tf.constant([1e-38], dtype=tf.float32)

def lower(a, x):
    return tf.math.igamma(a, x)

def upper(a, x):
    return tf.math.igammac(a, x)

print("TensorFlow:", tf.__version__)

print("igamma eager:   ", lower(a, x).numpy())
print("igamma grappler:", tf.function(lower, jit_compile=False)(a, x).numpy())
print("igamma xla:     ", tf.function(lower, jit_compile=True)(a, x).numpy())

print("igammac eager:   ", upper(a, x).numpy())
print("igammac grappler:", tf.function(upper, jit_compile=False)(a, x).numpy())
print("igammac xla:     ", tf.function(upper, jit_compile=True)(a, x).numpy())
```

Observed on TensorFlow 2.22.0-dev20260424 with GPU execution:

```text
igamma eager:    [0.]
igamma grappler: [0.]
igamma xla:      [0.00016659]

igammac eager:    [1.]
igammac grappler: [1.]
igammac xla:      [0.9998334]
```

Enabling op determinism did not change the result:

```python
tf.config.experimental.enable_op_determinism()
```

## Reference Behavior

SciPy:

```python
import numpy as np
import scipy.special as sp

a = np.float32(0.1)
x = np.float32(1e-38)

print(sp.gammainc(a, x))   # 0.00016659396
print(sp.gammaincc(a, x))  # 0.9998334
```

JAX:

```python
import jax
import jax.numpy as jnp
import jax.scipy.special as jsp

a = jnp.array([0.1], dtype=jnp.float32)
x = jnp.array([1e-38], dtype=jnp.float32)

print(jsp.gammainc(a, x))
print(jsp.gammaincc(a, x))
print(jax.jit(jsp.gammainc)(a, x))
print(jax.jit(jsp.gammaincc)(a, x))
# [0.00016659]
# [0.9998334]
```

## Expected Behavior

`tf.math.igamma(0.1, 1e-38)` should return a finite nonzero value close to
`1.6659e-4`, and `tf.math.igammac(0.1, 1e-38)` should return the corresponding
complement near `0.9998334`.

## Actual Behavior

TensorFlow GPU eager/Grappler returns exact `0.0` and `1.0`, losing the
nontrivial lower-tail probability.

## Why This Matters

`igamma` and `igammac` are used in statistical distributions and tail
probability computations. Small positive `x` and small shape parameters are
valid inputs; silently underflowing to exactly `0` or `1` can change likelihoods,
CDFs, and gradient-sensitive code.

## Relation to Existing igamma Issue

This is distinct from the existing large-argument `igamma` overflow issue. That
case concerns very large `a ~= x` values where the result can leave the
mathematical range `[0, 1]`. This case uses small, finite, valid inputs and
appears to be a small-`x` underflow/precision loss in the GPU eager kernel.

## Environment

- TensorFlow 2.22.0-dev20260424
- dtype: `float32`
- GPU eager/Grappler: underflows to `0.0` / `1.0`
- GPU XLA: matches SciPy/JAX
- JAX 0.9.2 GPU eager/jit: matches SciPy
- SciPy: finite nontrivial lower-tail value

## Recommendation

File in `tensorflow/tensorflow`. This appears to be a TensorFlow GPU kernel
issue rather than an OpenXLA issue, because TensorFlow XLA and JAX agree with
SciPy.
