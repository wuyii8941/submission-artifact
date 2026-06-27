# CPU XLA silently drops NaN in `reduce_max` / `reduce_min` / `maximum` / `minimum`

## Summary

When compiled with `tf.function(jit_compile=True)` and forced to CPU, TensorFlow
XLA changes NaN semantics for min/max operations:

- `tf.reduce_max([nan, 1, 3, 2])` returns `3.0` under CPU XLA, while eager and
  NumPy return `nan`.
- `tf.reduce_min([nan, 1, 3, 2])` returns `1.0` under CPU XLA, while eager and
  NumPy return `nan`.
- `tf.math.maximum(tf.constant(5.0), tf.constant(nan))` returns `5.0` under CPU
  XLA, while eager returns `nan`.
- `tf.math.minimum(tf.constant(5.0), tf.constant(nan))` returns `5.0` under CPU
  XLA, while eager returns `nan`.

This is a TensorFlow XLA integration / lowering issue rather than an XLA core
limitation: JAX on CPU correctly propagates NaN for the same max/min operations.

## Why this is a bug

TensorFlow eager mode propagates NaN for `reduce_max`, `reduce_min`,
`maximum`, and `minimum`, matching NumPy behavior. TensorFlow previously fixed
NaN propagation for these eager paths in issues such as
tensorflow/tensorflow#<anon> and tensorflow/tensorflow#<anon>. Compiling the same
TensorFlow program with `jit_compile=True` should not silently change this
observable behavior.

NaNs are also realistic inputs: they can arise from upstream operations such as
`0/0`, `inf - inf`, invalid normalization, or unstable training. Dropping them
can mask invalid values and change downstream control flow or indices.

## Minimal reproduction

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU only
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf

print(tf.__version__)

vals = tf.constant([float("nan"), 1.0, 3.0, 2.0], dtype=tf.float32)

print("reduce_max")
print("  eager:", tf.reduce_max(vals).numpy())
print("  xla:  ", tf.function(lambda x: tf.reduce_max(x), jit_compile=True)(vals).numpy())
print("  numpy:", np.max(vals.numpy()))

print("reduce_min")
print("  eager:", tf.reduce_min(vals).numpy())
print("  xla:  ", tf.function(lambda x: tf.reduce_min(x), jit_compile=True)(vals).numpy())
print("  numpy:", np.min(vals.numpy()))

a = tf.constant(5.0, dtype=tf.float32)
b = tf.constant(float("nan"), dtype=tf.float32)

print("maximum")
print("  eager:", tf.math.maximum(a, b).numpy())
print("  xla:  ", tf.function(lambda x, y: tf.math.maximum(x, y), jit_compile=True)(a, b).numpy())

print("minimum")
print("  eager:", tf.math.minimum(a, b).numpy())
print("  xla:  ", tf.function(lambda x, y: tf.math.minimum(x, y), jit_compile=True)(a, b).numpy())
```

Observed on TensorFlow 2.21.0 stable and TensorFlow 2.22.0-dev20260424:

```text
reduce_max
  eager: nan
  xla:   3.0
  numpy: nan
reduce_min
  eager: nan
  xla:   1.0
  numpy: nan
maximum
  eager: nan
  xla:   5.0
minimum
  eager: nan
  xla:   5.0
```

## Downstream impact

The issue can be amplified into a discrete wrong result. For example, NaN
detection through max/min can change a boolean predicate, and reductions feeding
index selection can silently choose a finite element instead of preserving the
invalid value.

```python
data = tf.constant([1.0, float("nan"), 3.0, 2.0], dtype=tf.float32)

def has_nan_via_max(x):
    return tf.math.is_nan(tf.reduce_max(x))

print(has_nan_via_max(data).numpy())  # True
print(tf.function(has_nan_via_max, jit_compile=True)(data).numpy())  # False on CPU XLA
```

## Environment

- TensorFlow 2.21.0 stable: reproduces
- TensorFlow 2.22.0-dev20260424 nightly: reproduces
- CPU XLA only
- GPU XLA propagates NaN for these simple reduce/max/min cases
- JAX 0.9.2 CPU behavior matches NumPy and propagates NaN

## Notes

There is a related but separate GPU issue for `max_pool2d` and
`tensor_scatter_nd_{min,max}` NaN behavior. That issue concerns XLA GPU
propagating NaN where TensorFlow eager GPU ignores it. This issue is the inverse
for CPU XLA simple min/max operations: CPU XLA drops NaN where TensorFlow eager
and NumPy propagate it.
