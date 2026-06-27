# `tf.debugging.check_numerics` silently passes NaN/Inf under `jit_compile=True`

## Summary

`tf.debugging.check_numerics` is supposed to raise `InvalidArgumentError` when the input tensor contains NaN or Inf. This works correctly in eager mode and under `tf.function(jit_compile=False)`, but under `jit_compile=True` the check is silently skipped — the tensor passes through without any error.

## Reproduction

```python
import tensorflow as tf

t = tf.constant([1.0, 2.0, float('nan')])

# Eager: correctly raises InvalidArgumentError
try:
    tf.debugging.check_numerics(t, "has nan")
except tf.errors.InvalidArgumentError:
    print("eager: raised")  # ✓

# jit_compile=True: silently passes
r = tf.function(lambda v: tf.debugging.check_numerics(v, "has nan"),
                jit_compile=True)(t)
print(r.numpy())  # [1. 2. nan] — no error raised
```

Same behavior with Inf:

```python
t_inf = tf.constant([1.0, float('inf')])

# Eager: raises
try:
    tf.debugging.check_numerics(t_inf, "has inf")
except tf.errors.InvalidArgumentError:
    print("eager: raised")  # ✓

# XLA: silently passes
r = tf.function(lambda v: tf.debugging.check_numerics(v, "has inf"),
                jit_compile=True)(t_inf)
print(r.numpy())  # [1. inf] — no error raised
```

`tf.function(jit_compile=False)` (Grappler) correctly raises in both cases.

## Impact

Code that uses `check_numerics` as a numerical guard will silently lose that protection when switching to `jit_compile=True`. NaN/Inf values propagate undetected through the rest of the computation, producing silently wrong results. This is particularly problematic because `jit_compile=True` is often toggled for inference optimization without expecting semantic changes in debugging behavior.

## Environment

- TensorFlow 2.22.0-dev20260429 (nightly)
- Tested on CPU
